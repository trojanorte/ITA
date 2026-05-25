"""
Cálculo de eficiência adaptativa municipal frente ao risco climático.

Recorte padrão: municípios do estado de São Paulo.
Método principal: DEA-BCC orientado a outputs.
Modelos auxiliares: Random Forest, árvore de decisão e agrupamento.

Uso básico:
    python eficiencia_adaptativa_sp_sem_ods6.py

Os arquivos XLSX devem estar na mesma pasta do script ou em uma pasta
informada pelo parâmetro --data-dir.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable, Literal

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import linprog
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor, export_text, plot_tree


IDSC_FILE = "Base_de_Dados_IDSC-BR_2025.xlsx"
FLOOD_FILE = (
    "AdaptaBrasil_adaptabrasil_desastres_geo-hidrologicos_indice_de_risco_"
    "para_inundacoes_enxurradas_e_alagamentos_BR_municipio_2015.xlsx"
)
LANDSLIDE_FILE = (
    "AdaptaBrasil_adaptabrasil_desastres_geo-hidrologicos_indice_de_risco_"
    "para_deslizamento_de_terra_BR_municipio_2015.xlsx"
)

IDSC_COLUMNS = [
    "ID",
    "Município",
    "UF",
    "População_2022",
    "Pontuação Indice ODS 2025",
    "Goal 6 Score",
    "Goal 11 Score",
    "Goal 13 Score",
    "Normalizado (0-100): SDG13_PDAR",
    "Normalizado (0-100): SDG6_6_SERV_AG",
    "Normalizado (0-100): SDG6_7_ESG_SAN",
    "Normalizado (0-100): SDG6_8_CLT_DML",
    "Normalizado (0-100): SDG13_4_PREV",
    "Normalizado (0-100): SDG9_1_INFRA",
    "Normalizado (0-100): SDG15_PRTAMB",
]

INPUT_COLS = [
    "risco_inundacao",
    "risco_deslizamento",
    "exposicao_domicilios_risco",
]

OUTPUT_COLS = [
    "saneamento_resiliente",
    "gestao_risco",
    "governanca_investimento_adaptativo",
]

DETAIL_OUTPUT_COLS = [
    "agua",
    "esgoto",
    "tratamento_esgoto",
    "infraestrutura",
    "politica_ambiental",
]

# Variáveis usadas pela IA para interpretar os escores DEA.
# O ODS 6 agregado não entra aqui para evitar sobreposição com água,
# esgoto e tratamento de esgoto, que já aparecem de forma desagregada.
IA_FEATURE_COLS = (
    INPUT_COLS
    + DETAIL_OUTPUT_COLS
    + [
        "risco_composto",
        "porte_populacional_log",
        "Pontuação Indice ODS 2025",
        "Goal 11 Score",
        "Goal 13 Score",
    ]
)

FIG_LABELS = {
    "risco_inundacao": "Risco inundação",
    "risco_deslizamento": "Risco deslizamento",
    "exposicao_domicilios_risco": "Exposição",
    "agua": "Água",
    "esgoto": "Esgoto",
    "tratamento_esgoto": "Trat. esgoto",
    "gestao_risco": "Gestão risco",
    "infraestrutura": "Infraestrutura",
    "politica_ambiental": "Política ambiental",
    "saneamento_resiliente": "Saneamento",
    "governanca_investimento_adaptativo": "Governança",
    "risco_composto": "Risco composto",
    "porte_populacional_log": "Porte populacional",
    "Pontuação Indice ODS 2025": "IDSC 2025",
    "Goal 6 Score": "ODS 6",
    "Goal 11 Score": "ODS 11",
    "Goal 13 Score": "ODS 13",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Calcula eficiência DEA e gera resultados para análise municipal."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path(__file__).resolve().parent,
        help="Pasta com as planilhas XLSX.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Pasta de saída. Padrão: ./resultados.",
    )
    parser.add_argument("--uf", default="SP", help="UF analisada. Padrão: SP.")
    parser.add_argument(
        "--dea-engine",
        choices=["auto", "opendea", "internal"],
        default="auto",
        help="Motor DEA. auto tenta opendea e usa internal se não estiver instalado.",
    )
    parser.add_argument(
        "--appendix",
        action="store_true",
        help="Também salva figuras secundárias para apêndice.",
    )
    return parser.parse_args()


def check_files(paths: Iterable[Path]) -> None:
    missing = [path.name for path in paths if not path.exists()]
    if missing:
        msg = "Arquivos não encontrados:\n" + "\n".join(f"- {item}" for item in missing)
        raise FileNotFoundError(msg)


def minmax_1_100(series: pd.Series) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce").astype(float)
    min_value = values.min()
    max_value = values.max()

    if pd.isna(min_value) or pd.isna(max_value):
        return pd.Series(np.nan, index=series.index)

    if np.isclose(min_value, max_value):
        return pd.Series(50.0, index=series.index)

    return 1 + 99 * (values - min_value) / (max_value - min_value)


def load_data(data_dir: Path, uf: str) -> pd.DataFrame:
    idsc_path = data_dir / IDSC_FILE
    flood_path = data_dir / FLOOD_FILE
    landslide_path = data_dir / LANDSLIDE_FILE

    check_files([idsc_path, flood_path, landslide_path])

    idsc = pd.read_excel(
        idsc_path,
        sheet_name="Todos os Dados",
        usecols=lambda col: col in IDSC_COLUMNS,
    )
    idsc = idsc[idsc["UF"].eq(uf)].copy()
    idsc["cod_ibge"] = idsc["ID"].astype(int)

    flood = (
        pd.read_excel(flood_path)
        .rename(
            columns={
                "geocod_ibge": "cod_ibge",
                "valor": "risco_inundacao",
                "classe": "classe_inundacao",
            }
        )[["cod_ibge", "risco_inundacao", "classe_inundacao"]]
    )

    landslide = (
        pd.read_excel(landslide_path)
        .rename(
            columns={
                "geocod_ibge": "cod_ibge",
                "valor": "risco_deslizamento",
                "classe": "classe_deslizamento",
            }
        )[["cod_ibge", "risco_deslizamento", "classe_deslizamento"]]
    )

    return idsc.merge(flood, on="cod_ibge", how="left").merge(
        landslide, on="cod_ibge", how="left"
    )


def prepare_model_data(df: pd.DataFrame) -> pd.DataFrame:
    data = df.copy()

    data["exposicao_domicilios_risco"] = (
        100 - data["Normalizado (0-100): SDG13_PDAR"]
    )

    data["agua"] = data["Normalizado (0-100): SDG6_6_SERV_AG"]
    data["esgoto"] = data["Normalizado (0-100): SDG6_7_ESG_SAN"]
    data["tratamento_esgoto"] = data["Normalizado (0-100): SDG6_8_CLT_DML"]
    data["gestao_risco"] = data["Normalizado (0-100): SDG13_4_PREV"]
    data["infraestrutura"] = data["Normalizado (0-100): SDG9_1_INFRA"]
    data["politica_ambiental"] = data["Normalizado (0-100): SDG15_PRTAMB"]

    data["saneamento_resiliente"] = data[
        ["agua", "esgoto", "tratamento_esgoto"]
    ].mean(axis=1)
    data["governanca_investimento_adaptativo"] = data[
        ["infraestrutura", "politica_ambiental"]
    ].mean(axis=1)

    data["risco_composto"] = data[["risco_inundacao", "risco_deslizamento"]].mean(
        axis=1
    )
    data["porte_populacional_log"] = np.log1p(data["População_2022"])

    id_cols = [
        "cod_ibge",
        "Município",
        "UF",
        "População_2022",
        "Pontuação Indice ODS 2025",
        "Goal 6 Score",
        "Goal 11 Score",
        "Goal 13 Score",
        "classe_inundacao",
        "classe_deslizamento",
    ]

    keep_cols = (
        id_cols
        + INPUT_COLS
        + OUTPUT_COLS
        + DETAIL_OUTPUT_COLS
        + ["risco_composto", "porte_populacional_log"]
    )

    model_df = data[keep_cols].dropna(subset=INPUT_COLS + OUTPUT_COLS).reset_index(
        drop=True
    )
    model_df["Município"] = model_df["Município"].astype(str).str.strip()
    model_df["dmu_id"] = (
        model_df["cod_ibge"].astype(str) + " - " + model_df["Município"]
    )

    return model_df


def dea_bcc_output_internal(X: np.ndarray, Y: np.ndarray, eps: float = 0.0) -> tuple:
    n_dmus, n_inputs = X.shape
    n_outputs = Y.shape[1]

    scores = np.full(n_dmus, np.nan)
    expansion = np.full(n_dmus, np.nan)
    statuses = []

    bounds = [(eps, None)] * n_inputs + [(eps, None)] * n_outputs + [(None, None)]

    for target in range(n_dmus):
        c = np.r_[X[target, :], np.zeros(n_outputs), 1.0]

        a_ub = np.zeros((n_dmus, n_inputs + n_outputs + 1))
        a_ub[:, :n_inputs] = -X
        a_ub[:, n_inputs : n_inputs + n_outputs] = Y
        a_ub[:, -1] = -1.0

        a_eq = np.zeros((1, n_inputs + n_outputs + 1))
        a_eq[0, n_inputs : n_inputs + n_outputs] = Y[target, :]

        result = linprog(
            c,
            A_ub=a_ub,
            b_ub=np.zeros(n_dmus),
            A_eq=a_eq,
            b_eq=np.array([1.0]),
            bounds=bounds,
            method="highs",
        )

        statuses.append(result.status)

        if result.success and result.fun > 0:
            phi = max(float(result.fun), 1.0)
            expansion[target] = phi
            scores[target] = 1.0 / phi

    return scores, expansion, statuses, None


def run_dea(
    model_df: pd.DataFrame,
    X_norm: pd.DataFrame,
    Y_norm: pd.DataFrame,
    engine: Literal["auto", "opendea", "internal"],
) -> tuple:
    if engine in {"auto", "opendea"}:
        try:
            from opendea import dea_bcc_output as opendea_bcc_output
        except ImportError:
            if engine == "opendea":
                raise
        else:
            input_cols = [f"in_{col}" for col in X_norm.columns]
            output_cols = [f"out_{col}" for col in Y_norm.columns]

            dea_table = pd.concat(
                [
                    X_norm.reset_index(drop=True).set_axis(input_cols, axis=1),
                    Y_norm.reset_index(drop=True).set_axis(output_cols, axis=1),
                ],
                axis=1,
            )
            dea_table.index = model_df["dmu_id"]

            result = opendea_bcc_output(
                dea_table,
                inputs=input_cols,
                outputs=output_cols,
            )

            if "phi" not in result.columns:
                raise ValueError("Resultado do opendea não contém a coluna 'phi'.")

            phi = pd.to_numeric(result["phi"], errors="coerce").to_numpy(dtype=float)
            phi = np.where(np.isfinite(phi), np.maximum(phi, 1.0), np.nan)
            scores = np.clip(1.0 / phi, 0.0, 1.0)
            statuses = ["opendea"] * len(scores)

            return scores, phi, statuses, result

    return dea_bcc_output_internal(
        X_norm.to_numpy(dtype=float),
        Y_norm.to_numpy(dtype=float),
    )


def extract_lambda_benchmarks(dea_result: pd.DataFrame, model_df: pd.DataFrame) -> pd.Series | None:
    if dea_result is None:
        return None

    lambda_cols = [col for col in dea_result.columns if str(col).startswith("lambda_")]
    if not lambda_cols:
        return None

    label_to_city = dict(zip(model_df["dmu_id"], model_df["Município"]))
    benchmarks = []

    for _, row in dea_result[lambda_cols].iterrows():
        peers = []
        for col, value in row.items():
            if pd.notna(value) and float(value) > 1e-5:
                label = str(col).replace("lambda_", "", 1)
                peers.append(label_to_city.get(label, label))
        benchmarks.append("; ".join(peers[:5]) if peers else "")

    return pd.Series(benchmarks, index=model_df.index)


def add_distance_benchmarks(
    model_df: pd.DataFrame,
    X_norm: pd.DataFrame,
    Y_norm: pd.DataFrame,
) -> pd.Series:
    base = pd.concat(
        [
            model_df[["Município", "eficiente"]].reset_index(drop=True),
            X_norm.add_prefix("X_").reset_index(drop=True),
            Y_norm.add_prefix("Y_").reset_index(drop=True),
        ],
        axis=1,
    )

    bm_cols = [col for col in base.columns if col.startswith(("X_", "Y_"))]
    efficient_mask = base["eficiente"].to_numpy()

    efficient_names = base.loc[efficient_mask, "Município"].to_numpy()
    efficient_matrix = base.loc[efficient_mask, bm_cols].astype(float).to_numpy()
    full_matrix = base[bm_cols].astype(float).to_numpy()

    benchmarks = []

    for i in range(len(base)):
        if base.loc[i, "eficiente"]:
            benchmarks.append(base.loc[i, "Município"])
            continue

        distances = np.sqrt(((efficient_matrix - full_matrix[i]) ** 2).sum(axis=1))
        top = np.argsort(distances)[:3]
        benchmarks.append("; ".join(efficient_names[top]))

    return pd.Series(benchmarks, index=model_df.index)


def add_benchmarks(
    model_df: pd.DataFrame,
    X_norm: pd.DataFrame,
    Y_norm: pd.DataFrame,
    dea_result: pd.DataFrame | None,
) -> pd.DataFrame:
    data = model_df.copy()

    lambda_benchmarks = extract_lambda_benchmarks(dea_result, data)

    if lambda_benchmarks is not None and lambda_benchmarks.str.len().sum() > 0:
        data["benchmarks_sugeridos"] = lambda_benchmarks.fillna("")
        data["tipo_benchmark"] = "lambda_dea"
        return data

    data["benchmarks_sugeridos"] = add_distance_benchmarks(data, X_norm, Y_norm)
    data["tipo_benchmark"] = "similaridade"
    return data


def add_priority_groups(model_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    data = model_df.copy()

    risk_median = data["risco_composto"].median()
    eff_median = data["eficiencia_dea"].median()

    risk_high = data["risco_composto"] >= risk_median
    eff_high = data["eficiencia_dea"] >= eff_median

    data["grupo_prioridade"] = np.select(
        [
            risk_high & eff_high,
            risk_high & ~eff_high,
            ~risk_high & eff_high,
            ~risk_high & ~eff_high,
        ],
        [
            "alto risco/alta eficiência",
            "alto risco/baixa eficiência",
            "baixo risco/alta eficiência",
            "baixo risco/baixa eficiência",
        ],
        default="não classificado",
    )

    data["risco_percentil"] = data["risco_composto"].rank(pct=True)
    data["ineficiencia_percentil"] = (1 - data["eficiencia_dea"]).rank(pct=True)
    data["indice_prioridade"] = (
        data["risco_percentil"] * data["ineficiencia_percentil"]
    )

    priority_summary = (
        data.groupby("grupo_prioridade")
        .agg(
            municipios=("cod_ibge", "count"),
            eficiencia_media=("eficiencia_dea", "mean"),
            risco_composto_medio=("risco_composto", "mean"),
            risco_inundacao_medio=("risco_inundacao", "mean"),
            risco_deslizamento_medio=("risco_deslizamento", "mean"),
            exposicao_media=("exposicao_domicilios_risco", "mean"),
            saneamento_resiliente_medio=("saneamento_resiliente", "mean"),
            gestao_risco_media=("gestao_risco", "mean"),
            governanca_investimento_medio=(
                "governanca_investimento_adaptativo",
                "mean",
            ),
            prioridade_media=("indice_prioridade", "mean"),
        )
        .reset_index()
    )

    return data, priority_summary


def add_kmeans_groups(model_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    data = model_df.copy()

    kmeans_cols = [
        "eficiencia_dea",
        "risco_composto",
        "exposicao_domicilios_risco",
        "saneamento_resiliente",
        "gestao_risco",
        "governanca_investimento_adaptativo",
    ]
    cluster_base = data[kmeans_cols].fillna(data[kmeans_cols].median(numeric_only=True))

    scaled = StandardScaler().fit_transform(cluster_base)

    kmeans = KMeans(n_clusters=4, random_state=42, n_init=20)
    data["cluster_kmeans"] = kmeans.fit_predict(scaled) + 1

    kmeans_summary = (
        data.groupby("cluster_kmeans")
        .agg(
            municipios=("cod_ibge", "count"),
            eficiencia_media=("eficiencia_dea", "mean"),
            risco_composto_medio=("risco_composto", "mean"),
            exposicao_media=("exposicao_domicilios_risco", "mean"),
            saneamento_resiliente_medio=("saneamento_resiliente", "mean"),
            gestao_risco_media=("gestao_risco", "mean"),
            governanca_investimento_medio=(
                "governanca_investimento_adaptativo",
                "mean",
            ),
        )
        .reset_index()
    )

    return data, kmeans_summary


def fit_interpretation_models(model_df: pd.DataFrame) -> tuple:
    X_ai = model_df[IA_FEATURE_COLS].copy()
    X_ai = X_ai.fillna(X_ai.median(numeric_only=True))

    y = model_df["eficiencia_dea"].fillna(model_df["eficiencia_dea"].median())

    rf = RandomForestRegressor(
        n_estimators=700,
        random_state=42,
        min_samples_leaf=5,
        n_jobs=1,
        oob_score=True,
    )
    rf.fit(X_ai, y)

    tree = DecisionTreeRegressor(
        max_depth=3,
        min_samples_leaf=30,
        random_state=42,
    )
    tree.fit(X_ai, y)

    importance = (
        pd.DataFrame(
            {
                "variavel": IA_FEATURE_COLS,
                "nome_grafico": [FIG_LABELS.get(col, col) for col in IA_FEATURE_COLS],
                "importancia_rf": rf.feature_importances_,
            }
        )
        .sort_values("importancia_rf", ascending=False)
        .reset_index(drop=True)
    )

    return X_ai, y, rf, tree, importance


def build_summary(
    args: argparse.Namespace,
    model_df: pd.DataFrame,
    ranking: pd.DataFrame,
    priority_df: pd.DataFrame,
    rf: RandomForestRegressor,
    X_ai: pd.DataFrame,
    y: pd.Series,
    rf_importance: pd.DataFrame,
    statuses: list,
    dea_engine_used: str,
) -> dict:
    predicted = rf.predict(X_ai)

    return {
        "uf": args.uf.upper(),
        "dea_engine": dea_engine_used,
        "municipios_analisados": int(len(model_df)),
        "municipios_eficientes": int(model_df["eficiente"].sum()),
        "eficiencia_media": float(model_df["eficiencia_dea"].mean()),
        "eficiencia_mediana": float(model_df["eficiencia_dea"].median()),
        "rf_oob_score": float(getattr(rf, "oob_score_", np.nan)),
        "rf_r2_in_sample": float(r2_score(y, predicted)),
        "rf_mae_in_sample": float(mean_absolute_error(y, predicted)),
        "dea_status_count": pd.Series(statuses).value_counts().to_dict(),
        "top10_eficiencia": ranking[
            ["Município", "eficiencia_dea", "risco_composto"]
        ]
        .head(10)
        .to_dict(orient="records"),
        "bottom10_eficiencia": model_df.sort_values("eficiencia_dea")[
            ["Município", "eficiencia_dea", "risco_composto", "grupo_prioridade"]
        ]
        .head(10)
        .to_dict(orient="records"),
        "top10_prioridade": priority_df[
            [
                "Município",
                "eficiencia_dea",
                "risco_composto",
                "indice_prioridade",
                "benchmarks_sugeridos",
            ]
        ]
        .head(10)
        .to_dict(orient="records"),
        "top_importance": rf_importance.head(8).to_dict(orient="records"),
    }


def save_tables(
    out_dir: Path,
    model_df: pd.DataFrame,
    rf_importance: pd.DataFrame,
    priority_summary: pd.DataFrame,
    kmeans_summary: pd.DataFrame,
    tree_rules: str,
    summary: dict,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    result_cols = [
        "ranking_dea",
        "cod_ibge",
        "Município",
        "UF",
        "População_2022",
        "eficiencia_dea",
        "phi_expansao_outputs",
        "eficiente",
        "benchmarks_sugeridos",
        "tipo_benchmark",
        "risco_inundacao",
        "risco_deslizamento",
        "exposicao_domicilios_risco",
        "saneamento_resiliente",
        "gestao_risco",
        "governanca_investimento_adaptativo",
        "agua",
        "esgoto",
        "tratamento_esgoto",
        "infraestrutura",
        "politica_ambiental",
        "risco_composto",
        "grupo_prioridade",
        "cluster_kmeans",
        "classe_inundacao",
        "classe_deslizamento",
        "risco_percentil",
        "ineficiencia_percentil",
        "indice_prioridade",
        "eficiencia_pred_rf",
    ]

    ranking = model_df[result_cols].sort_values(
        ["eficiencia_dea", "risco_composto"],
        ascending=[False, False],
    )

    priority_df = model_df[
        (model_df["risco_composto"] >= model_df["risco_composto"].quantile(0.75))
        & (model_df["eficiencia_dea"] <= model_df["eficiencia_dea"].quantile(0.25))
    ].sort_values(
        ["indice_prioridade", "risco_composto", "eficiencia_dea"],
        ascending=[False, False, True],
    )

    ranking.to_csv(
        out_dir / "ranking_dea_municipios_sp.csv",
        index=False,
        encoding="utf-8-sig",
    )
    priority_df[result_cols].to_csv(
        out_dir / "municipios_prioritarios_alto_risco_baixa_eficiencia.csv",
        index=False,
        encoding="utf-8-sig",
    )
    rf_importance.to_csv(
        out_dir / "importancia_variaveis_rf.csv",
        index=False,
        encoding="utf-8-sig",
    )
    priority_summary.to_csv(
        out_dir / "resumo_grupos_prioridade.csv",
        index=False,
        encoding="utf-8-sig",
    )
    kmeans_summary.to_csv(
        out_dir / "resumo_clusters_kmeans.csv",
        index=False,
        encoding="utf-8-sig",
    )

    (out_dir / "regras_arvore_decisao.txt").write_text(tree_rules, encoding="utf-8")

    with open(out_dir / "resumo_modelagem.json", "w", encoding="utf-8") as file:
        json.dump(summary, file, ensure_ascii=False, indent=2)

    return ranking, priority_df


def save_priority_matrix(out_dir: Path, model_df: pd.DataFrame, priority_df: pd.DataFrame) -> None:
    plt.figure(figsize=(10, 6))

    groups = [
        "baixo risco/baixa eficiência",
        "baixo risco/alta eficiência",
        "alto risco/alta eficiência",
        "alto risco/baixa eficiência",
    ]

    for group in groups:
        subset = model_df[model_df["grupo_prioridade"].eq(group)]
        if subset.empty:
            continue
        plt.scatter(
            subset["risco_composto"],
            subset["eficiencia_dea"],
            alpha=0.65,
            label=group,
            s=36,
        )

    plt.axvline(model_df["risco_composto"].median(), linestyle="--", linewidth=1)
    plt.axhline(model_df["eficiencia_dea"].median(), linestyle="--", linewidth=1)

    for _, row in priority_df.head(8).iterrows():
        plt.annotate(
            row["Município"],
            (row["risco_composto"], row["eficiencia_dea"]),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=8,
        )

    plt.title("Matriz de prioridade: risco climático x eficiência DEA")
    plt.xlabel("Risco climático composto")
    plt.ylabel("Eficiência DEA")
    plt.legend(fontsize=8, loc="lower left")
    plt.tight_layout()
    plt.savefig(out_dir / "fig_matriz_prioridade_risco_eficiencia.png", dpi=220)
    plt.close()


def save_priority_bar(out_dir: Path, priority_df: pd.DataFrame) -> None:
    if priority_df.empty:
        return

    plot_df = priority_df.head(20).sort_values("indice_prioridade")

    plt.figure(figsize=(10, 7))
    plt.barh(plot_df["Município"], plot_df["indice_prioridade"])
    plt.title("Municípios prioritários: alto risco e baixa eficiência")
    plt.xlabel("Índice de prioridade")
    plt.ylabel("")
    plt.tight_layout()
    plt.savefig(out_dir / "fig_municipios_prioritarios_top20.png", dpi=220)
    plt.close()


def save_group_profile(out_dir: Path, model_df: pd.DataFrame) -> None:
    profile_cols = {
        "risco_composto": "Risco",
        "exposicao_domicilios_risco": "Exposição",
        "saneamento_resiliente": "Saneamento",
        "gestao_risco": "Gestão risco",
        "governanca_investimento_adaptativo": "Governança",
        "eficiencia_dea": "Eficiência",
    }

    profile = (
        model_df.groupby("grupo_prioridade")[list(profile_cols)]
        .mean()
        .rename(columns=profile_cols)
    )

    if "Risco" in profile.columns:
        profile["Risco"] = profile["Risco"] * 100
    if "Eficiência" in profile.columns:
        profile["Eficiência"] = profile["Eficiência"] * 100

    order = [
        "alto risco/baixa eficiência",
        "alto risco/alta eficiência",
        "baixo risco/baixa eficiência",
        "baixo risco/alta eficiência",
    ]
    profile = profile.reindex([idx for idx in order if idx in profile.index])

    ax = profile.T.plot(figsize=(11, 6), marker="o")
    ax.set_title("Perfil médio dos grupos de prioridade")
    ax.set_ylabel("Escala aproximada 0-100")
    ax.set_xlabel("")
    ax.legend(fontsize=8, loc="best")
    plt.xticks(rotation=25, ha="right")
    plt.tight_layout()
    plt.savefig(out_dir / "fig_perfil_grupos_prioridade.png", dpi=220)
    plt.close()


def save_rf_importance(out_dir: Path, rf_importance: pd.DataFrame) -> None:
    plot_df = rf_importance.head(10).sort_values("importancia_rf")

    plt.figure(figsize=(9, 6))
    plt.barh(plot_df["nome_grafico"], plot_df["importancia_rf"])
    plt.title("Fatores associados à eficiência DEA no Random Forest")
    plt.xlabel("Importância relativa")
    plt.ylabel("")
    plt.tight_layout()
    plt.savefig(out_dir / "fig_importancia_rf.png", dpi=220)
    plt.close()


def save_decision_tree(
    out_dir: Path,
    tree: DecisionTreeRegressor,
    feature_cols: list[str],
) -> None:
    labels = [FIG_LABELS.get(col, col) for col in feature_cols]

    plt.figure(figsize=(18, 9))
    plot_tree(
        tree,
        feature_names=labels,
        filled=True,
        rounded=True,
        fontsize=9,
        impurity=False,
    )
    plt.title("Árvore de decisão: padrões associados à eficiência DEA")
    plt.tight_layout()
    plt.savefig(out_dir / "fig_arvore_decisao_simplificada.png", dpi=220, bbox_inches="tight")
    plt.close()


def save_appendix_figures(
    out_dir: Path,
    model_df: pd.DataFrame,
    ranking: pd.DataFrame,
) -> None:
    plt.figure(figsize=(9, 5))
    model_df["eficiencia_dea"].hist(bins=20)
    plt.title("Distribuição da eficiência DEA")
    plt.xlabel("Eficiência DEA")
    plt.ylabel("Número de municípios")
    plt.tight_layout()
    plt.savefig(out_dir / "apendice_fig_hist_eficiencia.png", dpi=180)
    plt.close()

    plt.figure(figsize=(10, 5))
    ranking.head(15).iloc[::-1].plot.barh(
        x="Município",
        y="eficiencia_dea",
        legend=False,
    )
    plt.title("Top 15 municípios por eficiência DEA")
    plt.xlabel("Eficiência DEA")
    plt.ylabel("")
    plt.tight_layout()
    plt.savefig(out_dir / "apendice_fig_top15_dea.png", dpi=180)
    plt.close()

    plt.figure(figsize=(9, 5))
    for cluster_id in sorted(model_df["cluster_kmeans"].unique()):
        subset = model_df[model_df["cluster_kmeans"].eq(cluster_id)]
        plt.scatter(
            subset["risco_composto"],
            subset["eficiencia_dea"],
            alpha=0.65,
            label=f"Cluster {cluster_id}",
        )
    plt.title("Agrupamento K-means")
    plt.xlabel("Risco composto")
    plt.ylabel("Eficiência DEA")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "apendice_fig_clusters_kmeans.png", dpi=180)
    plt.close()


def save_figures(
    out_dir: Path,
    model_df: pd.DataFrame,
    ranking: pd.DataFrame,
    priority_df: pd.DataFrame,
    rf_importance: pd.DataFrame,
    tree: DecisionTreeRegressor,
    feature_cols: list[str],
    appendix: bool,
) -> None:
    save_priority_matrix(out_dir, model_df, priority_df)
    save_priority_bar(out_dir, priority_df)
    save_group_profile(out_dir, model_df)
    save_rf_importance(out_dir, rf_importance)
    save_decision_tree(out_dir, tree, feature_cols)

    if appendix:
        save_appendix_figures(out_dir, model_df, ranking)


def main() -> None:
    args = parse_args()

    data_dir = args.data_dir.resolve()
    out_dir = (args.out_dir or data_dir / "resultados").resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    raw_df = load_data(data_dir, args.uf.upper())
    model_df = prepare_model_data(raw_df)

    X_norm = model_df[INPUT_COLS].apply(minmax_1_100)
    Y_norm = model_df[OUTPUT_COLS].apply(minmax_1_100)

    scores, expansion, statuses, dea_result = run_dea(
        model_df=model_df,
        X_norm=X_norm,
        Y_norm=Y_norm,
        engine=args.dea_engine,
    )

    dea_engine_used = "opendea" if dea_result is not None else "internal"

    model_df["eficiencia_dea"] = scores
    model_df["phi_expansao_outputs"] = expansion
    model_df["eficiente"] = model_df["eficiencia_dea"] >= 0.999
    model_df["ranking_dea"] = (
        model_df["eficiencia_dea"].rank(ascending=False, method="min").astype(int)
    )

    model_df = add_benchmarks(model_df, X_norm, Y_norm, dea_result)
    model_df, priority_summary = add_priority_groups(model_df)
    model_df, kmeans_summary = add_kmeans_groups(model_df)

    X_ai, y, rf, tree, rf_importance = fit_interpretation_models(model_df)
    model_df["eficiencia_pred_rf"] = rf.predict(X_ai)

    ranking_preview = model_df.sort_values(
        ["eficiencia_dea", "risco_composto"],
        ascending=[False, False],
    )

    priority_preview = model_df[
        (model_df["risco_composto"] >= model_df["risco_composto"].quantile(0.75))
        & (model_df["eficiencia_dea"] <= model_df["eficiencia_dea"].quantile(0.25))
    ].sort_values(
        ["indice_prioridade", "risco_composto", "eficiencia_dea"],
        ascending=[False, False, True],
    )

    tree_rules = export_text(
        tree,
        feature_names=[FIG_LABELS.get(col, col) for col in IA_FEATURE_COLS],
        decimals=3,
    )

    summary = build_summary(
        args=args,
        model_df=model_df,
        ranking=ranking_preview,
        priority_df=priority_preview,
        rf=rf,
        X_ai=X_ai,
        y=y,
        rf_importance=rf_importance,
        statuses=statuses,
        dea_engine_used=dea_engine_used,
    )

    ranking, priority_df = save_tables(
        out_dir=out_dir,
        model_df=model_df,
        rf_importance=rf_importance,
        priority_summary=priority_summary,
        kmeans_summary=kmeans_summary,
        tree_rules=tree_rules,
        summary=summary,
    )

    save_figures(
        out_dir=out_dir,
        model_df=model_df,
        ranking=ranking,
        priority_df=priority_df,
        rf_importance=rf_importance,
        tree=tree,
        feature_cols=IA_FEATURE_COLS,
        appendix=args.appendix,
    )

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"\nResultados salvos em: {out_dir}")


if __name__ == "__main__":
    main()
