# Eficiência Adaptativa Municipal frente ao Risco Climático

Este repositório apresenta uma análise técnico-acadêmica sobre eficiência adaptativa municipal frente ao risco climático, com aplicação aos municípios do estado de São Paulo.

O estudo utiliza Análise Envoltória de Dados (DEA) como método principal e modelos de Inteligência Artificial como apoio interpretativo para explicar padrões de eficiência e ineficiência.

## Tema do projeto

**Eficiência adaptativa municipal frente ao risco climático: uma abordagem integrada com DEA e Inteligência Artificial**

## Objetivo

Avaliar a eficiência relativa dos municípios paulistas na conversão de condições socioambientais, institucionais e estruturais em capacidade adaptativa frente a riscos climáticos.

O estudo busca:

- calcular escores de eficiência adaptativa por DEA;
- ranquear os municípios analisados;
- identificar municípios eficientes, ineficientes e prioritários;
- sugerir benchmarks para municípios ineficientes;
- usar modelos de IA para interpretar fatores associados à eficiência;
- gerar tabelas e gráficos para análise técnica e apresentação dos resultados.

## Bases de dados utilizadas

O projeto integra diferentes bases municipais por meio do código IBGE:

- **IDSC-BR 2025**: indicadores municipais relacionados aos Objetivos de Desenvolvimento Sustentável;
- **AdaptaBrasil**: indicadores de risco climático para inundação, enxurrada, alagamento e deslizamento de terra;
- **IBGE / Atlas Brasil**: dados complementares municipais, quando necessário.

## Recorte territorial

O recorte principal do estudo é o estado de **São Paulo (SP)**.

A unidade de análise é o município, tratado como **DMU** (*Decision Making Unit*) no modelo DEA.

## Metodologia

A análise segue as seguintes etapas:

1. Carregamento das bases municipais;
2. Integração dos dados pelo código IBGE;
3. Filtragem dos municípios do estado de São Paulo;
4. Seleção e tratamento dos indicadores;
5. Normalização dos inputs e outputs;
6. Aplicação do modelo DEA-BCC orientado a outputs;
7. Cálculo do ranking de eficiência municipal;
8. Identificação de municípios eficientes, ineficientes e prioritários;
9. Aplicação de modelos de Inteligência Artificial para interpretação dos resultados;
10. Geração de tabelas, gráficos e arquivos de saída.

## Modelo DEA

O modelo principal utilizado é o:

**DEA-BCC orientado a outputs**

A escolha do modelo BCC se justifica porque os municípios possuem portes, estruturas e capacidades institucionais diferentes. Assim, o modelo com retornos variáveis de escala é mais adequado do que o modelo CCR, que assume retornos constantes.

A orientação a outputs foi adotada porque o interesse do estudo é avaliar a capacidade dos municípios de ampliar seus resultados adaptativos diante do risco climático.

## Variáveis do modelo

### Inputs

Os inputs representam pressões, riscos ou necessidades adaptativas:

- risco de inundação, enxurrada e alagamento;
- risco de deslizamento de terra;
- exposição de domicílios ao risco climático.

### Outputs

Os outputs representam capacidades adaptativas desejáveis:

- saneamento resiliente;
- gestão de risco;
- governança e investimento adaptativo.

## Inteligência Artificial

A IA foi usada como apoio interpretativo, e não como substituta da DEA.

Foram aplicados modelos como:

- Random Forest;
- árvore de decisão;
- agrupamento de municípios;
- análise de importância das variáveis.

Esses modelos ajudam a explicar quais fatores estão associados aos padrões de eficiência e ineficiência encontrados pela DEA.

## Principais saídas

O script gera arquivos como:

- ranking dos municípios por eficiência DEA;
- lista de municípios prioritários;
- importância das variáveis no Random Forest;
- regras da árvore de decisão;
- resumo da modelagem;
- gráficos para relatório e apresentação.

Exemplos de saídas esperadas:

```text
ranking_dea_municipios_sp.csv
municipios_prioritarios_alto_risco_baixa_eficiencia.csv
importancia_variaveis_rf.csv
resumo_modelagem.json
regras_arvore_decisao.txt
fig_matriz_prioridade_risco_eficiencia.png
fig_importancia_rf.png
fig_arvore_decisao_simplificada.png
fig_municipios_prioritarios_top20.png
