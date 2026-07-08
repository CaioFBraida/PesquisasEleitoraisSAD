# Ranking de Confiabilidade de Institutos de Pesquisa Eleitoral

Trabalho da disciplina Sistemas de Apoio a Decisao (DCC166).

## Ideia geral

O objetivo e usar clusterizacao (K-Means) para classificar institutos de pesquisa
em tres niveis de confiabilidade -- alta, moderada e baixa -- com base no historico
de erros deles nas eleicoes presidenciais brasileiras de 2002 a 2022.

A intuicao e simples: se um instituto erra pouco, erra de forma consistente e nao
tem vies sistematico (nao pende sempre para o mesmo lado), ele e mais confiavel.

O script faz duas analises:

1. Ranking por instituto -- agrupa todos os erros que cada instituto cometeu e
   aplica K-Means em tres dimensoes: erro absoluto medio (MAE), vies absoluto e
   desvio padrao do erro.

2. Ranking por instituto x tipo de pesquisa -- mesma logica, mas separando
   pesquisas estimuladas, espontaneas e de rejeicao, para ver se um mesmo
   instituto erra mais em um tipo do que em outro.

So entram no ranking institutos com pelo menos 15 pesquisas na reta final da
campanha (ate 30 dias antes da eleicao). Pesquisas muito antigas erram mais --
o eleitorado ainda esta indeciso -- e nao faz sentido penalizar o instituto por
isso.

## Como rodar

```
pip install pandas numpy matplotlib scikit-learn
python analise_confiabilidade_institutos.py
```

Os CSVs de entrada ficam em `input/`. Os graficos e tabelas de saida vao para
`output/`.

## Estrutura

```
input/
    resultado_pesquisas.csv                             dados de pesquisas (Poder360)
    resultados_eleicoes_presidenciais_brasil.csv         resultados oficiais (TSE)
output/
    00_erro_por_proximidade_eleicao.png                  justificativa do corte de 30 dias
    01_ranking_institutos_mae.png                        ranking final (barras)
    02_boxplot_erro_por_tipo.png                         erro por tipo de pesquisa
    03_erro_ao_longo_do_tempo.png                        erro medio por eleicao
    04_metodo_cotovelo.png                               escolha do k
    05_erro_por_turno.png                                1o turno vs 2o turno
    ranking_confiabilidade_institutos.csv                tabela final do ranking
    ranking_instituto_tipo_pesquisa.csv                  tabela por instituto x tipo
    base_pesquisas_vs_resultado_real_completa.csv        base crua (todas as pesquisas)
    base_pesquisas_vs_resultado_real_reta_final.csv      base filtrada (30 dias)
analise_confiabilidade_institutos.py                     script principal
```

## Metricas usadas

- **MAE** (Mean Absolute Error): media dos erros absolutos. Quanto menor, mais
  preciso o instituto.
- **Vies medio**: media dos erros com sinal. Positivo = instituto superestima
  candidatos; negativo = subestima.
- **Desvio padrao do erro**: mede a consistencia. Um instituto que ora erra 1pp,
  ora erra 15pp tem desvio alto e e menos confiavel.
- **Silhouette Score**: metrica de validacao da clusterizacao. Quanto mais proximo
  de 1, melhor a separacao entre os grupos.
- **Taxa de acerto do vencedor**: percentual de pesquisas da reta final cujo
  candidato com maior intencao de voto era de fato o eleito.
