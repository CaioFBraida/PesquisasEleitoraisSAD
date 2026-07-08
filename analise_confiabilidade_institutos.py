"""
Sistemas de Apoio a Decisao (DCC166) - TVC3
Ranking de Confiabilidade de Institutos de Pesquisa Eleitoral

Pergunta-problema:
Por meio de clusterizacao de dados historicos de pesquisas eleitorais, e possivel
identificar padroes de acuracia e vies entre institutos nas eleicoes presidenciais
brasileiras?

O que este script faz:
1. Le o csv de microdados de pesquisas (Poder360) e o csv de resultados oficiais (TSE).
2. Cruza cada pesquisa com o resultado real do candidato (mesmo ano/turno/candidato).
3. Calcula o erro de cada pesquisa (percentual estimado - percentual real).
4. Agrega os erros por INSTITUTO e faz uma clusterizacao (K-Means) para gerar um
   Ranking de Confiabilidade (Alta / Media / Baixa confiabilidade).
5. Agrega os erros por INSTITUTO + TIPO DE PESQUISA (estimulada/espontanea/rejeicao)
   e faz uma segunda clusterizacao para identificar em qual "tipo" cada instituto
   costuma errar mais ou menos.
6. Gera graficos (matplotlib) e um csv final com o resultado do ranking.

Como rodar (mesma pasta que os dois arquivos csv):
    python analise_confiabilidade_institutos.py

Arquivos esperados na mesma pasta:
    - br_poder360_pesquisas_microdados__1__csv.gz  (ou o .csv descompactado)
    - resultados_eleicoes_presidenciais_brasil.csv
"""

import os
import sys
import gzip
import shutil
import warnings

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# CONFIGURACOES GERAIS
# ---------------------------------------------------------------------------
PASTA = os.path.dirname(os.path.abspath(__file__))

# Lista de nomes possiveis para o csv de pesquisas (aceita varios, na ordem abaixo).
# Ajuste aqui se voce renomear o arquivo.
NOMES_POSSIVEIS_PESQUISAS = [
    "resultado_pesquisas.csv",
    "br_poder360_pesquisas_microdados__1__csv.csv",
    "br_poder360_pesquisas_microdados.csv",
]
ARQ_PESQUISAS_GZ = os.path.join(PASTA, "br_poder360_pesquisas_microdados__1__csv.gz")
ARQ_RESULTADOS = os.path.join(PASTA, "resultados_eleicoes_presidenciais_brasil.csv")

PASTA_SAIDA = os.path.join(PASTA, "saida")
os.makedirs(PASTA_SAIDA, exist_ok=True)

MIN_PESQUISAS_POR_INSTITUTO = 15   # minimo de pesquisas p/ entrar no ranking principal
MIN_PESQUISAS_POR_GRUPO = 8        # minimo p/ entrar no ranking instituto x tipo
N_CLUSTERS = 3                     # Alta / Media / Baixa confiabilidade

# Pesquisas feitas muito antes da eleicao tendem a errar mais (o eleitorado ainda
# esta indeciso / o cenario politico muda). Para um ranking justo de "quem acerta
# o resultado final", usamos apenas pesquisas divulgadas na reta final da campanha.
DIAS_MAX_ANTES_ELEICAO = 30

plt.rcParams["figure.dpi"] = 110
plt.rcParams["axes.grid"] = True
plt.rcParams["grid.alpha"] = 0.3


# ---------------------------------------------------------------------------
# 1. CARGA DOS DADOS
# ---------------------------------------------------------------------------
def carregar_pesquisas():
    """Carrega o csv de microdados de pesquisas, tentando varios nomes possiveis
    e, por ultimo, o .gz original (descompactando-o se precisar)."""
    for nome in NOMES_POSSIVEIS_PESQUISAS:
        caminho = os.path.join(PASTA, nome)
        if os.path.exists(caminho):
            return pd.read_csv(caminho, low_memory=False)

    if os.path.exists(ARQ_PESQUISAS_GZ):
        caminho_extraido = os.path.join(PASTA, "_pesquisas_extraido.csv")
        with gzip.open(ARQ_PESQUISAS_GZ, "rb") as f_in:
            with open(caminho_extraido, "wb") as f_out:
                shutil.copyfileobj(f_in, f_out)
        return pd.read_csv(caminho_extraido, low_memory=False)

    sys.exit(
        "ERRO: nao encontrei o csv de pesquisas. Coloque um destes arquivos na "
        f"mesma pasta deste script: {NOMES_POSSIVEIS_PESQUISAS + ['br_poder360_pesquisas_microdados__1__csv.gz']}"
    )


def carregar_resultados_reais():
    if not os.path.exists(ARQ_RESULTADOS):
        sys.exit(
            "ERRO: nao encontrei 'resultados_eleicoes_presidenciais_brasil.csv' "
            "na mesma pasta deste script."
        )
    return pd.read_csv(ARQ_RESULTADOS)


# ---------------------------------------------------------------------------
# 2. PRE-PROCESSAMENTO / JOIN COM RESULTADO REAL
# ---------------------------------------------------------------------------
def preparar_base(pesquisas: pd.DataFrame, resultados: pd.DataFrame) -> pd.DataFrame:
    # Escopo do trabalho: eleicoes presidenciais (pergunta-problema)
    df = pesquisas[pesquisas["cargo"] == "presidente"].copy()

    # mantem apenas "votos totais" e cenario estimulado/espontaneo/rejeicao coerente
    df = df[df["tipo_voto"] == "votos totais"].copy()

    # data da pesquisa (usa 'data' que e a data de divulgacao/fim de campo)
    df["data"] = pd.to_datetime(df["data"], errors="coerce")

    # junta com o resultado oficial (match exato por ano, turno, cargo e candidato)
    base = df.merge(
        resultados,
        on=["ano", "turno", "cargo", "nome_candidato"],
        how="inner",
        suffixes=("", "_real"),
    )

    # erro = o que a pesquisa dizia menos o que realmente aconteceu
    base["erro"] = base["percentual"] - base["percentual_votos_validos"]
    base["erro_abs"] = base["erro"].abs()

    # dias entre a pesquisa e a eleicao (aprox.: usa a data de cada turno)
    datas_eleicao = {
        (2002, 1): "2002-10-06", (2002, 2): "2002-10-27",
        (2006, 1): "2006-10-01", (2006, 2): "2006-10-29",
        (2010, 1): "2010-10-03", (2010, 2): "2010-10-31",
        (2014, 1): "2014-10-05", (2014, 2): "2014-10-26",
        (2018, 1): "2018-10-07", (2018, 2): "2018-10-28",
        (2022, 1): "2022-10-02", (2022, 2): "2022-10-30",
    }
    base["data_eleicao"] = base.apply(
        lambda r: pd.Timestamp(datas_eleicao.get((r["ano"], r["turno"]), pd.NaT)),
        axis=1,
    )
    base["dias_ate_eleicao"] = (base["data_eleicao"] - base["data"]).dt.days

    return base


# ---------------------------------------------------------------------------
# 3. AGREGACAO POR INSTITUTO + CLUSTERIZACAO (RANKING PRINCIPAL)
# ---------------------------------------------------------------------------
def agregar_por_instituto(base: pd.DataFrame) -> pd.DataFrame:
    agg = base.groupby("instituto").agg(
        n_pesquisas=("erro", "size"),
        mae=("erro_abs", "mean"),
        vies_medio=("erro", "mean"),
        desvio_padrao_erro=("erro", "std"),
        erro_mediano=("erro_abs", "median"),
    ).reset_index()

    agg["desvio_padrao_erro"] = agg["desvio_padrao_erro"].fillna(0)
    return agg


def clusterizar_institutos(agg: pd.DataFrame) -> pd.DataFrame:
    elegiveis = agg[agg["n_pesquisas"] >= MIN_PESQUISAS_POR_INSTITUTO].copy()

    features = elegiveis[["mae", "vies_medio", "desvio_padrao_erro"]].copy()
    features["vies_abs"] = features["vies_medio"].abs()
    X = features[["mae", "vies_abs", "desvio_padrao_erro"]].values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    k = min(N_CLUSTERS, len(elegiveis)) if len(elegiveis) >= 2 else 1
    km = KMeans(n_clusters=k, n_init=10, random_state=42)
    elegiveis["cluster"] = km.fit_predict(X_scaled)

    # ordena os clusters pelo MAE medio (menor MAE = melhor) e da nomes legiveis
    ordem_clusters = (
        elegiveis.groupby("cluster")["mae"].mean().sort_values().index.tolist()
    )
    nomes = ["Alta confiabilidade", "Confiabilidade moderada", "Baixa confiabilidade"]
    mapa_nomes = {cl: nomes[i] if i < len(nomes) else f"Grupo {i}"
                  for i, cl in enumerate(ordem_clusters)}
    elegiveis["confiabilidade"] = elegiveis["cluster"].map(mapa_nomes)

    elegiveis = elegiveis.sort_values("mae").reset_index(drop=True)
    elegiveis.insert(0, "ranking", elegiveis.index + 1)
    return elegiveis, X_scaled, km


# ---------------------------------------------------------------------------
# 4. AGREGACAO POR INSTITUTO x TIPO DE PESQUISA + CLUSTERIZACAO
# ---------------------------------------------------------------------------
def agregar_por_instituto_tipo(base: pd.DataFrame) -> pd.DataFrame:
    base_tipo = base.dropna(subset=["tipo"]).copy()
    agg = base_tipo.groupby(["instituto", "tipo"]).agg(
        n_pesquisas=("erro", "size"),
        mae=("erro_abs", "mean"),
        vies_medio=("erro", "mean"),
    ).reset_index()
    return agg


def clusterizar_instituto_tipo(agg: pd.DataFrame) -> pd.DataFrame:
    elegiveis = agg[agg["n_pesquisas"] >= MIN_PESQUISAS_POR_GRUPO].copy()
    if len(elegiveis) < 3:
        elegiveis["confiabilidade"] = "dados insuficientes"
        return elegiveis

    features = elegiveis[["mae", "vies_medio"]].copy()
    features["vies_abs"] = features["vies_medio"].abs()
    X = features[["mae", "vies_abs"]].values
    X_scaled = StandardScaler().fit_transform(X)

    k = min(N_CLUSTERS, len(elegiveis))
    km = KMeans(n_clusters=k, n_init=10, random_state=42)
    elegiveis["cluster"] = km.fit_predict(X_scaled)

    ordem_clusters = (
        elegiveis.groupby("cluster")["mae"].mean().sort_values().index.tolist()
    )
    nomes = ["Alta confiabilidade", "Confiabilidade moderada", "Baixa confiabilidade"]
    mapa_nomes = {cl: nomes[i] if i < len(nomes) else f"Grupo {i}"
                  for i, cl in enumerate(ordem_clusters)}
    elegiveis["confiabilidade"] = elegiveis["cluster"].map(mapa_nomes)
    elegiveis = elegiveis.sort_values("mae").reset_index(drop=True)
    return elegiveis


# ---------------------------------------------------------------------------
# 5. GRAFICOS
# ---------------------------------------------------------------------------
CORES_CLUSTER = {
    "Alta confiabilidade": "#2a9d8f",
    "Confiabilidade moderada": "#e9c46a",
    "Baixa confiabilidade": "#e76f51",
}


def grafico_ranking_institutos(ranking: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(9, max(4, 0.4 * len(ranking))))
    ranking_plot = ranking.sort_values("mae", ascending=True)
    cores = [CORES_CLUSTER.get(c, "#999999") for c in ranking_plot["confiabilidade"]]

    ax.barh(ranking_plot["instituto"], ranking_plot["mae"], color=cores)
    ax.invert_yaxis()
    ax.set_xlabel("Erro Absoluto Medio - MAE (pontos percentuais)")
    ax.set_title("Ranking de Confiabilidade dos Institutos de Pesquisa\n"
                 "(eleicoes presidenciais - quanto menor o MAE, mais preciso)")

    handles = [plt.Rectangle((0, 0), 1, 1, color=cor) for cor in CORES_CLUSTER.values()]
    ax.legend(handles, CORES_CLUSTER.keys(), loc="lower right")

    fig.tight_layout()
    caminho = os.path.join(PASTA_SAIDA, "01_ranking_institutos_mae.png")
    fig.savefig(caminho)
    print(f"[grafico salvo] {caminho}")
    plt.show()
    plt.close(fig)


def grafico_boxplot_tipo(base: pd.DataFrame):
    tipos = [t for t in base["tipo"].dropna().unique()]
    dados = [base.loc[base["tipo"] == t, "erro_abs"].dropna().values for t in tipos]

    fig, ax = plt.subplots(figsize=(7, 5))
    bp = ax.boxplot(dados, labels=tipos, showfliers=False, patch_artist=True)
    for patch in bp["boxes"]:
        patch.set_facecolor("#8ecae6")

    ax.set_ylabel("Erro Absoluto (pontos percentuais)")
    ax.set_title("Distribuicao do Erro por Tipo de Pesquisa")
    fig.tight_layout()
    caminho = os.path.join(PASTA_SAIDA, "02_boxplot_erro_por_tipo.png")
    fig.savefig(caminho)
    print(f"[grafico salvo] {caminho}")
    plt.show()
    plt.close(fig)


def grafico_erro_por_turno(base: pd.DataFrame):
    """Compara o MAE entre 1º e 2º turno — sera que as pesquisas erram mais
    no 1º turno (muitos candidatos) ou no 2º (cenario polarizado)?"""
    agg = base.groupby("turno")["erro_abs"].mean()

    fig, ax = plt.subplots(figsize=(5, 4.5))
    rotulos = {1: "1º Turno", 2: "2º Turno"}
    barras = [agg.get(t, 0) for t in [1, 2]]
    cores = ["#457b9d", "#e63946"]
    ax.bar([rotulos[t] for t in [1, 2]], barras, color=cores)
    for i, v in enumerate(barras):
        ax.text(i, v + 0.05, f"{v:.2f} pp", ha="center", fontweight="bold")
    ax.set_ylabel("Erro Absoluto Medio (pontos percentuais)")
    ax.set_title("Erro Medio das Pesquisas: 1º Turno vs 2º Turno")
    fig.tight_layout()
    caminho = os.path.join(PASTA_SAIDA, "05_erro_por_turno.png")
    fig.savefig(caminho)
    print(f"[grafico salvo] {caminho}")
    plt.show()
    plt.close(fig)


def grafico_erro_ao_longo_do_tempo(base: pd.DataFrame):
    agg_ano = base.groupby("ano")["erro_abs"].mean().reset_index()

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(agg_ano["ano"], agg_ano["erro_abs"], marker="o", color="#264653")
    ax.set_xlabel("Ano da eleicao")
    ax.set_ylabel("Erro Absoluto Medio (pontos percentuais)")
    ax.set_title("Evolucao do Erro Medio das Pesquisas por Eleicao")
    ax.set_xticks(agg_ano["ano"])
    fig.tight_layout()
    caminho = os.path.join(PASTA_SAIDA, "03_erro_ao_longo_do_tempo.png")
    fig.savefig(caminho)
    print(f"[grafico salvo] {caminho}")
    plt.show()
    plt.close(fig)


def grafico_erro_por_proximidade(base: pd.DataFrame):
    """Mostra por que faz sentido usar so a reta final da campanha no ranking:
    pesquisas feitas muito antes da eleicao erram muito mais."""
    b = base.dropna(subset=["dias_ate_eleicao"]).copy()
    b = b[b["dias_ate_eleicao"] >= 0]
    bins = [-1, 7, 15, 30, 60, 120, 99999]
    labels = ["0-7", "8-15", "16-30", "31-60", "61-120", "120+"]
    b["faixa"] = pd.cut(b["dias_ate_eleicao"], bins=bins, labels=labels)
    agg = b.groupby("faixa", observed=True)["erro_abs"].mean()

    fig, ax = plt.subplots(figsize=(7, 5))
    cores = ["#2a9d8f" if lbl == "0-7" or lbl == "8-15" or lbl == "16-30"
             else "#e76f51" for lbl in agg.index]
    ax.bar(agg.index.astype(str), agg.values, color=cores)
    ax.axvline(2.5, color="gray", linestyle="--", linewidth=1)
    ax.set_xlabel("Dias antes da eleicao em que a pesquisa foi divulgada")
    ax.set_ylabel("Erro Absoluto Medio (pontos percentuais)")
    ax.set_title("Erro das Pesquisas conforme a Distancia da Eleicao\n"
                 f"(ranking principal usa apenas ate {DIAS_MAX_ANTES_ELEICAO} dias - area verde)")
    fig.tight_layout()
    caminho = os.path.join(PASTA_SAIDA, "00_erro_por_proximidade_eleicao.png")
    fig.savefig(caminho)
    print(f"[grafico salvo] {caminho}")
    plt.show()
    plt.close(fig)


def grafico_cotovelo(X_scaled: np.ndarray):
    """Grafico do metodo do cotovelo, para justificar o numero de clusters (k)."""
    if len(X_scaled) < 5:
        return
    inercias = []
    ks = range(1, min(7, len(X_scaled)) + 1)
    for k in ks:
        km = KMeans(n_clusters=k, n_init=10, random_state=42).fit(X_scaled)
        inercias.append(km.inertia_)

    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.plot(list(ks), inercias, marker="o", color="#023047")
    ax.axvline(N_CLUSTERS, color="red", linestyle="--", alpha=0.6,
               label=f"k escolhido = {N_CLUSTERS}")
    ax.set_xlabel("Numero de clusters (k)")
    ax.set_ylabel("Inercia (WCSS)")
    ax.set_title("Metodo do Cotovelo - Escolha de k")
    ax.legend()
    fig.tight_layout()
    caminho = os.path.join(PASTA_SAIDA, "04_metodo_cotovelo.png")
    fig.savefig(caminho)
    print(f"[grafico salvo] {caminho}")
    plt.show()
    plt.close(fig)


# ---------------------------------------------------------------------------
# 6. IMPRESSAO NO TERMINAL
# ---------------------------------------------------------------------------
def imprimir_ranking(ranking: pd.DataFrame):
    print("\n" + "=" * 78)
    print("RANKING DE CONFIABILIDADE DOS INSTITUTOS DE PESQUISA ELEITORAL")
    print(f"(institutos com no minimo {MIN_PESQUISAS_POR_INSTITUTO} pesquisas em eleicoes presidenciais)")
    print("=" * 78)
    cols = ["ranking", "instituto", "n_pesquisas", "mae", "vies_medio",
            "desvio_padrao_erro", "confiabilidade"]
    tabela = ranking[cols].copy()
    tabela.columns = ["#", "Instituto", "Nº pesquisas", "MAE", "Viés médio",
                       "Desvio padrão", "Classificação"]
    for col in ["MAE", "Viés médio", "Desvio padrão"]:
        tabela[col] = tabela[col].round(2)
    print(tabela.to_string(index=False))
    print("=" * 78)
    print("MAE = erro absoluto medio (pontos percentuais). Quanto menor, mais preciso.")
    print("Vies medio > 0 = instituto tende a SUPERESTIMAR o candidato.")
    print("Vies medio < 0 = instituto tende a SUBESTIMAR o candidato.")


def imprimir_ranking_tipo(ranking_tipo: pd.DataFrame):
    print("\n" + "=" * 78)
    print("RANKING POR INSTITUTO x TIPO DE PESQUISA (estimulada / espontanea / rejeicao)")
    print(f"(combinacoes com no minimo {MIN_PESQUISAS_POR_GRUPO} pesquisas)")
    print("=" * 78)
    if "cluster" not in ranking_tipo.columns:
        print("Dados insuficientes para clusterizar por tipo de pesquisa.")
        return
    cols = ["instituto", "tipo", "n_pesquisas", "mae", "vies_medio", "confiabilidade"]
    tabela = ranking_tipo[cols].copy()
    tabela.columns = ["Instituto", "Tipo", "Nº pesquisas", "MAE", "Viés médio", "Classificação"]
    for col in ["MAE", "Viés médio"]:
        tabela[col] = tabela[col].round(2)
    print(tabela.to_string(index=False))
    print("=" * 78)


def imprimir_taxa_acerto_vencedor(base: pd.DataFrame):
    """Calcula quantas pesquisas da reta final 'acertaram' o candidato vencedor
    (o candidato com maior percentual na pesquisa era realmente o eleito)."""
    if "eleito" not in base.columns:
        return
    # para cada pesquisa (id unico), o candidato com maior percentual estimado
    idx_max = base.groupby(["id_pesquisa", "ano", "turno"])["percentual"].idxmax()
    tops = base.loc[idx_max]
    acertos = tops["eleito"].sum()
    total = len(tops)
    taxa = 100 * acertos / total if total > 0 else 0

    print("\n" + "=" * 78)
    print("TAXA DE ACERTO DO VENCEDOR (reta final da campanha)")
    print("=" * 78)
    print(f"  Pesquisas analisadas: {total:,}".replace(",", "."))
    print(f"  Acertaram o vencedor: {acertos:,}".replace(",", "."))
    print(f"  Taxa de acerto:       {taxa:.1f}%")
    print("=" * 78)


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------
def main():
    print("Carregando dados...")
    pesquisas = carregar_pesquisas()
    resultados = carregar_resultados_reais()
    print(f"  -> {len(pesquisas):,} linhas de pesquisas carregadas".replace(",", "."))
    print(f"  -> {len(resultados)} linhas de resultados oficiais carregadas")

    base = preparar_base(pesquisas, resultados)
    print(f"  -> {len(base):,} registros pesquisa x candidato cruzados com resultado real".replace(",", "."))

    if base.empty:
        sys.exit("Nenhum registro cruzou com o resultado real. Verifique os nomes de candidatos.")

    # Pesquisas muito distantes da eleicao erram bem mais (ver grafico 00).
    # Por isso o ranking de confiabilidade usa apenas a reta final da campanha.
    base_reta_final = base[
        (base["dias_ate_eleicao"] >= 0) & (base["dias_ate_eleicao"] <= DIAS_MAX_ANTES_ELEICAO)
    ].copy()
    print(f"  -> {len(base_reta_final):,} registros na reta final "
          f"(ate {DIAS_MAX_ANTES_ELEICAO} dias antes da eleicao)".replace(",", "."))

    # ---- Ranking principal por instituto (reta final da campanha) ---------
    agg_instituto = agregar_por_instituto(base_reta_final)
    ranking, X_scaled, _km = clusterizar_institutos(agg_instituto)
    imprimir_ranking(ranking)

    # ---- Ranking por instituto x tipo de pesquisa (reta final) -------------
    agg_tipo = agregar_por_instituto_tipo(base_reta_final)
    ranking_tipo = clusterizar_instituto_tipo(agg_tipo)
    imprimir_ranking_tipo(ranking_tipo)

    # ---- Metricas complementares --------------------------------------------
    sil = silhouette_score(X_scaled, _km.labels_) if len(set(_km.labels_)) > 1 else 0
    print(f"\nCoeficiente de Silhueta (k={N_CLUSTERS}): {sil:.3f}  (quanto mais proximo de 1, melhor a separacao dos clusters)")

    imprimir_taxa_acerto_vencedor(base_reta_final)

    # ---- Graficos -----------------------------------------------------------
    print("\nGerando graficos (salvos em ./saida)...")
    grafico_erro_por_proximidade(base)          # usa TODAS as pesquisas (justificativa)
    grafico_ranking_institutos(ranking)
    grafico_boxplot_tipo(base_reta_final)
    grafico_erro_ao_longo_do_tempo(base)        # usa TODAS as pesquisas (visao historica)
    grafico_erro_por_turno(base_reta_final)
    grafico_cotovelo(X_scaled)

    # ---- Exporta CSVs finais --------------------------------------------
    caminho_ranking = os.path.join(PASTA_SAIDA, "ranking_confiabilidade_institutos.csv")
    ranking.to_csv(caminho_ranking, index=False)
    print(f"\n[csv salvo] {caminho_ranking}")

    caminho_ranking_tipo = os.path.join(PASTA_SAIDA, "ranking_instituto_tipo_pesquisa.csv")
    ranking_tipo.to_csv(caminho_ranking_tipo, index=False)
    print(f"[csv salvo] {caminho_ranking_tipo}")

    caminho_base = os.path.join(PASTA_SAIDA, "base_pesquisas_vs_resultado_real_completa.csv")
    base.to_csv(caminho_base, index=False)
    print(f"[csv salvo] {caminho_base}")

    caminho_base_reta = os.path.join(PASTA_SAIDA, "base_pesquisas_vs_resultado_real_reta_final.csv")
    base_reta_final.to_csv(caminho_base_reta, index=False)
    print(f"[csv salvo] {caminho_base_reta}")

    print("\nConcluido.")


if __name__ == "__main__":
    main()