import base64
import io
import os
import sys
from datetime import datetime

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

COLUNAS = ["name", "year", "selling_price", "km_driven",
           "fuel", "seller_type", "transmission", "owner"]
NUMERICAS = ["year", "km_driven", "selling_price"]
TEXTOS = ["name", "fuel", "seller_type", "transmission", "owner"]
ANO_MINIMO, LIMITE_KM, PERCENTIL = 1980, 1_000_000, 0.99
MIN_GRUPO = 30


def _rs(v):
    return "Rs " + f"{v:,.0f}".replace(",", ".")


def _minusculas(df):
    df = df.copy()
    df.columns = df.columns.astype(str).str.strip().str.lower()
    return df


def validar_csv(arquivo):
    nome = getattr(arquivo, "filename", None) or str(arquivo)
    if not nome.lower().endswith(".csv"):
        return False, "Formato inválido: envie um arquivo .csv.", None
    try:
        df = pd.read_csv(arquivo, sep=None, engine="python", encoding_errors="replace")
    except Exception:
        return False, "Não foi possível ler o arquivo: ele pode estar vazio ou ter formato/separador inválido.", None

    df = _minusculas(df)
    if df.empty:
        return False, "O arquivo não tem linhas de dados.", None
    faltando = [c for c in COLUNAS if c not in df.columns]
    if faltando:
        return False, f"Colunas obrigatórias ausentes: {', '.join(faltando)}.", None
    ruins = [c for c in NUMERICAS if pd.to_numeric(df[c], errors="coerce").notna().mean() < 0.5]
    if ruins:
        return False, f"As colunas {', '.join(ruins)} deveriam ter números, mas não têm.", None
    return True, f"Arquivo válido: {len(df)} linhas e {df.shape[1]} colunas.", df


def diagnostico(df):
    d = _minusculas(df)
    num = {c: pd.to_numeric(d[c], errors="coerce") for c in NUMERICAS}
    return {
        "linhas": len(d),
        "colunas": d.shape[1],
        "tipos": {c: str(t) for c, t in d.dtypes.items()},
        "ausentes": {c: int(v) for c, v in d.isna().sum().items()},
        "duplicatas": int(d.duplicated().sum()),
        "preco_invalido": int((num["selling_price"] <= 0).sum()),
        "km_invalido": int(((num["km_driven"] < 0) | (num["km_driven"] > LIMITE_KM)).sum()),
    }


def limpar(df):
    d = _minusculas(df)
    log = [f"Tamanho inicial: {len(d)} linhas."]

    for c in NUMERICAS:
        d[c] = pd.to_numeric(d[c], errors="coerce")
    for c in TEXTOS:
        d[c] = d[c].astype("string").str.strip().str.title().replace("", pd.NA)
    d["fuel"] = d["fuel"].replace({"Cng": "CNG", "Lpg": "LPG"})

    n = len(d)
    d = d.dropna(subset=COLUNAS)
    log.append(f"Removidas: {n - len(d)} linhas com valores ausentes.")

    n = len(d)
    d = d.drop_duplicates()
    log.append(f"Removidas: {n - len(d)} linhas duplicadas.")

    n = len(d)
    d = d[(d["selling_price"] > 0) & (d["km_driven"] >= 0) & (d["km_driven"] <= LIMITE_KM)
          & (d["year"] >= ANO_MINIMO) & (d["year"] <= datetime.now().year)]
    log.append(f"Valores impossíveis: {n - len(d)} linhas removidas (preço <= 0, km negativo "
               f"ou acima de 1 milhão, ano fora de {ANO_MINIMO}-hoje).")

    if d.empty:
        raise ValueError("Depois da limpeza não sobrou nenhuma linha válida.")

    n = len(d)
    corte = d["selling_price"].quantile(PERCENTIL)
    d = d[d["selling_price"] <= corte]
    log.append(f"Outliers: {n - len(d)} linhas removidas (preço acima do percentil "
               f"{int(PERCENTIL * 100)}, {_rs(corte)}).")

    d["brand"] = (d["name"].str.split().str[0]
                  .replace({"Bmw": "BMW", "Mg": "MG", "Land": "Land Rover", "Ashok": "Ashok Leyland"}))
    log.append("Marca: criada a coluna 'brand' com a primeira palavra de 'name'.")

    d["year"] = d["year"].astype(int)
    d["km_driven"] = d["km_driven"].astype(int)
    for c in TEXTOS + ["brand"]:
        d[c] = d[c].astype(str)
    d = d[["name", "brand", "year", "km_driven", "fuel",
           "seller_type", "transmission", "owner", "selling_price"]].reset_index(drop=True)
    log.append(f"Tamanho final: {len(d)} linhas.")
    return d, log


def achados(df):
    res = []
    for col, rotulo in [("fuel", "Combustível"), ("transmission", "Câmbio"), ("owner", "Número de donos")]:
        g = df.groupby(col)["selling_price"].agg(["median", "count"])
        g = g[g["count"] >= MIN_GRUPO].sort_values("median")
        if len(g) >= 2:
            res.append(f"{rotulo}: '{g.index[-1]}' tem o maior preço mediano ({_rs(g['median'].iloc[-1])}) "
                       f"e '{g.index[0]}' o menor ({_rs(g['median'].iloc[0])}).")
    for col, rotulo in [("year", "Ano"), ("km_driven", "Quilometragem")]:
        r = df[col].corr(df["selling_price"])
        res.append(f"{rotulo}: correlação de {r:.2f} com o preço "
                   f"({'quanto maior, maior o preço' if r > 0 else 'quanto maior, menor o preço'}).")
    return res


def _png(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight")
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode()


def gerar_graficos(df):
    preco = df["selling_price"]
    med, media, q3 = preco.median(), preco.mean(), preco.quantile(0.75)

    f1, ax = plt.subplots(figsize=(7, 4))
    ax.hist(preco / 1000, bins=40, color="#4C78A8", edgecolor="white")
    ax.set_title("Distribuição do preço de venda")
    ax.set_xlabel("Preço (mil rúpias)"); ax.set_ylabel("Quantidade de carros")
    forma = ("cauda longa à direita: poucos carros caros puxam a média para cima"
             if media > med else "distribuição sem cauda de preços altos")
    t1 = (f"Metade dos carros custa até {_rs(med)} e 75% até {_rs(q3)}. A média ({_rs(media)}) é "
          f"{'maior' if media > med else 'menor'} que a mediana: {forma}. "
          f"O maior preço da base é {_rs(preco.max())}.")


    g = df.groupby("fuel")["selling_price"].agg(["median", "count"]).sort_values("median", ascending=False)
    f2, ax = plt.subplots(figsize=(7, 4))
    ax.bar([f"{c}\n(n={int(n)})" for c, n in g["count"].items()], g["median"] / 1000, color="#4C78A8")
    ax.set_title("Preço mediano por combustível")
    ax.set_xlabel("Combustível (n = quantidade de carros)"); ax.set_ylabel("Preço mediano (mil rúpias)")
    ref = g[g["count"] >= MIN_GRUPO] if (g["count"] >= MIN_GRUPO).sum() >= 2 else g
    t2 = (f"'{ref.index[0]}' tem a maior mediana ({_rs(ref['median'].iloc[0])}) e '{ref.index[-1]}' a menor "
          f"({_rs(ref['median'].iloc[-1])}). A base é desbalanceada: '{g['count'].idxmax()}' concentra "
          f"{100 * g['count'].max() / g['count'].sum():.0f}% dos carros, e categorias com poucos carros "
          "têm medianas pouco confiáveis.")


    r = df["km_driven"].corr(preco)
    f3, ax = plt.subplots(figsize=(7, 4))
    ax.scatter(df["km_driven"] / 1000, preco / 1000, alpha=0.3, s=12, color="#4C78A8")
    ax.set_title("Quilometragem x preço de venda")
    ax.set_xlabel("Quilometragem (mil km)"); ax.set_ylabel("Preço (mil rúpias)")
    t3 = (f"A correlação entre quilometragem e preço é de {r:.2f}, indicando que veículos mais rodados tendem a ter menor valor de mercado.")

    return [{"titulo": "Distribuição do preço", "img": _png(f1), "texto": t1},
            {"titulo": "Preço por combustível", "img": _png(f2), "texto": t2},
            {"titulo": "Quilometragem x preço", "img": _png(f3), "texto": t3}]

if __name__ == "__main__":
    ok, msg, bruto = validar_csv(sys.argv[1])
    print(msg)
    if ok:
        limpo, log = limpar(bruto)
        print(*log, sep="\n")
        os.makedirs("data", exist_ok=True)
        limpo.to_csv("data/carros_limpo.csv", index=False)
        print("CSV limpo salvo em data/carros_limpo.csv")
