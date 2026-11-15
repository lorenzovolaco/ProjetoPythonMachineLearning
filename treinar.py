from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeRegressor

from modelo import COLUNAS_X, COLUNA_Y


PASTA = Path(__file__).resolve().parent
CAMINHO_CSV = PASTA / "data" / "car_data.csv"
PASTA_MODELS = PASTA / "models"

PASTA_MODELS.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42
TEST_SIZE = 0.20


def carregar_dados(caminho_csv=CAMINHO_CSV):
    """Carrega e limpa os dados do CSV"""

    caminho_csv = Path(caminho_csv)

    if not caminho_csv.exists():
        raise FileNotFoundError(
            f"Dataset não encontrado: {caminho_csv}"
        )

    df = pd.read_csv(caminho_csv)

    df.columns = df.columns.astype(str).str.strip().str.lower()

    obrigatorias = COLUNAS_X + [COLUNA_Y]

    faltantes = [
        coluna for coluna in obrigatorias
        if coluna not in df.columns
    ]

    if faltantes:
        raise ValueError(
            f"Colunas ausentes no CSV: {faltantes}"
        )

    df = df[obrigatorias].copy()

    for coluna in ["year", "km_driven", "selling_price"]:
        df[coluna] = pd.to_numeric(
            df[coluna], errors="coerce"
        )

    for coluna in ["fuel", "transmission"]:
        df[coluna] = df[coluna].astype("string").str.strip()
        df[coluna] = df[coluna].replace("", pd.NA)

    df = df.dropna(subset=[COLUNA_Y])

    df = df[
        (df[COLUNA_Y] > 0)
        & df["year"].between(1980, 2100)
        & (df["km_driven"] >= 0)
    ]

    df = df.drop_duplicates().reset_index(drop=True)

    if len(df) < 20:
        raise ValueError(
            "Quantidade insuficiente de dados para treinamento."
        )

    return df


def criar_preprocessador():
    """Prepara as variaveis numericas e categoricas"""

    numericas = ["year", "km_driven"]
    categoricas = ["fuel", "transmission"]

    tratamento_numerico = Pipeline(steps=[
        ("imputador", SimpleImputer(strategy="median")),
        ("escala", StandardScaler())
    ])

    tratamento_categorico = Pipeline(steps=[
        ("imputador", SimpleImputer(strategy="most_frequent")),
        ("codificacao", OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocessador = ColumnTransformer(transformers=[
        ("numericas", tratamento_numerico, numericas),
        ("categoricas", tratamento_categorico, categoricas)
    ])

    return preprocessador


def criar_modelos():
    """Cria os modelos que serao comparados"""

    return {
        "Regressao Linear": LinearRegression(),

        "Arvore de Decisao": DecisionTreeRegressor(
            max_depth=10,
            random_state=RANDOM_STATE
        ),

        "Random Forest": RandomForestRegressor(
            n_estimators=150,
            max_depth=15,
            random_state=RANDOM_STATE,
            n_jobs=-1
        )
    }


def treinar():
    """Treina, avalia e salva o melhor modelo"""

    df = carregar_dados()

    X = df[COLUNAS_X]
    y = df[COLUNA_Y]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE
    )

    resultados = []
    pipelines = {}

    for nome, estimador in criar_modelos().items():

        pipeline = Pipeline(steps=[
            ("preparacao", criar_preprocessador()),
            ("regressor", estimador)
        ])

        pipeline.fit(X_train, y_train)

        previsoes = pipeline.predict(X_test)

        mae = mean_absolute_error(y_test, previsoes)

        rmse = np.sqrt(
            mean_squared_error(y_test, previsoes)
        )

        r2 = r2_score(y_test, previsoes)

        resultados.append({
            "modelo": nome,
            "MAE": float(mae),
            "RMSE": float(rmse),
            "R2": float(r2)
        })

        pipelines[nome] = pipeline

    tabela = pd.DataFrame(resultados)

    tabela = tabela.sort_values(
        "MAE"
    ).reset_index(drop=True)

    melhor_nome = tabela.loc[0, "modelo"]

    melhor_modelo = pipelines[melhor_nome]

    joblib.dump(
        melhor_modelo,
        PASTA_MODELS / "modelo_veiculos.joblib"
    )

    tabela.to_csv(
        PASTA_MODELS / "resultados_modelos.csv",
        index=False
    )

    metadados = {
        "modelo_escolhido": melhor_nome,
        "variaveis_entrada": COLUNAS_X,
        "variavel_alvo": COLUNA_Y,
        "registros_validos": int(len(df)),
        "registros_treino": int(len(X_train)),
        "registros_teste": int(len(X_test)),
        "test_size": TEST_SIZE,
        "random_state": RANDOM_STATE,
        "mae": float(tabela.loc[0, "MAE"]),
        "rmse": float(tabela.loc[0, "RMSE"]),
        "r2": float(tabela.loc[0, "R2"]),
        "moeda": "INR - Rúpia Indiana"
    }

    with open(
        PASTA_MODELS / "metadados_modelo.json",
        "w",
        encoding="utf-8"
    ) as arquivo:

        json.dump(
            metadados,
            arquivo,
            ensure_ascii=False,
            indent=4
        )

    print("\nQuantidade de registros:", len(df))

    print("\nComparação dos modelos:")
    print(tabela.to_string(index=False))

    print("\nModelo escolhido:", melhor_nome)

    print("\nModelo salvo com sucesso!")

    return tabela, melhor_nome


if __name__ == "__main__":
    treinar()