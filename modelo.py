from pathlib import Path
import json
import joblib
import pandas as pd

PASTA_PROJETO = Path(__file__).resolve().parent
PASTA_MODELS = PASTA_PROJETO / "models"
CAMINHO_MODELO = PASTA_MODELS / "modelo_veiculos.joblib"
CAMINHO_METADADOS = PASTA_MODELS / "metadados_modelo.json"

COLUNAS_X = ["year", "km_driven", "fuel", "transmission"]
COLUNA_Y = "selling_price"


def carregar_modelo():
    """Carrega o modelo treinado"""

    if not CAMINHO_MODELO.exists():
        raise FileNotFoundError(
            "Modelo não encontrado. Execute python treinar.py primeiro."
        )

    return joblib.load(CAMINHO_MODELO)


def prever(ano, km_driven, fuel, transmission, modelo=None):
    """Recebe os dados do veiculo e retorna o preco previsto"""

    if modelo is None:
        modelo = carregar_modelo()

    try:
        ano = int(ano)
        km_driven = float(km_driven)
    except (TypeError, ValueError):
        raise ValueError("Ano e quilometragem devem ser números válidos.")

    if not 1980 <= ano <= 2100:
        raise ValueError("Informe um ano válido.")

    if km_driven < 0:
        raise ValueError("A quilometragem não pode ser negativa.")

    fuel = str(fuel).strip()
    transmission = str(transmission).strip()

    if not fuel or not transmission:
        raise ValueError("Informe combustível e transmissão.")

    entrada = pd.DataFrame([{
        "year": ano,
        "km_driven": km_driven,
        "fuel": fuel,
        "transmission": transmission
    }], columns=COLUNAS_X)

    previsao = float(modelo.predict(entrada)[0])

    return max(0.0, previsao)


def obter_metadados():
    """Retorna informacoes do treinamento"""

    if not CAMINHO_METADADOS.exists():
        return {}

    with open(CAMINHO_METADADOS, "r", encoding="utf-8") as arquivo:
        return json.load(arquivo)