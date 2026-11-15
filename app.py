from pathlib import Path
from datetime import datetime
from flask import Flask, render_template, request, redirect, url_for, flash
import pandas as pd

from dados import validar_csv, diagnostico, limpar, achados, gerar_graficos
from modelo import carregar_modelo, prever, obter_metadados

PASTA = Path(__file__).resolve().parent
app = Flask(__name__)
app.secret_key = "chave-local-do-projeto"
app.config["MAX_CONTENT_LENGTH"] = 20 * 1024 * 1024


@app.template_filter("numero")
def numero(valor):
    try:
        return f"{int(valor):,}".replace(",", ".")
    except (TypeError, ValueError):
        return "0"


@app.template_filter("rs")
def rs(valor):
    try:
        return "Rs " + f"{float(valor):,.0f}".replace(",", ".")
    except (TypeError, ValueError):
        return "—"


def informacoes_modelo():
    dados = obter_metadados()
    if not dados:
        return None
    return {
        "base": "data/car_data.csv",
        "versao": "1.0",
        "data": dados.get("data_treinamento", ""),
        "algoritmo": dados.get("modelo_escolhido", "Modelo treinado"),
        "registros": dados.get("registros_validos", 0),
        "treino": dados.get("registros_treino", 0),
        "teste": dados.get("registros_teste", 0),
        "mae": dados.get("mae", 0),
        "r2": dados.get("r2", 0),
    }


def opcoes_previsao():
    caminho = PASTA / "data" / "car_data.csv"
    try:
        df = pd.read_csv(caminho)
        combustiveis = sorted(df["fuel"].dropna().astype(str).str.strip().unique())
        cambios = sorted(df["transmission"].dropna().astype(str).str.strip().unique())
    except Exception:
        combustiveis, cambios = ["Diesel", "Petrol", "CNG", "LPG"], ["Manual", "Automatic"]
    return combustiveis, cambios


@app.route("/")
@app.route("/index")
def index():
    return render_template("index.html", modelo_info=informacoes_modelo())


@app.route("/upload", methods=["POST"])
def upload():
    arquivo = request.files.get("arquivo")
    if not arquivo or not arquivo.filename:
        flash("Selecione um arquivo CSV.", "erro")
        return redirect(url_for("index"))
    ok, mensagem, bruto = validar_csv(arquivo)
    if not ok:
        flash(mensagem, "erro")
        return redirect(url_for("index"))
    try:
        diag = diagnostico(bruto)
        limpo, log = limpar(bruto)
        (PASTA / "data").mkdir(exist_ok=True)
        limpo.to_csv(PASTA / "data" / "carros_limpo.csv", index=False)
        graficos = gerar_graficos(limpo)
        return render_template(
            "analise.html",
            nome_original=arquivo.filename,
            antes=len(bruto),
            colunas=bruto.shape[1],
            depois=len(limpo),
            amostra=bruto.head(10).to_html(index=False, classes="tabela", border=0),
            diag=diag,
            log=log,
            achados=achados(limpo),
            graficos=graficos,
            modelo_info=informacoes_modelo(),
        )
    except Exception as erro:
        flash(f"Não foi possível analisar o arquivo: {erro}", "erro")
        return redirect(url_for("index"))


@app.route("/analise")
def analise():
    flash("Envie um arquivo CSV pela página inicial para gerar a análise.", "info")
    return redirect(url_for("index"))


@app.route("/previsao", methods=["GET", "POST"])
def previsao():
    combustiveis, cambios = opcoes_previsao()
    form = request.form if request.method == "POST" else {}
    erros, resultado = [], None
    ano_min, ano_max, km_max = 1980, datetime.now().year, 1_000_000
    if request.method == "POST":
        try:
            ano = int(form.get("ano", ""))
            km = float(form.get("km", ""))
            combustivel = form.get("combustivel", "").strip()
            cambio = form.get("cambio", "").strip()
            if not ano_min <= ano <= ano_max:
                raise ValueError(f"O ano deve estar entre {ano_min} e {ano_max}.")
            if not 0 <= km <= km_max:
                raise ValueError(f"A quilometragem deve estar entre 0 e {numero(km_max)} km.")
            if combustivel not in combustiveis or cambio not in cambios:
                raise ValueError("Selecione um combustível e um câmbio válidos.")
            valor = prever(ano, km, combustivel, cambio)
            resultado = {"valor": valor, "ano": ano, "km": km,
                         "combustivel": combustivel, "cambio": cambio}
        except FileNotFoundError as erro:
            erros.append(str(erro))
        except (TypeError, ValueError) as erro:
            erros.append(str(erro) or "Confira os dados informados.")
        except Exception as erro:
            erros.append(f"Não foi possível realizar a previsão: {erro}")
    return render_template(
        "previsao.html", erros=erros, form=form, ano_min=ano_min,
        ano_max=ano_max, km_max=km_max, combustiveis=combustiveis,
        cambios=cambios, resultado=resultado, modelo_info=informacoes_modelo()
    )


@app.errorhandler(413)
def arquivo_grande(_):
    flash("O arquivo ultrapassa o limite de 20 MB.", "erro")
    return redirect(url_for("index"))


if __name__ == "__main__":
    app.run(debug=True)
