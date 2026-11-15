"""Integrante 1 - Aplicação Flask: rotas, upload, páginas e integração.

Usa:
  - dados.py  (Integrante 2): validar_csv, diagnostico, limpar, achados, gerar_graficos
  - modelo.py (Integrante 3): carregar_modelo, prever, obter_metadados
"""
import os
import uuid
from datetime import datetime
from functools import lru_cache
from pathlib import Path

from flask import Flask, flash, redirect, render_template, request, session, url_for
from werkzeug.exceptions import RequestEntityTooLarge

import dados
import modelo

PASTA = Path(__file__).resolve().parent
PASTA_ORIGINAIS = PASTA / "data" / "originais"   # CSV enviado, nunca alterado
CAMINHO_LIMPO = PASTA / "data" / "carros_limpo.csv"
PASTA_ORIGINAIS.mkdir(parents=True, exist_ok=True)

VERSAO_ENTREGA = "v1.0"
BASE_TREINO = "Vehicle Dataset / CarDekho (data/vehicles.csv)"
ANO_MIN, ANO_MAX = 1980, datetime.now().year
KM_MAX = 1_000_000

app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", "troque-esta-chave-em-producao")
app.config["MAX_CONTENT_LENGTH"] = 20 * 1024 * 1024  # 20 MB


# ---------- utilidades ----------
@app.template_filter("rs")
def formatar_rs(valor):
    """12345678 -> 'Rs 12.345.678'"""
    return "Rs " + f"{valor:,.0f}".replace(",", ".")


@app.template_filter("numero")
def formatar_numero(valor):
    return f"{valor:,.0f}".replace(",", ".")


@lru_cache(maxsize=1)
def modelo_carregado():
    return modelo.carregar_modelo()


def opcoes_do_modelo():
    """Lê do próprio pipeline as categorias que ele conhece (combustível, câmbio)."""
    try:
        enc = (modelo_carregado().named_steps["preparacao"]
               .named_transformers_["categoricas"].named_steps["onehot"])
        fuel, trans = enc.categories_
        return sorted(map(str, fuel)), sorted(map(str, trans))
    except Exception:
        return ["CNG", "Diesel", "LPG", "Petrol"], ["Automatic", "Manual"]


def info_modelo():
    """Texto de 'qual base e versão treinou o modelo' para exibir na tela."""
    meta = modelo.obter_metadados()
    if not meta:
        return None
    treinado_em = None
    if modelo.CAMINHO_MODELO.exists():
        treinado_em = datetime.fromtimestamp(modelo.CAMINHO_MODELO.stat().st_mtime)
    return {
        "base": BASE_TREINO,
        "versao": VERSAO_ENTREGA,
        "algoritmo": meta.get("modelo_escolhido", "?"),
        "registros": meta.get("quantidade_registros_validos"),
        "treino": meta.get("registros_treino"),
        "teste": meta.get("registros_teste"),
        "mae": meta.get("mae"),
        "r2": meta.get("r2"),
        "data": treinado_em.strftime("%d/%m/%Y %H:%M") if treinado_em else None,
    }


def arquivo_da_sessao():
    nome = session.get("arquivo")
    if not nome:
        return None
    caminho = PASTA_ORIGINAIS / nome
    return caminho if caminho.exists() else None


@app.context_processor
def injetar_global():
    return {"tem_arquivo": arquivo_da_sessao() is not None,
            "nome_original": session.get("nome_original")}


# ---------- rotas ----------
@app.route("/")
def index():
    return render_template("index.html", modelo_info=info_modelo())


@app.route("/upload", methods=["POST"])
def upload():
    arquivo = request.files.get("arquivo")
    if arquivo is None or not arquivo.filename:
        flash("Selecione um arquivo .csv antes de enviar.", "erro")
        return redirect(url_for("index"))

    if not arquivo.filename.lower().endswith(".csv"):
        flash("Formato inválido: envie um arquivo .csv.", "erro")
        return redirect(url_for("index"))

    # Salva o original com nome único (nunca é sobrescrito nem alterado)
    nome_salvo = f"{datetime.now():%Y%m%d_%H%M%S}_{uuid.uuid4().hex[:6]}.csv"
    caminho = PASTA_ORIGINAIS / nome_salvo
    arquivo.save(caminho)

    ok, mensagem, _ = dados.validar_csv(str(caminho))
    if not ok:
        caminho.unlink(missing_ok=True)
        flash(mensagem, "erro")
        return redirect(url_for("index"))

    session["arquivo"] = nome_salvo
    session["nome_original"] = os.path.basename(arquivo.filename)
    flash(mensagem, "ok")
    return redirect(url_for("analise"))


@app.route("/analise")
def analise():
    caminho = arquivo_da_sessao()
    if caminho is None:
        flash("Envie um CSV para ver a análise.", "erro")
        return redirect(url_for("index"))

    ok, mensagem, bruto = dados.validar_csv(str(caminho))
    if not ok:
        flash(mensagem, "erro")
        return redirect(url_for("index"))

    try:
        diag = dados.diagnostico(bruto)
        limpo, log = dados.limpar(bruto)
        limpo.to_csv(CAMINHO_LIMPO, index=False)
        texto_achados = dados.achados(limpo)
        graficos = dados.gerar_graficos(limpo)
    except ValueError as erro:
        flash(str(erro), "erro")
        return redirect(url_for("index"))

    amostra = bruto.head(10).to_html(classes="tabela", index=False, border=0)
    return render_template(
        "analise.html",
        mensagem=mensagem, diag=diag, log=log, amostra=amostra,
        antes=len(bruto), depois=len(limpo), colunas=bruto.shape[1],
        achados=texto_achados, graficos=graficos,
    )


@app.route("/previsao", methods=["GET", "POST"])
def previsao():
    combustiveis, cambios = opcoes_do_modelo()
    contexto = dict(
        combustiveis=combustiveis, cambios=cambios, modelo_info=info_modelo(),
        ano_min=ANO_MIN, ano_max=ANO_MAX, km_max=KM_MAX,
        form={}, erros=[], resultado=None,
    )

    if not modelo.CAMINHO_MODELO.exists():
        flash("O modelo ainda não foi treinado. Execute: python treinar.py", "erro")
        return render_template("previsao.html", **contexto)

    if request.method == "POST":
        form = {k: request.form.get(k, "").strip() for k in ("ano", "km", "combustivel", "cambio")}
        contexto["form"] = form
        erros = []

        ano = km = None
        try:
            ano = int(form["ano"])
            if not ANO_MIN <= ano <= ANO_MAX:
                erros.append(f"O ano deve estar entre {ANO_MIN} e {ANO_MAX}.")
        except ValueError:
            erros.append("Informe o ano como número inteiro (ex.: 2015).")

        try:
            km = float(form["km"].replace(",", "."))
            if km < 0:
                erros.append("A quilometragem não pode ser negativa.")
            elif km > KM_MAX:
                erros.append(f"A quilometragem deve ser de no máximo {KM_MAX:,}.".replace(",", "."))
        except ValueError:
            erros.append("Informe a quilometragem como número (ex.: 60000).")

        if form["combustivel"] not in combustiveis:
            erros.append("Escolha um combustível da lista.")
        if form["cambio"] not in cambios:
            erros.append("Escolha um tipo de câmbio da lista.")

        if not erros:
            try:
                valor = modelo.prever(ano, km, form["combustivel"], form["cambio"],
                                      modelo=modelo_carregado())
                contexto["resultado"] = {"valor": valor, "ano": ano, "km": km,
                                         "combustivel": form["combustivel"],
                                         "cambio": form["cambio"]}
            except ValueError as e:
                erros.append(str(e))
        contexto["erros"] = erros

    return render_template("previsao.html", **contexto)


@app.errorhandler(RequestEntityTooLarge)
def arquivo_grande(_):
    flash("Arquivo grande demais: o limite é 20 MB.", "erro")
    return redirect(url_for("index"))


if __name__ == "__main__":
    app.run(debug=True)