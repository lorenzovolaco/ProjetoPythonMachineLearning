# Projeto de previsão de preços de veículos usados

## Requisitos
- Python 3.10 ou superior
- Dataset em `data/car_data.csv` (já incluído)

## Instalação e execução (Linux / Arch Linux)

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python treinar.py
python app.py
```

Abra `http://127.0.0.1:5000` no navegador.

No Windows, ative o ambiente com `.venv\\Scripts\\activate`.

## Funcionalidades
- Treinamento e comparação entre Regressão Linear, Árvore de Decisão e Random Forest.
- Salvamento do melhor modelo em `models/`.
- Upload de CSV para diagnóstico, limpeza e gráficos.
- Previsão de preço a partir do ano, quilometragem, combustível e câmbio.

Os preços da base são apresentados em rúpias indianas (Rs), conforme a origem dos dados.
