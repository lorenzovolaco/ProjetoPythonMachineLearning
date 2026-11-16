# Previsão de preços de veículos usados

Aplicação web em Flask que analisa uma base de veículos usados (upload de CSV, limpeza com Pandas e gráficos) e prevê o preço de venda de um carro a partir de ano, quilometragem, combustível e câmbio, usando um modelo de Machine Learning do scikit-learn.

Trabalho Final de Tópicos Especiais em Ciência da Computação. Tema 06: Preço de veículos usados.

---

## 1. Integrantes e responsabilidades

| Integrante | Responsabilidades |
|---|---|
| Lorenzo Volaco (`lorenzovolaco`) | `modelo.py`, `treinar.py`, `models/`, `README.md` e `requirements.txt` (treino, comparação de modelos, previsão e documentação) |
| [PREENCHER: nome] (`AngAkira`) | `app.py`, `templates/` e `static/` (rotas Flask, páginas HTML e estilo) |
| [PREENCHER: nome] (`heloisa-flores`) | `dados.py` e `data/` (validação, limpeza, análise e gráficos; dataset) |

> Cada integrante é responsável por uma parte do código, mas todos precisam saber explicar o fluxo completo (limpeza, X/y, treino/teste, modelos, métricas e integração com o Flask).

## 2. Pergunta do projeto

**Quanto vale um carro usado, dados o ano, a quilometragem, o combustível e o câmbio?**

É um problema de **regressão supervisionada**: o alvo (`selling_price`) é um valor numérico contínuo.

## 3. Dados

- **Fonte:** Vehicle Dataset (CarDekho), de Nehal Birla, Nishant Verma e Nikhil Kushwaha, publicado no Kaggle: <https://www.kaggle.com/datasets/nehalbirla/vehicle-dataset-from-cardekho>. Os dados vêm de anúncios de carros usados do site CarDekho (Índia). O arquivo usado é o de 4.340 linhas com as colunas do dicionário abaixo (`CAR DETAILS FROM CAR DEKHO.csv` no Kaggle, renomeado para `car_data.csv`).

- **Arquivo de exemplo:** `data/car_data.csv` (já incluído no repositório).
- **Moeda:** rúpias indianas (Rs / INR). Os valores **não** foram convertidos para reais; a aplicação sempre exibe "Rs".

### Dicionário de dados

| Coluna | Tipo | Descrição |
|---|---|---|
| `name` | texto | Marca e modelo do carro |
| `year` | inteiro | Ano de fabricação |
| `selling_price` | número | Preço de venda em rúpias (**alvo**) |
| `km_driven` | inteiro | Quilometragem rodada |
| `fuel` | texto | Combustível (Diesel, Petrol, CNG, LPG, Electric) |
| `seller_type` | texto | Tipo de vendedor (Individual, Dealer, Trustmark Dealer) |
| `transmission` | texto | Câmbio (Manual, Automatic) |
| `owner` | texto | Quantidade de donos anteriores |

### Esquema aceito no upload

O CSV enviado deve ter **todas** as 8 colunas acima (os nomes de colunas não diferenciam maiúsculas de minúsculas). O separador é detectado automaticamente. Arquivos com extensão diferente de `.csv`, vazios, sem alguma coluna obrigatória ou com `year`, `km_driven` ou `selling_price` majoritariamente não numéricos são rejeitados com uma mensagem explicando o motivo.

## 4. Variáveis de entrada (X) e alvo (y)

**y:** `selling_price`.

**X:**

| Variável | Por que ajuda a prever o preço |
|---|---|
| `year` | Carros mais novos tendem a valer mais |
| `km_driven` | Mais uso significa mais desgaste e menor valor |
| `fuel` | O tipo de combustível influencia o perfil e o preço do veículo |
| `transmission` | Carros automáticos costumam ser mais caros |

**Ficaram de fora:**

- `name`: tem muitos valores únicos e exigiria tratamento próprio (marca/modelo).
- `seller_type` e `owner`: não foram usadas na versão atual, para manter o formulário simples e usar apenas informações que o usuário sabe no momento da previsão.

O alvo não entra em X e não há identificadores nem variáveis que revelem o preço.

## 5. Tratamento dos dados

Há dois tratamentos, em arquivos diferentes.

### 5.1 Análise do upload (`dados.py`)

Aplicado ao arquivo enviado pelo usuário. O resultado é salvo em `data/carros_limpo.csv` e cada etapa aparece no log da página de análise.

| Etapa | Justificativa |
|---|---|
| Padronizar nomes de colunas e textos (espaços, capitalização) | Evita categorias duplicadas como `Diesel` e `diesel ` |
| Converter `year`, `km_driven` e `selling_price` para número | Valores inválidos viram nulos em vez de quebrar a análise |
| Remover linhas com valores ausentes | Cada linha tem poucas colunas e há muitos registros; remover é mais seguro do que inventar valores |
| Remover duplicatas | Registros repetidos dariam peso extra a um mesmo carro |
| Remover valores impossíveis (preço <= 0, km negativo ou acima de 1 milhão, ano fora de 1980-hoje) | São erros de digitação ou de coleta |
| Remover outliers de preço acima do percentil 99 | Poucos carros de luxo distorcem médias e gráficos |
| Criar a coluna `brand` (primeira palavra de `name`) | Facilita agrupamentos |

O diagnóstico mostra tipos, ausentes, duplicatas, valores inválidos e o tamanho da base antes e depois. O arquivo enviado não é alterado em disco pelo Pandas; apenas a versão limpa é gravada.

### 5.2 Treinamento (`treinar.py`)

Aplicado a `data/car_data.csv`. Mantém apenas as colunas de X e y, converte tipos, remove linhas sem preço, preço <= 0, ano fora de 1980-2100, km negativa e duplicatas. **Não remove outliers de preço**, diferente da análise do upload. Valores ausentes em X são tratados dentro do `Pipeline` (seção 6), aprendendo apenas com o conjunto de treino.

## 6. Modelos e avaliação

- **Divisão:** `train_test_split` com 80% treino e 20% teste, `random_state=42`. A mesma divisão é usada para todos os modelos.
- **Pré-processamento (dentro de um `Pipeline`, salvo junto com o modelo):**
  - numéricas (`year`, `km_driven`): imputação pela mediana + `StandardScaler`;
  - categóricas (`fuel`, `transmission`): imputação pelo valor mais frequente + `OneHotEncoder(handle_unknown="ignore")`.
  - Tudo é ajustado (`fit`) somente com `X_train`, o que evita vazamento de dados.
- **Modelos comparados:**

| Modelo | Configuração |
|---|---|
| Regressão Linear | padrão |
| Árvore de Decisão | `max_depth=10` |
| Random Forest | `n_estimators=150`, `max_depth=15` |

- **Métricas:**
  - **MAE** (principal): erro absoluto médio, na mesma unidade do alvo (Rs). Fácil de interpretar e menos sensível a erros extremos do que o RMSE.
  - **RMSE**: penaliza mais os erros grandes, o que é importante em preços.
  - **R²**: proporção da variação do preço explicada pelo modelo.

### Resultados

Gerados por `python treinar.py` e salvos em `models/resultados_modelos.csv`.

| Modelo | MAE (Rs) | RMSE (Rs) | R² |
|---|---|---|---|
| Árvore de Decisão | 199.640 | 414.124 | 0,505 |
| Random Forest | 200.203 | 413.965 | 0,505 |
| Regressão Linear | 219.747 | 451.778 | 0,411 |

Base de treino/teste: 3.415 registros válidos (2.732 treino e 683 teste). O preço mediano da base é Rs 350.000.

**Erro no treino × erro no teste (MAE):**

| Modelo | MAE treino (Rs) | MAE teste (Rs) |
|---|---|---|
| Regressão Linear | 208.004 | 219.747 |
| Árvore de Decisão | 117.617 | 199.640 |
| Random Forest | 111.253 | 200.203 |

### Escolha final

O modelo com **menor MAE no teste** é salvo em `models/modelo_veiculos.joblib` e usado na previsão. O escolhido foi a **Árvore de Decisão** (`max_depth=10`).

- **Por quê:** teve o menor MAE no teste, que é a métrica principal do projeto. A Random Forest ficou praticamente empatada (diferença de cerca de Rs 560 no MAE, e R² e RMSE quase idênticos), então a árvore é a opção mais simples, mais rápida e mais fácil de explicar com resultado equivalente. A Regressão Linear foi claramente pior (MAE maior e R² de 0,41), o que indica que a relação entre as variáveis e o preço não é bem descrita por uma reta.
- **Tamanho do erro:** o MAE de cerca de Rs 200 mil equivale a aproximadamente 57% do preço mediano (Rs 350 mil). O modelo explica cerca de metade da variação dos preços (R² ≈ 0,5), então serve como **estimativa aproximada**, não como avaliação precisa.
- **Overfitting:** a Árvore e a Random Forest erram bem menos no treino (cerca de Rs 117 mil e Rs 111 mil) do que no teste (cerca de Rs 200 mil), sinal de que decoram parte dos dados de treino. A Regressão Linear tem erros parecidos nos dois conjuntos, mas é limitada demais para ser melhor no teste.
- **Por que o erro é alto:** só 4 variáveis entram no modelo. Marca, modelo e estado do carro, que pesam muito no preço, ficaram de fora.

## 7. Estrutura do projeto

```
projeto/
├── app.py               # rotas Flask e integração
├── dados.py             # validação, limpeza, análise e gráficos
├── modelo.py            # carregamento do modelo e previsão
├── treinar.py           # treino, comparação e avaliação
├── templates/
│   ├── index.html       # página inicial e upload
│   ├── analise.html     # resultado da análise
│   └── previsao.html    # formulário de previsão
├── static/              # recursos estáticos
├── data/
│   └── car_data.csv     # dataset de exemplo
├── models/              # modelo, métricas e metadados gerados pelo treino
├── requirements.txt
└── README.md
```

## 8. Instalação e execução

Requer Python 3.10 ou superior.

**Linux / macOS:**

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python treinar.py
python app.py
```

**Windows (PowerShell):**

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python treinar.py
python app.py
```

Abra `http://127.0.0.1:5000` no navegador.

> É preciso executar `python treinar.py` antes de abrir a página de previsão. Ele cria a pasta `models/` com o modelo, a tabela de resultados e os metadados.

## 9. Como usar

1. **Página inicial:** envie um CSV com o esquema da seção 3 (por exemplo, `data/car_data.csv`).
2. **Análise:** veja a amostra, o diagnóstico, o log de limpeza, os achados e os três gráficos (distribuição do preço, preço mediano por combustível e quilometragem × preço), cada um com interpretação.
3. **Previsão:** informe ano, quilometragem, combustível e câmbio e veja o preço estimado em Rs. Entradas inválidas geram mensagens de erro claras.

**Observação sobre o modelo:** o upload atualiza apenas a análise e os gráficos. O modelo é treinado uma vez, com `data/car_data.csv`, e permanece fixo. A página mostra a base, a data de treino e as métricas do modelo em uso.

## 10. Limitações

- Preços em rúpias indianas, refletindo o mercado indiano de uma época específica; não valem diretamente para o Brasil.
- O modelo usa só 4 variáveis. Marca, modelo, estado de conservação, cidade e número de donos não entram, então carros muito diferentes com o mesmo perfil recebem previsões parecidas.
- Combustíveis ou câmbios raros na base têm poucos exemplos e previsões menos confiáveis.
- O modelo não extrapola bem para anos ou quilometragens muito fora do intervalo visto no treino.
- A análise do upload remove outliers de preço (percentil 99); o treino não, então os números de uma não são diretamente comparáveis aos da outra.
- O modelo não é retreinado com arquivos enviados pelo usuário.

## 11. Fontes e ferramentas

- Dataset: Birla, N.; Verma, N.; Kushwaha, N. *Vehicle dataset* (CarDekho), Kaggle, 2018. <https://www.kaggle.com/datasets/nehalbirla/vehicle-dataset-from-cardekho>.
- Bibliotecas: Flask, Pandas, NumPy, scikit-learn, joblib, Matplotlib.
- Ferramentas de IA utilizadas: Claude(para revisão do código)
