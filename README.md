# Análise de Sentimento de Resenhas (IMDB) — Site Streamlit

Protótipo de aplicação web que usa um modelo já treinado para classificar
o sentimento (positivo/negativo) de resenhas de filmes em inglês.

## Estrutura

```
imdb_site/
├── app.py              # aplicação Streamlit (interface + lógica de previsão)
├── requirements.txt
├── README.md
└── modelos/
    ├── imdb_modelo.joblib       # modelo treinado (ex.: MLPClassifier)
    └── imdb_artefatos.joblib    # vetorizador TF-IDF + nomes das classes
```

## 1. Colocar os arquivos do modelo

Copie os arquivos `imdb_modelo.joblib` e `imdb_artefatos.joblib` (já treinados)
para a pasta `modelos/`, dentro desta mesma pasta `imdb_site/`.

## 2. Instalar dependências

```bash
pip install -r requirements.txt
```

## 3. Rodar o site

```bash
streamlit run app.py
```

O Streamlit abre automaticamente uma aba no navegador (geralmente em
`http://localhost:8501`). Se não abrir, copie o link que aparecer no
terminal e cole no navegador.

## Como usar

1. Digite o texto de uma resenha de filme **em inglês** na caixa de texto
   (ou clique em um dos botões de exemplo, positivo ou negativo).
2. Clique em **"Analisar sentimento"**.
3. O resultado aparece destacado (Positivo ou Negativo), junto com o nível
   de confiança do modelo.

## Deploy no Render

Este projeto já inclui os arquivos necessários para rodar no Render
(`render.yaml` e `.streamlit/config.toml`).

### Passo a passo

1. **Coloque os arquivos do modelo na pasta `modelos/`** (substituindo o
   `.gitkeep`): `imdb_modelo.joblib` e `imdb_artefatos.joblib`.

2. **Suba este projeto para um repositório no GitHub** (o Render faz deploy
   a partir de um repositório Git):
   ```bash
   git init
   git add .
   git commit -m "Site de análise de sentimento IMDB"
   git branch -M main
   git remote add origin <URL_DO_SEU_REPOSITORIO>
   git push -u origin main
   ```
   > Os arquivos `.joblib` deste projeto somam poucas dezenas de MB, dentro
   > do limite padrão do GitHub (100 MB por arquivo). Se algum arquivo for
   > maior que isso, use [Git LFS](https://git-lfs.com/).

3. **No painel do Render** (https://dashboard.render.com):
   - Clique em **New** → **Web Service**.
   - Conecte o repositório que você acabou de criar.
   - O Render deve detectar o `render.yaml` automaticamente e preencher:
     - **Build Command:** `pip install -r requirements.txt`
     - **Start Command:** `streamlit run app.py --server.port $PORT --server.address 0.0.0.0`
   - Caso não detecte automaticamente, preencha esses dois campos manualmente.
   - Escolha o plano **Free** (suficiente para este protótipo).
   - Clique em **Create Web Service**.

4. Aguarde o build (alguns minutos). Quando terminar, o Render fornece uma
   URL pública (algo como `https://imdb-sentiment-app.onrender.com`) — é o
   link do seu site, acessível de qualquer navegador.

### Observações sobre o plano gratuito do Render

- O serviço "dorme" depois de um período de inatividade; a primeira
  requisição depois disso demora mais (o Render precisa religar o
  container).
- Não é necessário configurar nada além do que já está nos arquivos deste
  projeto — `--server.port $PORT` faz o Streamlit escutar na porta que o
  Render define automaticamente.

## Observações gerais

- O modelo só entende texto em **inglês**, pois foi treinado no dataset
  IMDB Dataset of 50K Movie Reviews (resenhas originalmente em inglês).
- O texto digitado passa pela mesma limpeza usada no treinamento: remoção
  de tags HTML, conversão para minúsculas e remoção de pontuação/números,
  antes de ser vetorizado com o mesmo `TfidfVectorizer` salvo em
  `imdb_artefatos.joblib`.
- Esta aplicação é independente de qualquer outro projeto — só depende dos
  dois arquivos `.joblib` citados acima.

