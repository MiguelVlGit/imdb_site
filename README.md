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

## Observações

- O modelo só entende texto em **inglês**, pois foi treinado no dataset
  IMDB Dataset of 50K Movie Reviews (resenhas originalmente em inglês).
- O texto digitado passa pela mesma limpeza usada no treinamento: remoção
  de tags HTML, conversão para minúsculas e remoção de pontuação/números,
  antes de ser vetorizado com o mesmo `TfidfVectorizer` salvo em
  `imdb_artefatos.joblib`.
- Esta aplicação é independente de qualquer outro projeto — só depende dos
  dois arquivos `.joblib` citados acima.
