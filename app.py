"""
app.py
======
Protótipo de aplicação web (Streamlit) para análise de sentimento de
resenhas de filmes, usando um modelo de classificação já treinado para o
dataset IMDB (50K Movie Reviews).

Esta aplicação é independente de qualquer outro projeto: ela apenas carrega
os artefatos já treinados (modelo + vetorizador TF-IDF) e os utiliza para
prever o sentimento de um texto digitado pelo usuário.

Arquivos esperados na pasta `modelos/`:
    - imdb_modelo.joblib      -> modelo de classificação treinado (sklearn)
    - imdb_artefatos.joblib   -> dicionário com o TfidfVectorizer e os nomes
                                 das classes (ex.: {"vetorizador": ...,
                                 "nomes_classes": ["negative", "positive"]})

Como executar:
    streamlit run app.py
"""

import os
import re

import joblib
import streamlit as st

MODELOS_DIR = "modelos"
CAMINHO_MODELO = os.path.join(MODELOS_DIR, "imdb_modelo.joblib")
CAMINHO_ARTEFATOS = os.path.join(MODELOS_DIR, "imdb_artefatos.joblib")

# ---------------------------------------------------------------------------
# Pré-processamento de texto (precisa ser idêntico ao usado no treinamento)
# ---------------------------------------------------------------------------
_TAG_HTML = re.compile(r"<[^>]+>")
_PONTUACAO = re.compile(r"[^a-z\s]")


def limpar_texto(texto):
    """Remove HTML, converte para minúsculas e elimina pontuação/números."""
    texto = _TAG_HTML.sub(" ", str(texto))
    texto = texto.lower()
    texto = _PONTUACAO.sub(" ", texto)
    texto = re.sub(r"\s+", " ", texto).strip()
    return texto


# ---------------------------------------------------------------------------
# Carregamento dos artefatos (cacheado para não recarregar a cada interação)
# ---------------------------------------------------------------------------
@st.cache_resource
def carregar_artefatos():
    """Carrega o modelo e o vetorizador TF-IDF treinados. Retorna None se ausentes."""
    if not os.path.exists(CAMINHO_MODELO) or not os.path.exists(CAMINHO_ARTEFATOS):
        return None, None

    modelo = joblib.load(CAMINHO_MODELO)
    artefatos = joblib.load(CAMINHO_ARTEFATOS)

    vetorizador = None
    for chave in ("vetorizador", "tfidf", "vectorizer", "tfidf_vectorizer"):
        if isinstance(artefatos, dict) and chave in artefatos:
            vetorizador = artefatos[chave]
            break
    if vetorizador is None and hasattr(artefatos, "transform"):
        vetorizador = artefatos

    nomes_classes = None
    if isinstance(artefatos, dict):
        for chave in ("nomes_classes", "classes", "labels"):
            if chave in artefatos:
                nomes_classes = artefatos[chave]
                break
    if nomes_classes is None:
        nomes_classes = ["negative", "positive"]

    return modelo, {"vetorizador": vetorizador, "nomes_classes": nomes_classes}


def prever_sentimento(texto, modelo, artefatos):
    """Aplica a limpeza, vetorização e o modelo para prever o sentimento."""
    limpo = limpar_texto(texto)
    X = artefatos["vetorizador"].transform([limpo])
    pred = int(modelo.predict(X)[0])
    rotulo = artefatos["nomes_classes"][pred]

    confianca = None
    if hasattr(modelo, "predict_proba"):
        confianca = float(modelo.predict_proba(X)[0][pred])

    return rotulo, confianca


# ---------------------------------------------------------------------------
# Estilo — paleta neutra, tipografia sóbria, sem decoração supérflua
# ---------------------------------------------------------------------------
def injetar_estilo():
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Source+Serif+4:opsz,wght@8..60,400;8..60,600&family=Inter:wght@400;500;600&display=swap');

        :root {
            --ink: #2B2A28;
            --paper: #FAF9F7;
            --line: #DEDAD3;
            --muted: #8A8580;
            --accent: #5B5650;
            --font-body: 'Inter', -apple-system, "Segoe UI", Roboto, sans-serif;
            --font-serif: 'Source Serif 4', Georgia, "Times New Roman", serif;
        }

        html, body, [class*="css"] {
            font-family: var(--font-body);
        }

        .stApp {
            background-color: var(--paper);
        }

        .block-container {
            max-width: 720px;
            padding-top: 4rem;
            padding-bottom: 4rem;
        }

        /* Cabeçalho */
        .app-eyebrow {
            font-family: 'Inter', sans-serif;
            font-size: 0.78rem;
            letter-spacing: 0.12em;
            text-transform: uppercase;
            color: var(--muted);
            margin-bottom: 0.6rem;
        }
        .app-title {
            font-family: var(--font-serif);
            font-size: 2.35rem;
            font-weight: 600;
            color: var(--ink);
            line-height: 1.15;
            margin-bottom: 0.9rem;
        }
        .app-subtitle {
            font-size: 0.98rem;
            color: var(--accent);
            line-height: 1.55;
            max-width: 540px;
            margin-bottom: 2.4rem;
        }

        hr.app-rule {
            border: none;
            border-top: 1px solid var(--line);
            margin: 2.2rem 0;
        }

        /* Inputs */
        .stTextArea textarea {
            background-color: #FFFFFF;
            border: 1px solid var(--line);
            border-radius: 4px;
            color: var(--ink);
            font-size: 0.95rem;
            line-height: 1.5;
        }
        .stTextArea textarea:focus {
            border-color: var(--accent);
            box-shadow: none;
        }
        label {
            font-size: 0.85rem !important;
            color: var(--accent) !important;
            font-weight: 500 !important;
        }

        /* Botão principal */
        .stButton button[kind="primary"] {
            background-color: var(--ink);
            color: var(--paper);
            border: none;
            border-radius: 4px;
            padding: 0.55rem 1.6rem;
            font-size: 0.9rem;
            font-weight: 500;
            letter-spacing: 0.01em;
            transition: opacity 0.15s ease;
        }
        .stButton button[kind="primary"]:hover {
            opacity: 0.82;
            color: var(--paper);
        }

        /* Botões secundários (exemplos) */
        .stButton button[kind="secondary"] {
            background-color: transparent;
            color: var(--accent);
            border: 1px solid var(--line);
            border-radius: 4px;
            font-size: 0.85rem;
            padding: 0.45rem 1rem;
        }
        .stButton button[kind="secondary"]:hover {
            border-color: var(--accent);
            color: var(--ink);
        }

        /* Resultado */
        .result-box {
            border: 1px solid var(--line);
            border-left: 3px solid var(--ink);
            border-radius: 2px;
            padding: 1.4rem 1.6rem;
            margin-top: 1.6rem;
            background-color: #FFFFFF;
        }
        .result-box.negative {
            border-left-color: #8C5B52;
        }
        .result-label {
            font-size: 0.75rem;
            letter-spacing: 0.1em;
            text-transform: uppercase;
            color: var(--muted);
            margin-bottom: 0.35rem;
        }
        .result-value {
            font-family: var(--font-serif);
            font-size: 1.6rem;
            font-weight: 600;
            color: var(--ink);
            margin-bottom: 0.7rem;
        }
        .confidence-track {
            width: 100%;
            height: 5px;
            background-color: var(--line);
            border-radius: 3px;
            overflow: hidden;
            margin-top: 0.4rem;
        }
        .confidence-fill {
            height: 100%;
            background-color: var(--ink);
            border-radius: 3px;
        }
        .confidence-text {
            font-size: 0.82rem;
            color: var(--muted);
            margin-top: 0.5rem;
        }
        .confidence-note {
            font-size: 0.8rem;
            color: #8C5B52;
            margin-top: 0.6rem;
            font-style: italic;
        }

        footer, #MainMenu, header {
            visibility: hidden;
        }
        .app-footer {
            font-size: 0.78rem;
            color: var(--muted);
            line-height: 1.6;
            margin-top: 0.5rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


EXEMPLO_POSITIVO = (
    "This movie was absolutely fantastic. The acting was superb and the "
    "story kept me engaged from start to finish."
)
EXEMPLO_NEGATIVO = (
    "This movie was terrible. The plot made no sense and the acting was "
    "painfully bad. A complete waste of time."
)


# ---------------------------------------------------------------------------
# Interface
# ---------------------------------------------------------------------------
def main():
    st.set_page_config(
        page_title="Análise de Sentimento — IMDB",
        page_icon=None,
        layout="centered",
    )
    injetar_estilo()

    st.markdown('<div class="app-eyebrow">Modelo de classificação · IMDB 50K Reviews</div>', unsafe_allow_html=True)
    st.markdown('<div class="app-title">Análise de sentimento de resenhas</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="app-subtitle">Insira o texto de uma resenha de filme em inglês. '
        "O modelo, treinado sobre 50 mil resenhas do IMDB, classifica o "
        "sentimento como positivo ou negativo.</div>",
        unsafe_allow_html=True,
    )

    modelo, artefatos = carregar_artefatos()

    if modelo is None:
        st.error(
            "Modelo não encontrado. Verifique se `imdb_modelo.joblib` e "
            "`imdb_artefatos.joblib` estão na pasta `modelos/`."
        )
        st.stop()

    if artefatos["vetorizador"] is None:
        st.error(
            "O vetorizador TF-IDF não foi encontrado dentro de "
            "`imdb_artefatos.joblib`. Verifique o conteúdo desse arquivo."
        )
        st.stop()

    # Streamlit não permite escrever em st.session_state[key] depois que o
    # widget com essa key já foi instanciado no mesmo rerun. Por isso os
    # botões de exemplo são processados ANTES da criação do text_area —
    # a escrita acontece a tempo de valer para esta mesma execução do script.
    if "texto_resenha" not in st.session_state:
        st.session_state["texto_resenha"] = ""

    col1, col2, col3 = st.columns([1, 1, 2])
    with col1:
        if st.button("Exemplo positivo", type="secondary", use_container_width=True):
            st.session_state["texto_resenha"] = EXEMPLO_POSITIVO
    with col2:
        if st.button("Exemplo negativo", type="secondary", use_container_width=True):
            st.session_state["texto_resenha"] = EXEMPLO_NEGATIVO

    texto = st.text_area(
        "Resenha",
        height=160,
        placeholder="Type or paste a movie review in English…",
        key="texto_resenha",
        label_visibility="collapsed",
    )

    st.write("")
    analisar = st.button("Analisar resenha", type="primary")

    if analisar:
        if not texto or not texto.strip():
            st.warning("Insira um texto antes de analisar.")
        else:
            rotulo, confianca = prever_sentimento(texto, modelo, artefatos)
            positivo = rotulo == "positive"
            classe_css = "" if positivo else "negative"
            texto_resultado = "Positivo" if positivo else "Negativo"

            # Importante: o HTML não pode ter indentação no início das linhas
            # dentro do f-string — st.markdown trata 4+ espaços de indentação
            # como bloco de código e exibe o HTML como texto bruto em vez de
            # renderizá-lo.
            partes = [
                f'<div class="result-box {classe_css}">',
                '<div class="result-label">Sentimento previsto</div>',
                f'<div class="result-value">{texto_resultado}</div>',
            ]
            if confianca is not None:
                pct = confianca * 100
                partes.append(
                    f'<div class="confidence-track">'
                    f'<div class="confidence-fill" style="width:{pct:.1f}%"></div>'
                    f'</div>'
                )
                partes.append(f'<div class="confidence-text">Confiança do modelo: {pct:.1f}%</div>')
                if confianca < 0.65:
                    partes.append(
                        '<div class="confidence-note">Confiança baixa — o texto '
                        "pode conter sinais mistos ou ser ambíguo para o modelo.</div>"
                    )
            partes.append("</div>")

            html = "".join(partes)
            st.markdown(html, unsafe_allow_html=True)

    st.markdown('<hr class="app-rule">', unsafe_allow_html=True)
    st.markdown(
        '<div class="app-footer">Pré-processamento: remoção de HTML, conversão '
        "para minúsculas, remoção de pontuação e vetorização TF-IDF — "
        "aplicado de forma idêntica ao utilizado no treinamento do modelo."
        "</div>",
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
