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

    # O dicionário de artefatos pode ter sido salvo com chaves diferentes
    # dependendo de como o treinamento foi feito; tentamos alguns nomes comuns.
    vetorizador = None
    for chave in ("vetorizador", "tfidf", "vectorizer", "tfidf_vectorizer"):
        if isinstance(artefatos, dict) and chave in artefatos:
            vetorizador = artefatos[chave]
            break
    if vetorizador is None and hasattr(artefatos, "transform"):
        # Caso o próprio artefato salvo já seja o vetorizador
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
# Interface
# ---------------------------------------------------------------------------
def main():
    st.set_page_config(page_title="Análise de Sentimento — IMDB", page_icon="🎬", layout="centered")

    st.title("🎬 Análise de Sentimento de Resenhas de Filmes")
    st.write(
        "Digite o texto de uma resenha de filme **em inglês** e o modelo "
        "treinado no dataset IMDB (50K Movie Reviews) vai prever se o "
        "sentimento é **positivo** ou **negativo**."
    )

    modelo, artefatos = carregar_artefatos()

    if modelo is None:
        st.error(
            "Não encontrei os arquivos do modelo. Verifique se "
            f"`{CAMINHO_MODELO}` e `{CAMINHO_ARTEFATOS}` existem na pasta "
            "`modelos/`, na mesma pasta deste app.py."
        )
        st.stop()

    if artefatos["vetorizador"] is None:
        st.error(
            "O modelo foi carregado, mas não encontrei o vetorizador TF-IDF "
            "dentro de imdb_artefatos.joblib. Verifique o conteúdo desse "
            "arquivo."
        )
        st.stop()

    texto = st.text_area(
        "Resenha do filme (em inglês):",
        height=180,
        placeholder="Example: This movie was absolutely amazing, the acting was incredible...",
    )

    exemplos = st.expander("Ver exemplos de teste")
    with exemplos:
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Exemplo positivo"):
                st.session_state["texto_exemplo"] = (
                    "This movie was absolutely fantastic. The acting was "
                    "superb and the story kept me engaged from start to finish."
                )
        with col2:
            if st.button("Exemplo negativo"):
                st.session_state["texto_exemplo"] = (
                    "This movie was terrible. The plot made no sense and "
                    "the acting was painfully bad. A complete waste of time."
                )

    if "texto_exemplo" in st.session_state:
        texto = st.session_state["texto_exemplo"]
        st.text_area("Texto do exemplo selecionado:", value=texto, height=100, disabled=True)

    if st.button("Analisar sentimento", type="primary"):
        if not texto or not texto.strip():
            st.warning("Digite ou selecione um texto antes de analisar.")
        else:
            rotulo, confianca = prever_sentimento(texto, modelo, artefatos)

            if rotulo == "positive":
                st.success(f"### Sentimento previsto: Positivo 🙂")
            else:
                st.error(f"### Sentimento previsto: Negativo 🙁")

            if confianca is not None:
                st.metric("Confiança do modelo", f"{confianca:.1%}")
                st.progress(confianca)

    st.divider()
    st.caption(
        "Modelo treinado sobre o dataset IMDB Dataset of 50K Movie Reviews, "
        "com pré-processamento de limpeza de HTML, normalização para "
        "minúsculas, remoção de pontuação e vetorização TF-IDF."
    )


if __name__ == "__main__":
    main()
