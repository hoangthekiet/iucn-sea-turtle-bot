__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import random
import streamlit as st

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

from server.config import BaseConfig
from server.constants.prompt import RAG_TEMPLATE
from server.constants.view import ICON_BOT, ICON_USER, ICON_ERROR, ABOUT, DISCLAIMER
from server.services.ChatService import ChatService
from utils.formatter import format_docs, format_references, format_about


# Setup page layout
def icon(emoji: str):
    """Shows an emoji as a Notion-style page icon."""
    st.write(
        f'<span style="font-size: 78px; line-height: 1">{emoji}</span>',
        unsafe_allow_html=True,
    )

st.set_page_config(page_icon="💬",
                   layout="wide",
                   page_title="Hỏi Đáp Về Rùa Biển",
                   menu_items={"About": format_about(BaseConfig.LLM_OPTION),
                               "Get help": "mailto:hoangthekiet@gmail.com",
                               "Report a bug": "https://docs.google.com/forms/d/e/1FAIpQLScgzuGFF7v8Fyxwnjm_KR71Wx1YX1_F2FhuhsQCE3bzzzpjwQ/viewform?usp=sf_link"})

with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/f/fa/IUCN_logo.svg", width=64)
    st.info(ABOUT)
    st.warning(DISCLAIMER)

icon("💬")
st.subheader("Hỏi Đáp Về Rùa Biển 🇻🇳", divider="rainbow", anchor=False)

# Initialize chat history and selected model
if "messages" not in st.session_state:
    st.session_state.messages = []

@st.cache_resource
def load_embed_model() -> HuggingFaceEmbeddings:
    """
    Load embedding model from Huggingface.
    """
    return HuggingFaceEmbeddings(model_name=BaseConfig.EMBED_MODEL_HF,
                                 model_kwargs={"trust_remote_code": True},
                                 cache_folder=BaseConfig.CACHE_FOLDER)

if "selected_model" not in st.session_state:
    embed_model = load_embed_model()
    vector_store = Chroma(persist_directory=BaseConfig.VECTOR_DB_DIR,
                          collection_name=BaseConfig.VECTOR_DB_CLT,
                          embedding_function=embed_model)

    st.session_state.selected_model = BaseConfig.LLM_OPTION
    st.session_state.chat_service = ChatService(llm=st.session_state.selected_model, vector_store=vector_store,
                                                temperature=BaseConfig.TEMPERATURE, num_doc=BaseConfig.NUM_DOC)

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    avatar = ICON_BOT if message["role"] == "assistant" else ICON_USER
    st.chat_message(message["role"], avatar=avatar).write(message["content"])


# Display chat view
if prompt := st.chat_input("Mời bạn đặt câu hỏi về Rùa biển...", max_chars=100):
    
    st.chat_message("user", avatar=ICON_USER).write(prompt)

    try:
        # Fetch response from Groq API
        references, full_response = st.session_state.chat_service.execute_rag_chain(prompt, st.session_state.messages)
        st.chat_message("assistant", avatar=ICON_BOT).write(full_response)
        # Display references
        with st.expander(label = "**Nguồn dữ liệu**\n"):
            st.markdown(format_references(references))
        # Append the full response to session_state.messages
        if isinstance(full_response, str):
            st.session_state.messages.append({"role": "user", "content": prompt})
            st.session_state.messages.append({"role": "assistant", "content": full_response})
    except Exception as e:
        st.error(e, icon=ICON_ERROR)


def get_samples(k: int):
    """
    Get `k` random question samples from predefined dataset.
    """
    if "samples" not in st.session_state:
        with open("assets/examples.txt") as f:
            st.session_state.samples = f.read().splitlines()
    return random.sample(st.session_state.samples, 3)

# Display suggestions
cols = st.columns(3)
questions = get_samples(3)
for col, s in zip(cols, questions):
    with col:
        st.markdown(f"```\n{s}\n```")