__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import random
import streamlit as st

from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_chroma import Chroma

from server.config import BaseConfig
from server.constants.prompt import RAG_PROMPT
from server.constants.view import ICON_BOT, ICON_USER, ICON_ERROR, ABOUT, DISCLAIMER
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
    st.session_state.retriever = vector_store.as_retriever(search_kwargs={"k": BaseConfig.NUM_DOC})

    st.session_state.selected_model = BaseConfig.LLM_OPTION
    rag_llm = ChatGroq(model=BaseConfig.LLM_OPTION, temperature=BaseConfig.TEMPERATURE)
    st.session_state.rag_chain = (
        {
            "context": st.session_state.retriever | format_docs, # Use retriever to retrieve docs from vectorstore -> format the documents into a string
            "input": RunnablePassthrough() # Propogate the 'input' variable to the next step
        } 
        | RAG_PROMPT # format prompt with 'context' and 'input' variables
        | rag_llm # get response from LLM using the formatted prompt
        | StrOutputParser() # Parse through LLM response to get only the string response
    )


# Display chat messages from history on app rerun
for message in st.session_state.messages:
    avatar = ICON_BOT if message["role"] == "assistant" else ICON_USER
    st.chat_message(message["role"], avatar=avatar).write(message["content"])


# Display chat view
if prompt := st.chat_input("Mời bạn đặt câu hỏi về Rùa biển...", max_chars=100):
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user", avatar=ICON_USER).write(prompt)

    try:
        # Fetch response from Groq API
        full_response = st.session_state.rag_chain.invoke(prompt)
        st.chat_message("assistant", avatar=ICON_BOT).write(full_response)
        # Display references
        references = st.session_state.retriever.invoke(prompt)
        with st.expander(label = "**Nguồn dữ liệu**\n"):
            st.markdown(format_references(references))
    except Exception as e:
        st.error(e, icon=ICON_ERROR)

    # Append the full response to session_state.messages
    if isinstance(full_response, str):
        st.session_state.messages.append(
            {"role": "assistant", "content": full_response}
        )


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