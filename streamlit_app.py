__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import os
import streamlit as st
from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_chroma import Chroma

from server.constants.models import LLMTags, EmbedModelNames
from server.constants.prompt import RAG_PROMPT
from server.constants.view import ICON_BOT, ICON_USER, ICON_ERROR, ABOUT, DISCLAIMER
from utils.formatter import format_docs, format_references


# Load environment variables
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY", None)
LLM_OPTION = os.getenv("LLM_NAME", LLMTags.LLAMA_3_2)
EMBED_MODEL_HF = os.getenv("EMBED_MODEL_HF", EmbedModelNames.VIET_LONG)
TOKENIZERS_PARALLELISM = os.getenv("TOKENIZERS_PARALLELISM", False)
NUM_DOC = int(os.getenv("NUM_DOC", 3))
MAX_EMBED_TOKEN = int(os.getenv("MAX_EMBED_TOKEN", 8000))
TEMPERATURE = float(os.getenv("TEMPERATURE", 0.3))


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
                   menu_items={"About": f"*Powered by `{'-'.join(LLM_OPTION.split('-')[:2])}` via **Groq®**.*\n--",
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
def load_embed_model(embed_model_name):
    return HuggingFaceEmbeddings(model_name=embed_model_name,
                                 model_kwargs={"trust_remote_code": True},
                                 cache_folder="./model_dir/")

if "selected_model" not in st.session_state:
    embed_model = load_embed_model(EMBED_MODEL_HF)
    vectorstore = Chroma(persist_directory="./data/chroma_db", collection_name="groq_rag", embedding_function=embed_model)
    st.session_state.retriever = vectorstore.as_retriever(search_kwargs={"k": NUM_DOC})

    st.session_state.selected_model = LLM_OPTION
    rag_llm = ChatGroq(model=LLM_OPTION, temperature=TEMPERATURE)
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
