__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import os
import streamlit as st
from typing import Generator
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_chroma import Chroma
from dotenv import load_dotenv
from server.constants.prompt import RAG_PROMPT, format_docs


load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY", None)
TOKENIZERS_PARALLELISM = os.getenv("TOKENIZERS_PARALLELISM", False)
MAX_EMBED_TOKEN = 8000


st.set_page_config(page_icon="💬", layout="wide", page_title="Bắt đầu! 🌊🌊🌊")

def icon(emoji: str):
    """Shows an emoji as a Notion-style page icon."""
    st.write(
        f'<span style="font-size: 78px; line-height: 1">{emoji}</span>',
        unsafe_allow_html=True,
    )

icon("🐢")

st.subheader("Hỏi Đáp Về Rùa Biển — IUCN Việt Nam", divider="rainbow", anchor=False)


if not GROQ_API_KEY:
    GROQ_API_KEY = st.text_input("Groq Key")

# Initialize chat history and selected model
if "messages" not in st.session_state:
    st.session_state.messages = []

# Define model details
models = [
    "llama-3.1-8b-instant", # "name": "LLaMA3-8b-Instant", "tokens": 8192, "developer": "Meta"
    "llama-3.2-11b-text-preview", # "name": "LLaMA3-11b-Preview", "tokens": 8192, "developer": "Meta"}
    "gemma2-9b-it" # "name": "Gemma2-9b-it", "tokens": 8192, "developer": "Google"
]

@st.cache_resource
def load_embed_model(embed_model_name = "dangvantuan/vietnamese-embedding-LongContext"):
    return HuggingFaceEmbeddings(model_name=embed_model_name,
                                 model_kwargs={"trust_remote_code": True})

if "selected_model" not in st.session_state:
    model_option = models[1]
    st.session_state.selected_model = model_option
    rag_llm = ChatGroq(model=model_option, temperature=0.3)

    embed_model = load_embed_model() # jinaai/jina-embeddings-v3
    vectorstore = Chroma(persist_directory="./data/chroma_db", collection_name="groq_rag", embedding_function=embed_model)
    st.session_state.retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

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
    avatar = "🐾" if message["role"] == "assistant" else "👨‍💻"
    with st.chat_message(message["role"], avatar=avatar):
        st.markdown(message["content"])


if prompt := st.chat_input("Enter your prompt here..."):
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("user", avatar="👨‍💻"):
        st.markdown(prompt)

    # Fetch response from Groq API
    try:
        related_docs = st.session_state.retriever.invoke(prompt)

        def _snippet(doc):
            lines = [line for line in doc.page_content.split("\n") if len(line) > 0]
            q = lines[0]
            a = lines[1]
            return f"{q}\n{a} […]"

        st.markdown("**Nguồn:**\n" + "\n".join(["```\n" + _snippet(d) + "\n```" for d in related_docs]))

        full_response = st.session_state.rag_chain.invoke(prompt)
        with st.chat_message("assistant", avatar="🐾"):
            st.markdown(full_response)
    except Exception as e:
        st.error(e, icon="🚨")

    # Append the full response to session_state.messages
    if isinstance(full_response, str):
        st.session_state.messages.append(
            {"role": "assistant", "content": full_response}
        )