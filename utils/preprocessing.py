import os
import re
import shutil
from dotenv import find_dotenv, load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document


# Load environment variables
load_dotenv(find_dotenv(), override=True)


def create_vector_data(embed_model: HuggingFaceEmbeddings,
                       embed_tokens: int,
                       chunk_overlap: int,
                       vector_db_dir: str,
                       collection: str) -> Chroma:
    # Define splitter
    token_counter = lambda x: len(re.findall(r"\w+(?:'\w+)?|[^\w\s]", x))
    text_splitter = RecursiveCharacterTextSplitter(separators=["\n\n", "\n"],
                                                   chunk_size=embed_tokens,
                                                   chunk_overlap=chunk_overlap,
                                                   length_function=token_counter)

    # Read data
    with open("./data/raw/iucn_data.txt") as f:
        docs = f.read().split("\n--\n")
        print("* Number of documents:", len(docs))

    splitted_documents = text_splitter.create_documents(docs)
    splitted_documents = [Document(doc.page_content, metadata={"len": token_counter(doc.page_content)})\
                          for doc in splitted_documents]
    print("* Number of splitted documents:", len(splitted_documents))

    # Vectorize data
    if os.path.exists(vector_db_dir):
        shutil.rmtree(vector_db_dir)
    else:
        os.makedirs(vector_db_dir)
    vector_store = Chroma.from_documents(splitted_documents,
                                         embedding=embed_model,
                                         collection_name=collection,
                                         persist_directory=vector_db_dir)
    print("Vectorization done.")

    return vector_store


EMBED_MODEL_HF = os.getenv("EMBED_MODEL_HF")
CACHE_FOLDER = os.getenv('CACHE_FOLDER', './model_dir/')
VECTOR_DB_DIR = os.getenv('VECTOR_DB_DIR', './data/chroma_db/' + EMBED_MODEL_HF.split('/')[-1])
VECTOR_DB_CLT = os.getenv('VECTOR_DB_CLT', 'groq_rag')
MAX_EMBED_TOKEN = int(os.getenv("MAX_EMBED_TOKEN", 8192))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 100))

embed_model = HuggingFaceEmbeddings(model_name=EMBED_MODEL_HF,
                                    model_kwargs={"trust_remote_code": True},
                                    cache_folder=CACHE_FOLDER)

create_vector_data(embed_model,
                   MAX_EMBED_TOKEN,
                   CHUNK_OVERLAP,
                   VECTOR_DB_DIR,
                   VECTOR_DB_CLT)
