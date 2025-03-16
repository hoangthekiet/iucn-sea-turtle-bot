import os
import re
import shutil
from dotenv import find_dotenv, load_dotenv
import pickle

from langchain.retrievers import ParentDocumentRetriever
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document

from local_store import LocalStore


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
                                                   length_function=token_counter,
                                                   add_start_index=True)

    # Read data
    with open("./data/raw/iucn_data.txt") as f:
        texts = f.read().split("\n--\n")
        print("* Number of documents:", len(texts))

    documents = [Document(texts[i], metadata={'page': i}) for i in range(len(texts))]
    # Chunk data
    # chunks = text_splitter.create_documents(texts)
    # page = 0
    # for i in range(len(chunks)):
    #     _metadata = chunks[i].metadata
    #     if _metadata['start_index'] == 0:
    #         page += 1
    #     _metadata['page'] = page
    #     _doc = Document(chunks[i].page_content, metadata=_metadata)
    #     chunks[i] = _doc
    # print("* Number of splitted documents:", len(chunks), embed_tokens)

    # Vectorize data
    if os.path.exists(vector_db_dir):
        shutil.rmtree(vector_db_dir)
    else:
        os.makedirs(vector_db_dir)
    
    vector_store = Chroma(collection_name=collection,
                          embedding_function=embed_model,
                          persist_directory=vector_db_dir)
    # vector_store = Chroma.from_documents(chunks,
    #                                      embedding=embed_model,
    #                                      collection_name=collection,
    #                                      persist_directory=vector_db_dir)
    
    max_doc_len = max([token_counter(doc.page_content) for doc in documents])
    print(max_doc_len, MAX_EMBED_TOKEN)

    if max_doc_len > MAX_EMBED_TOKEN:
        # Use parent document retriever
        print("Using parent document retriever.")
        parent_docstore = LocalStore("./data/docstore/parent")
        retriever = ParentDocumentRetriever(vectorstore=vector_store,
                                            docstore=parent_docstore,
                                            child_splitter=text_splitter)
    else:
        # Use normal retriever
        print("Using normal retriever.")
        retriever = vector_store.as_retriever()
    
    retriever.add_documents(documents, ids=None)
    print("Vectorization done.")

    return retriever


EMBED_MODEL_HF = os.getenv("EMBED_MODEL_HF")
CACHE_FOLDER = os.getenv('CACHE_FOLDER', './model_dir/')
VECTOR_DB_DIR = os.getenv('VECTOR_DB_DIR', './data/chroma_db/' + EMBED_MODEL_HF.split('/')[-1])
VECTOR_DB_CLT = os.getenv('VECTOR_DB_CLT', 'groq_rag')
MAX_EMBED_TOKEN = int(os.getenv("MAX_EMBED_TOKEN", 8192))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 0))

embed_model = HuggingFaceEmbeddings(model_name=EMBED_MODEL_HF,
                                    model_kwargs={"trust_remote_code": True},
                                    cache_folder=CACHE_FOLDER)

retriever = create_vector_data(embed_model,
                               MAX_EMBED_TOKEN,
                               CHUNK_OVERLAP,
                               VECTOR_DB_DIR,
                               VECTOR_DB_CLT)
