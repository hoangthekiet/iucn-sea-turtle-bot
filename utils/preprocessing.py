import os
import shutil
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document


# Load embedding model
load_dotenv()
EMBED_MODEL_HF = os.getenv("EMBED_MODEL_HF")
embed_model = HuggingFaceEmbeddings(model_name=EMBED_MODEL_HF,
                                    model_kwargs={"trust_remote_code": True},
                                    cache_folder="./model_dir/")

# Load data
with open('./data/raw/iucn_data.txt') as f:
    docs = f.read().split("\n--\n")
    print("Number of documents:", len(docs))
documents = [Document(doc, metadata={"len": len(doc.split())}) for doc in docs]

# Vectorize data
DB_DIR = "./data/chroma_db"
shutil.rmtree(DB_DIR)
vectorstore = Chroma.from_documents(documents, embedding=embed_model, collection_name="groq_rag", persist_directory=DB_DIR)
print("Vectorization done.")