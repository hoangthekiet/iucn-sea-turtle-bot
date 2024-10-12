from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document

with open('../data/iucn_data.txt') as f:
    docs = f.read().split("\n--\n")
print("Number of documents:", len(docs))

embed_model = HuggingFaceEmbeddings(model_name="dangvantuan/vietnamese-embedding-LongContext", model_kwargs={"trust_remote_code": True}) # jinaai/jina-embeddings-v3

documents = [Document(doc, metadata={"len": len(doc.split())}) for doc in docs]

vectorstore = Chroma.from_documents(documents, embedding=embed_model, collection_name="groq_rag", persist_directory="../data/chroma_db")
print("Vectorization done.")