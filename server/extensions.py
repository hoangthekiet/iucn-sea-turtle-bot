# from flask_jwt_extended.jwt_manager import JWTManager
from healthcheck import HealthCheck

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

from server.config import BaseConfig


# jwt = JWTManager()
health = HealthCheck()


# add your own check function to the healthcheck
def print_ok():
    return True, "OK"


health.add_check(print_ok)


embed_model = HuggingFaceEmbeddings(model_name=BaseConfig.EMBED_MODEL_HF,
                                    model_kwargs={"trust_remote_code": True},
                                    cache_folder=BaseConfig.CACHE_FOLDER)
print('Loaded embedding model:', BaseConfig.EMBED_MODEL_HF)

vector_store = vector_store = Chroma(embedding_function=embed_model,
                                     collection_name=BaseConfig.VECTOR_DB_CLT,
                                     persist_directory=BaseConfig.VECTOR_DB_DIR)
