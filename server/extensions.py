# from flask_jwt_extended.jwt_manager import JWTManager
from healthcheck import HealthCheck

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

from server.config import BaseConfig


# jwt = JWTManager()
health = HealthCheck()


# add your own check function to the healthcheck
def print_ok():
    return True, "OK"


health.add_check(print_ok)


def _load_embed_model() -> HuggingFaceEmbeddings:
    return HuggingFaceEmbeddings(model_name=BaseConfig.EMBED_MODEL_HF,
                                 model_kwargs={"trust_remote_code": True},
                                 cache_folder=BaseConfig.CACHE_FOLDER)

def _load_vector_store(embed_model: HuggingFaceEmbeddings) -> Chroma:
    return Chroma(persist_directory=BaseConfig.VECTOR_DB_DIR,
                  collection_name=BaseConfig.VECTOR_DB_CLT,
                  embedding_function=embed_model)


embed_model = _load_embed_model()
vector_store = _load_vector_store(embed_model)
