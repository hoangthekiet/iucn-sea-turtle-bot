import os
from datetime import timedelta
from dotenv import find_dotenv, load_dotenv

from server.constants.models import LLMTags, EmbedModelNames


# Load environment variables
load_dotenv(find_dotenv(), override=True)

class BaseConfig(object):
    """Base configuration."""
    # Authentication settings
    SECRET_KEY = os.getenv('SECRET_KEY')
    # against the blacklist
    JWT_BLACKLIST_ENABLED = False
    JWT_BLACKLIST_TOKEN_CHECKS = ['access', 'refresh']
    PERMANENT_SESSION_LIFETIME = timedelta(minutes=1)
    # Read environment variables
    UPLOAD_FOLDER = os.getenv('UPLOAD_FOLDER', '/tmp')
    SERVICE_NAME = os.getenv('SERVICE_NAME', 'sea-turtle')
    GROQ_API_KEY = os.getenv("GROQ_API_KEY", None)
    LLM_OPTION = os.getenv("LLM_NAME", LLMTags.LLAMA_3_1)
    EMBED_MODEL_HF = os.getenv("EMBED_MODEL_HF", EmbedModelNames.VI_LONG)
    CACHE_FOLDER = os.getenv('CACHE_FOLDER', './model_dir/')
    VECTOR_DB_DIR = os.getenv('VECTOR_DB_DIR', './data/chroma_db/' + EMBED_MODEL_HF.split('/')[-1])
    VECTOR_DB_CLT = os.getenv('VECTOR_DB_CLT', 'groq_rag')
    NUM_DOC = int(os.getenv("NUM_DOC", 5))
    TEMPERATURE = float(os.getenv("TEMPERATURE", 0.1))



class DevelopmentConfig(BaseConfig):
    """Development configuration."""
    DEBUG = True,
    BCRYPT_LOG_ROUNDS = 4,


class ProductionConfig(BaseConfig):
    """Production configuration."""
    DEBUG = False


class TestingConfig(BaseConfig):
    """Testing configuration."""
    DEBUG = True
    TESTING = True
    WTF_CSRF_ENABLED = False
    BCRYPT_LOG_ROUNDS = 4


app_config = {
    'development': DevelopmentConfig,
    'production': ProductionConfig,
    'testing': TestingConfig,
}
