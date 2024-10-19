from flask.blueprints import Blueprint
from flask import request

from server.config import BaseConfig
from server.extensions import embed_model, vector_store
from server.middlewares import Authority
from server.services import ChatService


my_service_api = Blueprint('my_service_api', __name__)


@my_service_api.route('chat', methods=['POST'])
@my_service_api.route('chat/', methods=['POST'])
@Authority.no_authen
def chat():
    # Get params
    llm_option = request.json.get("llm", BaseConfig.LLM_OPTION)
    temperature = request.json.get("temp", BaseConfig.TEMPERATURE)
    num_doc = request.json.get("k", BaseConfig.NUM_DOC)
    query = request.json.get("query")
    # Create ChatService instance
    chat_service = ChatService.ChatService(llm=llm_option,
                                           temperature=temperature,
                                           vector_store=vector_store,
                                           num_doc=num_doc)
    response = chat_service.chat(query)
    return chat_service.build_output(response)
