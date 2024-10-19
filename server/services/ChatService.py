from langchain_groq import ChatGroq
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from typing import List

from server.constants.prompt import RAG_PROMPT
from server.services.BaseService import BaseService
from utils.formatter import format_docs


class ChatService(BaseService):

    def __init__(self, llm, temperature, vector_store, num_doc):
        self.retriever = vector_store.as_retriever(search_kwargs={"k": num_doc})
        self.selected_model = llm
        self.rag_llm = ChatGroq(model=llm, temperature=temperature)
        self.rag_chain = (
            {
                "context": self.retriever | format_docs, # Use retriever to retrieve docs from vectorstore -> format the documents into a string
                "input": RunnablePassthrough() # Propogate the 'input' variable to the next step
            } 
            | RAG_PROMPT # RAG_PROMPT # format prompt with 'context' and 'input' variables
            | self.rag_llm # get response from LLM using the formatted prompt
            | StrOutputParser() # Parse through LLM response to get only the string response
        )

    def retrieve(self, text: str) -> List[str]:
        references = self.retriever.invoke(text)
        return [doc.page_content for doc in references]

    def chat(self, text: str) -> str:
        full_response = self.rag_chain.invoke(text)
        return str(full_response)
