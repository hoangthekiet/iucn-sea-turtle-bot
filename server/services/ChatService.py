import os

from langchain_groq import ChatGroq
# from langchain_core.runnables import RunnablePassthrough
# from langchain_core.output_parsers import StrOutputParser
# from langchain_core.vectorstores import VectorStoreRetriever
from typing import List, Tuple
from langchain.chains import ConversationalRetrievalChain
from langchain.retrievers import ParentDocumentRetriever
from langchain.text_splitter import RecursiveCharacterTextSplitter

from server.constants.prompt import RAG_MEMORY_TEMPLATE
from server.services.BaseService import BaseService
from utils.formatter import format_docs
from utils.local_store import LocalStore


class ChatService(BaseService):

    def __init__(self, llm, temperature, vector_store, num_doc):
        if os.path.exists("./data/docstore/parent.dat"):
            # Use parent document retriever
            print("Using parent document retriever.")
            parent_docstore = LocalStore("./data/docstore/parent")
            text_splitter = RecursiveCharacterTextSplitter()
            self.retriever = ParentDocumentRetriever(vectorstore=vector_store,
                                                     docstore=parent_docstore,
                                                     child_splitter=text_splitter,
                                                     search_kwargs={"k": num_doc})
        else:
            # Use normal retriever
            print("Using normal retriever.")
            self.retriever = vector_store.as_retriever(search_kwargs={"k": num_doc})

        self.llm = ChatGroq(model=llm, temperature=temperature)

        self.rag_chain = ConversationalRetrievalChain.from_llm(llm=self.llm,
                                                               chain_type="stuff",
                                                               retriever=self.retriever,
                                                               combine_docs_chain_kwargs={"prompt": RAG_MEMORY_TEMPLATE},
                                                               return_source_documents=True)
        
        """
        self.rag_chain = (
            {
                "context": self.retriever | format_docs, # Use retriever to retrieve docs from vectorstore -> format the documents into a string
                "input": RunnablePassthrough() # Propogate the 'input' variable to the next step
            } 
            | RAG_PROMPT # RAG_TEMPLATE # format prompt with 'context' and 'input' variables
            | self.rag_llm # get response from LLM using the formatted prompt
            | StrOutputParser() # Parse through LLM response to get only the string response
        )
        """

    def retrieve(self, text: str) -> List[str]:
        references = self.retriever.invoke(text)
        return [doc.page_content for doc in references]

    def chat(self, text: str) -> str:
        full_response = self.rag_chain.invoke(text)
        return str(full_response)

    def execute_rag_flow(self, query: str, is_for_api: bool = False) -> Tuple:
        retrieved_docs = self.retriever.invoke(query)
        context = format_docs(retrieved_docs)
        prompt = f"CONTEXT:\n\n{context}\n\n\nQUERY: {query}\n\n\nANSWER: "
        response = self.llm.invoke(prompt)
        return context if is_for_api else retrieved_docs, response.content

    def execute_rag_chain(self, query: str, h: List = []) -> Tuple:
        history = [(h[i]["content"], h[i+1]["content"]) for i in range(0, len(h) - 1, 2)]

        response = self.rag_chain.invoke({"question": query, "chat_history": history})
        retrieved_docs = response['source_documents']

        answer = response["answer"]

        return retrieved_docs, answer