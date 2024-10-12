from typing import List
from langchain_core.documents import Document
from langchain.prompts import ChatPromptTemplate


RAG_SYSTEM_PROMPT = """\
Bạn là trợ lý ảo, có nhiệm vụ trả lời các câu hỏi về rùa biển và bảo tồn rùa biển. \
Hãy sử dụng văn bản cung cấp dưới đây để trả lời câu hỏi của người dùng.
```
{context}
```
"""

RAG_HUMAN_PROMPT = "{input}"

RAG_PROMPT = ChatPromptTemplate.from_messages([
    ("system", RAG_SYSTEM_PROMPT),
    ("human", RAG_HUMAN_PROMPT)
])

def format_docs(docs: List[Document]):
    """Format the retrieved documents"""
    return "\n".join(doc.page_content for doc in docs)