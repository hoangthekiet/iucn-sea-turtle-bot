from typing import List
from langchain_core.documents import Document


def format_docs(docs: List[Document]) -> str:
    """Format the retrieved documents"""
    return "\n".join(doc.page_content for doc in docs)


def _snippet(doc: Document) -> str:
    lines = [line for line in doc.page_content.split("\n") if len(line) > 0]
    q = lines[0]
    a = lines[1]
    return f"{q}\n{a} […]"

def format_references(source: List[Document]) -> str:
    return "\n".join(["```\n" + _snippet(d) + "\n```" for d in source])

def format_about(llm_tag: str) -> str:
    return f"*Powered by `{'-'.join(llm_tag.split('-')[:2])}` via **Groq®**.*\n--"
