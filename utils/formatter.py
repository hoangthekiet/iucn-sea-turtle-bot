import re
from typing import List
from langchain_core.documents import Document


def format_docs(docs: List[Document]) -> str:
    """Format the retrieved documents"""
    docs = sorted(docs, key=lambda doc: doc.metadata['page'] * 1e6 + doc.metadata['start_index'])
    return "\n".join(doc.page_content for doc in docs)


def _snippet(doc: Document) -> str:
    lines = [line for line in re.split(r'\. |\! |\? |\n', doc.page_content) if len(line) > 0]
    q = lines[0]
    if len(lines) > 1:
        a = lines[1][:100]
        return f"{q}\n{a} […]"
    else:
        return q[:100] + " […]"

def format_references(source: List[Document]) -> str:
    return "\n".join(["```\n" + _snippet(d) + "\n```" for d in source])

def format_about(llm_tag: str) -> str:
    return f"*Powered by `{'-'.join(llm_tag.split('-')[:2])}` via **Groq®**.*\n--"
