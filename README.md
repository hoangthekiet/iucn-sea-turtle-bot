# IUCN Sea Turtle Chatbot
*This is a personal project as a part of IUCN's Sea Turtle Conversation volunteer program in Hon Cau island, Tuy Phong, Binh Thuan, Vietnam.*
## Summary
### Target
Build a very simple RAG-based chatbot for Sea turtle-related Q&A.
This project aims to provide two interfaces:
1. Chat UI
2. RESTful API
### Tech-stack
* Model: Open-source LLMs (e.g. Facebook's **Llama** or Google's **Gemma**)
* Model hosting: **Groq**
* RAG framework: **Langchain**
* VectorDB: **Chroma** (local)
* Demo UI: **Streamlit**
* RESTful API framework: **Flask**
### Knowledge base
**"Sea Turtle Conservation: 101 Q&A"** by **IUCN**.
### Chat UI
`streamlit run streamlit_app.py`
### RESTful API
`python flask_app.py`