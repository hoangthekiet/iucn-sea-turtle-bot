# IUCN Sea Turtle Chatbot
*This is a personal project as a part of IUCN's Sea Turtle Conversation volunteer program in Hon Cau island, Tuy Phong, Binh Thuan, Vietnam.*
## Summary
### Target
Build a very simple RAG-based chatbot for Sea turtle-related Q&A.
This project aims to provide two interfaces:
1. Chat UI
2. RESTful API
### Tech-stack
* Models:
    - Open-source LLM (e.g. Facebook's **Llama** or Google's **Gemma**)
    - Open-source Embedding (`dangvantuan/vietnamese-embedding-LongContext`)
* LLM hosting: **Groq**
* RAG framework: **Langchain**
* VectorDB: **Chroma** (local)
* Demo UI: **Streamlit**
* RESTful API framework: **Flask**
### Knowledge base
**"Sea Turtle Conservation: 101 Q&A"** by **IUCN**.
### Supported languages
* Vietnamese
## Project directory

## Chat UI
`python -m streamlit run streamlit_app.py`
## RESTful API
`python flask_app.py`
## License
Usage of this project is subject to the MIT License.
## Cite
https://iucn.org/sites/default/files/2023-02/101-q-a-about-marine-turtle-conservation-final.pdf
https://github.com/groq/groq-api-cookbook/blob/main/tutorials/groq-gradio/groq-gradio-tutorial.ipynb
https://github.com/groq/groq-api-cookbook/blob/main/tutorials/benchmarking-rag-langchain/benchmarking_rag.ipynb

