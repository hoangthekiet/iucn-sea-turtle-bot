# IUCN Sea Turtle Chatbot
*This is a personal project as a part of IUCN's Sea Turtle Conversation volunteer program in Hon Cau island, Tuy Phong, Binh Thuan, Vietnam.*

*We aim to provide a friendly and informative platform for those interested in sea turtles and the marine environment.*

*Our primary data source is the IUCN's 101 Q&A Handbook, which offers a wealth of information on sea turtle biology, conservation, and threats.*
## Summary
### Target
Build a very simple RAG-based chatbot for Sea turtle-related Q&A.

This project aims to provide two interfaces:
1. Web app
2. RESTful API
### Tech-stack
* Models:
    - Open-source LLM (e.g. Facebook's **Llama** or Google's **Gemma**)
    - Open-source Embedding (`dangvantuan/vietnamese-embedding-LongContext`)
* LLM hosting: **Groq**
* RAG framework: **Langchain**
* VectorDB: **Chroma** (local)
* Web app framework: **Streamlit**
* RESTful API framework: **Flask**
### Supported languages
* Vietnamese
## Project directory
```
├── run.py 						run project
├── server 						folder server app
│   ├── app.py 					init app and 3rd app
│   ├── extensions.py 			store all connect 3rd app: db, hf model
│   ├── config.py 				config env
│   ├── constants 				store constant variable
│   ├── utils					store functions that are used many times
│   │   ├── formatter.py		store functions for text formatting
│   │   ├── preprocessing.py	preprocess data and prebuild db
│   ├── controllers 			store controller layer
│   │   ├── ChatController.py	store route APIs for chatbot
│   ├── services 				store service layer
│   │   ├── BaseService.py		base service
│   │   ├── ChatService.py		store fuctions for chatbot
│   ├── models 					store model layer
│   ├── middlewares 			store middleware layer
│   │   ├── Authority.py 		store functions to check authen
```
## Web app
`python -m streamlit run streamlit_app.py`
## RESTful API
* Build Docker images: `docker build -t sea-turtle:1.0 .`
* Run docker image: `docker run -p 80:5000 -e GROQ_API_KEY=$KEY -e EMBED_MODEL_HF=$MODEL -it sea-turtle:1.0`
* Run server instantly: `python flask_run.py`
* Request:
```curl --location '<endpoint>/sea-turtle/chat-service/chat' \
--header 'Content-Type: application/json' \
--data '{
    "llm": "gemma2-9b-it",
    "temp": 0.1,
    "k": 5,
    "query": "loài rùa nào đẻ nhiều nhất"
}'
```
## License
Usage of this project is subject to the MIT License.
## Cite
https://iucn.org/sites/default/files/2023-02/101-q-a-about-marine-turtle-conservation-final.pdf

https://iucn.org/sites/default/files/2024-04/thong-bao-chuong-trinh-tnv-mua-he-nam-2024-link-dang-ky-final.pdf

https://iucn.org/sites/default/files/2024-05/thong-bao-tnv-mua-he-2024-final.pdf

https://github.com/groq/groq-api-cookbook/blob/main/tutorials/groq-gradio/groq-gradio-tutorial.ipynb

https://github.com/groq/groq-api-cookbook/blob/main/tutorials/benchmarking-rag-langchain/benchmarking_rag.ipynb