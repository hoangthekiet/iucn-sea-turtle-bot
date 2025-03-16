FROM python:3.11-slim

RUN apt-get update -y

WORKDIR /app

COPY ./requirements_flask.txt requirements.txt
RUN pip install -r requirements.txt

COPY . .

# model
ENV LLM_NAME="llama-3.3-70b-versatile"
ENV EMBED_MODEL_HF="keepitreal/vietnamese-sbert"
ENV MAX_EMBED_TOKEN=8192
ENV NUM_DOC=5
ENV TEMPERATURE=0.1

# server
ENV SERVICE_NAME='sea-turtle'
ENV SERVER_ENV='development'
ENV HOST_NAME='0.0.0.0'
ENV PORT_NAME='5000'
ENV NUMBER_WORKER=2
ENV NUMBER_THREAD=2

EXPOSE 5000
CMD python flask_run.py

# EXPOSE 8501
# CMD streamlit run /app/streamlit_app.py