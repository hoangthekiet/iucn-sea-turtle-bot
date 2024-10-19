FROM openjdk:slim

RUN apt-get update -y

FROM python:3.11-slim

WORKDIR /app

COPY ./requirements_flask.txt requirements.txt
RUN pip install -r requirements.txt

COPY . .

# run server
ENV SERVICE_NAME='sea-turtle'
ENV SERVER_ENV='development'
ENV HOST_NAME='0.0.0.0'
ENV PORT_NAME='5000'
ENV NUMBER_WORKER=2
ENV NUMBER_THREAD=2

EXPOSE 5000
CMD python flask_run.py
