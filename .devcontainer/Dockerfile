FROM python:3.7.6-slim

COPY ./requirements.txt /requirements.txt

RUN apt update && \
    apt install --no-install-recommends -y build-essential gcc && \
    apt install -y git && \
    apt clean && rm -rf /var/lib/apt/lists/* && \
    pip3 install --no-cache-dir -r /requirements.txt

EXPOSE 8080
