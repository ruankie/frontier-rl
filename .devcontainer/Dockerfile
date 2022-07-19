FROM python:3.7.6-slim

COPY ./requirements.txt /requirements.txt
COPY ./src/config/00_startup.py /root/.ipython/profile_default/startup/00_startup.py

RUN apt update && \
    apt install --no-install-recommends -y build-essential gcc && \
    apt install -y git && \
    apt clean && rm -rf /var/lib/apt/lists/* && \
    pip3 install --no-cache-dir -r /requirements.txt

WORKDIR /app

ENV SHELL=/bin/bash