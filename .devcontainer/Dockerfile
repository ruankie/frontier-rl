# See here for image contents: https://github.com/microsoft/vscode-dev-containers/tree/v0.231.2/containers/python-3-anaconda/.devcontainer/base.Dockerfile

#FROM mcr.microsoft.com/vscode/devcontainers/anaconda:0-3
FROM python:3.7.6-slim

RUN apt update && \
    apt install --no-install-recommends -y build-essential gcc && \
    apt install -y git && \
    apt clean && rm -rf /var/lib/apt/lists/*
COPY ./requirements.txt /requirements.txt
COPY ./setup.py /setup.py

RUN pip3 install --no-cache-dir -r /requirements.txt
EXPOSE 8080
