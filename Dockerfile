FROM python:3.10-slim-bullseye
RUN apt update -y && apt install awscli -y
WORKDIR /app
COPY . /app

ARG AWS_ACCESS_KEY_ID
ARG AWS_SECRET_ACCESS_KEY

ENV AWS_ACCESS_KEY_ID=$AWS_ACCESS_KEY_ID
ENV AWS_SECRET_ACCESS_KEY=$AWS_SECRET_ACCESS_KEY

RUN pip install --upgrade pip && pip install -r requirements.txt
CMD ["python3", "main.py"]