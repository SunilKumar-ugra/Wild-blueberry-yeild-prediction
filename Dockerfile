FROM python:3.9-slim-buster
RUN apt update -y && apt install awscli -y
WORKDIR /app
COPY . /app
CMD ["python3","app.py"]