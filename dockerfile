FROM python:3.11.7-slim

RUN mkdir /app
COPY . /app
WORKDIR /app

RUN apt-get -y update
RUN apt-get -y upgrade

RUN pip install --no-cache-dir -r requirements.txt

# 443 https로 연결
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "80"]
