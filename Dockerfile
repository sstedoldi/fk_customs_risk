# Dockerfile
FROM python:3.11-slim-bullseye

WORKDIR /fk_fraud_model_app

COPY requirements.txt /fk_fraud_model_app/

RUN python -m pip install --upgrade pip && pip install --no-cache-dir -r requirements.txt

COPY . /fk_fraud_model_app/

WORKDIR /fk_fraud_model_app/app

EXPOSE 5000

CMD ["python", "main.py"]