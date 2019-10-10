FROM ubuntu:latest
FROM python:3.6-stretch
RUN apt-get update -y
RUN apt-get install -y python-pip python-dev build-essential

COPY . /app
WORKDIR /app
RUN pip install -r requirements.txt

ENV PYTHONPATH /app/recommender/RecommenderLogic:$PYTHONPATH


ENTRYPOINT ["python", "recommender/app.py"]
EXPOSE 5000
CMD ["app.py"]
