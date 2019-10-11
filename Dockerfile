#FROM ubuntu:latest
#FROM python:3.7
FROM continuumio/anaconda3:latest
#RUN apt-get update -y
#RUN apt-get install -y python-pip python-dev build-essential

COPY . /app
WORKDIR /app
# RUN pip3 install -r requirements.txt
RUN conda env create -f environment.yml
RUN activate ki-demonstrator_wba_ima

ENV PYTHONPATH /app/recommender/RecommenderLogic:$PYTHONPATH


ENTRYPOINT ["python", "recommender/app.py"]
EXPOSE 5000
CMD ["app.py"]
