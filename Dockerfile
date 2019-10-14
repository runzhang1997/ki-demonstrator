#FROM ubuntu:latest
#FROM python:3.7
FROM continuumio/anaconda3:2019.03
#RUN apt-get update -y
#RUN apt-get install -y python-pip python-dev build-essential

COPY . /app
WORKDIR /app
#RUN pip3 install -r requirements.txt


# SHELL /opt/conda/etc/profile.d/conda.sh
RUN conda update -n base -c defaults conda
RUN conda env create -f environment.yml python=3.7

EXPOSE 5000
CMD activate ki-demonstrator_wba_ima && python app.py

#ENV PYTHONPATH /app/recommender/RecommenderLogic:$PYTHONPATH


#ENTRYPOINT ["python", "-V"]
#RUN conda info --envs

#CMD . /opt/conda/etc/profile.d/conda.sh && source activate ki-demonstrator_wba_ima && python app.py
#CMD ["app.py"]
