# Needs a VM with 4G
FROM python:3.6.5
MAINTAINER Andreas Vlachos <a.vlachos@sheffield.ac.uk>

RUN apt-get update -y
RUN apt-get install -y git

RUN git clone http://github.com/sheffieldnlp/cwi.git
WORKDIR /cwi
RUN pip install -r requirements_docker.txt

RUN python setup.py develop

RUN python -m spacy download de_core_news_sm
RUN python -m spacy download es_core_news_md
RUN python -m spacy download en_core_web_lg
RUN python -m spacy download fr_core_news_md

RUN python -m nltk.downloader wordnet

RUN python src/data/build_spacy_objects.py english
RUN python src/data/build_spacy_objects.py spanish
RUN python src/data/build_spacy_objects.py german
RUN python src/data/build_spacy_objects.py french
