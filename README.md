Cross-lingual Complex Word Identification
=========================================

Models for cross-lingual complex word identification.

Instructions
------------

We will follow the Fork and Pull Workflow. Instructions on how this development model works are given [here](https://reflectoring.io/github-fork-and-pull/).  

Once you have cloned your fork, run `setup.py develop` from the root folder, so that the `src` module is "installed" and can be imported from anywhere in the code.

In terms of requirements (i.e. versions of python modules), we will use these:
- python == 3.6.5
- spacy == 2.0.12
- sklearn == 0.19.1
- nltk == 3.3
- pyphen==0.9.4
- pandas==0.23.3
- googletrans==2.3.0

Spacy models:
- en_core_web_lg
- es_core_news_md
- de_core_news_sm
- fr_core_news_md

These can be installed using "$ python -m spacy download <MODEL>", or downloaded from from the team Google drive (currently link sharing is unavailable).

You are advised to use python environments. For reporting results you should only use the docker image that can be built as follows:
- create your own copy of Dockerfile_ADD_USERNAME_AND_PASSWORD
- replace USERNAME and PASSWORD with yours
- build it by running `docker build -t cwi - < Dockerfile`
- get an interactive terminal on the image with `docker run -i -t cwi bash`
- run commands as you normally would (remember this is a very minimal linux installation)

If you want to run the image with a new version of the code, add the option `--no-cache` to the build.

Project Organization
--------------------

    ├── LICENSE
    ├── Makefile           <- Makefile with commands like `make data` or `make train` (we won't use use for now)
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── external       <- Data from third party sources (e.g., dictionaries, word lists, etc.).
    │   ├── interim        <- Intermediate data that has been transformed (e.g., data after processing).
    │   ├── processed      <- The final, canonical data sets for modeling (the final datasets that would not suffer further processing).
    │   └── raw            <- The original, immutable data dump.
    │
    ├── docs               <- A default Sphinx project; see sphinx-doc.org for details (don't worry about this for now)
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │                          (if the trained models are going to be stored, this is the place to save them)
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's initials, and a short `-` delimited description, e.g.
    │                         `1.0-jqp-initial-data-exploration`.
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt` (don't worry about this for now)
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── data           <- Scripts to download, generate or preprocess data
    │   │
    │   ├── features       <- Scripts to turn raw data into features for modeling
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make predictions
    │   │
    │   └── visualization  <- Scripts to create exploratory and results oriented visualizations
    │
    └── tox.ini            <- tox file with settings for running tox; see tox.testrun.org


Cross Lingual Results Testing
--------------------
The following results need to be tested for the crosslingual model.

French Test Data
----------------
----------------

Below are the command line options for testing for various combinations of Language with French as test data:

| Language Choice Combination |             Command Line Command             |
|:---------------------------:|:--------------------------------------------:|
|         English Only        |  python src/models/run_crosslingual.py -s E  |
|         Spanish Only        |  python src/models/run_crosslingual.py -s S  |
|         German Only         |  python src/models/run_crosslingual.py -s G  |
|     English and Spanish     |  python src/models/run_crosslingual.py -s ES |
|      English and German     |  python src/models/run_crosslingual.py -s EG |
|      Spanish and German     |  python src/models/run_crosslingual.py -s SG |
| English, Spanish and German | python src/models/run_crosslingual.py -s ESG |


English Test Data
-----------------
-----------------

| Language Choice Combination |                  Command Line Command                  |
|:---------------------------:|:------------------------------------------------------:|
|         Spanish Only        |  python src/models/run_crosslingual.py -s S -l english |
|         German Only         |  python src/models/run_crosslingual.py -s G -l english |
|      Spanish and German     | python src/models/run_crosslingual.py -s SG -l english |


Spanish Test Data
-----------------
-----------------

| Language Choice Combination |                  Command Line Command                  |
|:---------------------------:|:------------------------------------------------------:|
|         English Only        |  python src/models/run_crosslingual.py -s E -l spanish |
|         German Only         |  python src/models/run_crosslingual.py -s G -l spanish |
|      English and German     | python src/models/run_crosslingual.py -s EG -l spanish |


German Test Data
-----------------
-----------------

| Language Choice Combination |                  Command Line Command                  |
|:---------------------------:|:------------------------------------------------------:|
|         English Only        |  python src/models/run_crosslingual.py -s E -l german |
|         Spanish Only         |  python src/models/run_crosslingual.py -s S -l german |
|      English and Spanish     | python src/models/run_crosslingual.py -s ES -l german |




Translation Baseline
---------------------
For Translating the French Test Data to English, please use the following command

- python src/models/run_crosslingual.py -t T

For using any of the above language combinations with French as test data, use the following command line command.

In the following command line example, we use english and spanish as training data for the crosslingual model and translate the French test data to english when testing. 
- python src/models/run_crosslingual.py -s ES -t T

Similarly we can do this for other language choices mentioned above. 

--------



<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
