Cross-lingual Complex Word Identification
=========================================

Models for cross-lingual complex word identification.

Instructions
------------

We will follow the Fork and Pull Workflow. Instructions on how this development model works are given [here](https://reflectoring.io/github-fork-and-pull/).  

Once you have cloned your fork, run `setup.py develop` from the root folder, so that the `src` module is "installed" and can be imported from anywhere in the code.

In terms of requirements (i.e. versions of python modules), we will use the same as those from the com4513-6513 module:
- python == 3.6.3
- spacy == 2.0.4
- gensim== 3.1.0
- sklearn == 0.19.1
- nltk == 3.2.4
- numpy== 1.12.1
- torch==0.3.0.post4

and we also need:
- pyphen

You are advised to use python environments.

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


--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
