# -*- coding: utf-8 -*-
"""
Created on Sun Jun 24 01:36:51 2018

@author: pmfin
"""

"""For running the baseline model

This models runs the baseline model on the datasets of all languages.

"""

from src.data.dataset import Dataset
from src.models.advanced import Advanced
from src.models.evaluation import report_binary_score


def execute_advanced(language, dataset_name):
    """Trains and tests the baseline system for a particular dataset of a particular language. Reports results.

    Args:
        language: The language of the dataset.
        dataset_name: The name of the dataset (all files should have it).

    """
    print("\nAdvanced Model for {} - {}.".format(language, dataset_name))

    data = Dataset(language, dataset_name)

    advanced = Advanced(language)

    advanced.train(data.train_set())

    print("\nResults on Development Data")
    predictions_dev = advanced.predict(data.dev_set())
    gold_labels_dev = [sent['gold_label'] for sent in data.dev_set()]
    print(report_binary_score(gold_labels_dev, predictions_dev))

    print("\nResults on Test Data")
    predictions_test = advanced.predict(data.test_set())
    gold_labels_test = [sent['gold_label'] for sent in data.test_set()]
    print(report_binary_score(gold_labels_test, predictions_test))

    print()


if __name__ == '__main__':


    execute_advanced("english", "News")
    execute_advanced("english", "WikiNews")
    execute_advanced("english", "Wikipedia")
    execute_advanced("spanish", "Spanish")
    execute_advanced("german", "German")


