"""For running the baseline model

This models runs the baseline model on the datasets of all languages.

"""

import sys
path_to_parent = "../.."
sys.path.append(path_to_parent)

from src.data.dataset import Dataset
from src.models.baseline import Baseline
from src.models.evaluation import report_binary_score


def execute_baseline(language, dataset_name):
    """Trains and tests the baseline system for a particular dataset of a particular language. Reports results.

    Args:
        language: The language of the dataset.
        dataset_name: The name of the dataset (all files should have it).

    """
    print("\nBaseline Model for {} - {}.".format(language, dataset_name))

    data = Dataset(language, dataset_name)

    baseline = Baseline(language)

    baseline.train(data.train_set())

    print("\nResults on Development Data")
    predictions_dev = baseline.predict(data.dev_set())
    gold_labels_dev = [sent['gold_label'] for sent in data.dev_set()]
    print(report_binary_score(gold_labels_dev, predictions_dev))

    print("\nResults on Test Data")
    predictions_test = baseline.predict(data.test_set())
    gold_labels_test = [sent['gold_label'] for sent in data.test_set()]
    print(report_binary_score(gold_labels_test, predictions_test))

    print()


if __name__ == '__main__':
    execute_baseline("english", "News")
    execute_baseline("english", "WikiNews")
    execute_baseline("english", "Wikipedia")
    execute_baseline("spanish", "Spanish")
    execute_baseline("german", "German")


