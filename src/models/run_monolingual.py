"""For running the baseline model

This models runs the baseline model on the datasets of all languages.

"""

import argparse

from src.data.dataset import Dataset
from src.models.monolingual import Baseline
from src.models.evaluation import report_binary_score


datasets_per_language = {"english": ["News", "WikiNews", "Wikipedia"],
                         "spanish": ["Spanish"],
                         "german": ["German"]}


def run_model(language, dataset_name, evaluation_split, detailed_report):
    """Trains and tests the CWI model for a particular dataset of a particular language. Reports results.

    Args:
        language: The language of the dataset.
        dataset_name: The name of the dataset (all files should have it).
        evaluation_split: The split of the data to use for evaluating the performance of the model (dev or test).
        detailed_report: Whether to display a detailed report or just overall score.

    """
    print("\nModel for {} - {}.".format(language, dataset_name))

    data = Dataset(language, dataset_name)

    baseline = Baseline(language)

    baseline.train(data.train_set())

    if evaluation_split in ["dev", "both"]:
        print("\nResults on Development Data")
        predictions_dev = baseline.predict(data.dev_set())
        gold_labels_dev = data.dev_set()['gold_label']
        print(report_binary_score(gold_labels_dev, predictions_dev, detailed_report))

    if evaluation_split in ["test", "both"]:
        print("\nResults on Test Data")
        predictions_test = baseline.predict(data.test_set())
        gold_labels_test = data.test_set()['gold_label']
        print(report_binary_score(gold_labels_test, predictions_test, detailed_report))

    print()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Trains and tests the model for all datasets of a language.")
    parser.add_argument('-l', '--language', choices=['english', 'spanish', 'german'], default='english',
                        help="language of the dataset(s).")
    parser.add_argument('-e', '--eval_split', choices=['dev', 'test', 'both'], default='test',
                        help="the split of the data to use for evaluating performance")
    parser.add_argument('-d', '--detailed_report', action='store_true',
                        help="to present a detailed performance report per label.")
    args = parser.parse_args()

    datasets = datasets_per_language[args.language]

    for dataset_name in datasets:
        run_model(args.language, dataset_name, args.eval_split, args.detailed_report)
