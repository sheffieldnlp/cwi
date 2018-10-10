import argparse
from src.data.dataset import Dataset
from src.models.crosslingual import CrosslingualCWI
from src.models.evaluation import report_binary_score
import pandas as pd


datasets_per_language = {"english": ["News", "WikiNews", "Wikipedia"],
                         "spanish": ["Spanish"],
                         "german": ["German"],
                         "french": ["French"]}


def run_model(test_language, evaluation_split, detailed_report):
    """ Trains the CWI model in all languages but one. Tests on all datasets of
        a particular language. Reports results.

    Args:
        test_language:      The language of the dataset to use for testing.
                            evaluation_split: The split of the data to use for
                            evaluating the performance of the model (dev or
                            test).

        detailed_report:    Whether to display a detailed report or just overall
                            score.

    """

    # collect the training data for all the languages but one
    train_data = []
    for language, datasets_names in datasets_per_language.items():
        if language != test_language:
            for dataset_name in datasets_names:
                data = Dataset(language, dataset_name)
                lang_train_set = data.train_set()
                if lang_train_set is None:
                    print("No training data found for language {}.".format(language))
                else:
                    train_data.append(lang_train_set)

    train_data = pd.concat(train_data)

    # train the CWI model
    cwi_model = CrosslingualCWI(list(datasets_per_language.keys()))
    cwi_model.train(train_data)


    # test the model
    test_datasets = datasets_per_language[test_language]

    for dataset_name in test_datasets:
        data = Dataset(test_language, dataset_name)

        print("\nTesting on  {} - {}.".format(test_language, dataset_name))

        if evaluation_split in ["dev", "both"]:
            print("\nResults on Development Data")
            predictions_dev = cwi_model.predict(data.dev_set())
            gold_labels_dev = data.dev_set()['gold_label']
            print(report_binary_score(gold_labels_dev, predictions_dev, detailed_report))

        if evaluation_split in ["test", "both"]:
            print("\nResults on Test Data")
            predictions_test = cwi_model.predict(data.test_set())
            gold_labels_test = data.test_set()['gold_label']
            print(report_binary_score(gold_labels_test, predictions_test, detailed_report))

    print()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Trains the model in all languages but one, and tests on the left out language.")
    parser.add_argument('-l', '--language', choices=['english', 'spanish', 'german', 'french'], default='french',
                        help="language of the dataset(s) where to test the model.")
    parser.add_argument('-e', '--eval_split', choices=['dev', 'test', 'both'], default='test',
                        help="the split of the data to use for evaluating performance")
    parser.add_argument('-d', '--detailed_report', action='store_true',
                        help="to present a detailed performance report per label.")
    args = parser.parse_args()

    run_model(args.language, args.eval_split, args.detailed_report)
