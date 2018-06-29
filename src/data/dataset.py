"""Dataset Reader

This module contains the class(es) and functions to read the datasets.

"""
import csv


class Dataset(object):
    """
    Utility class to easily load the datasets for training, development and testing.
    """

    def __init__(self, language, dataset_name):
        """Defines the basic properties of the dataset reader.

        Args:
            language: The language of the dataset.
            dataset_name: The name of the dataset (all files should have it).

        """
        self._language = language
        self._dataset_name = dataset_name

        # TODO: Maybe the paths should be passed as parameters or read from a configuration file.
        self._trainset_path = "data/raw/{}/{}_Train.tsv".format(language.lower(), dataset_name)
        self._devset_path = "data/raw/{}/{}_Dev.tsv".format(language.lower(), dataset_name)
        self._testset_path = "data/raw/{}/{}_Test.tsv".format(language.lower(), dataset_name)

        self._trainset = None
        self._devset = None
        self._testset = None

    def train_set(self):
        """list. Getter method for the training set. """
        if not self._trainset:  # loads the data to memory once and when requested.
            self._trainset = self.read_dataset(self._trainset_path)
        return self._trainset

    def dev_set(self):
        """list. Getter method for the development set. """
        if not self._devset:  # loads the data to memory once and when requested.
            self._devset = self.read_dataset(self._devset_path)
        return self._devset

    def test_set(self):
        """list. Getter method for the test set. """
        if not self._testset:  # loads the data to memory once and when requested.
            self._testset = self.read_dataset(self._testset_path)
        return self._testset

    def read_dataset(self, file_path):
        """Read the dataset file.

        Args:
            file_path (str): The path of the dataset file. The file should follow the structure specified in the
                    2018 CWI Shared Task.

        Returns:
            list. A list of dictionaries that contain the information of each sentence in the dataset.

        """
        with open(file_path, encoding = "utf-8") as file:
            fieldnames = ['hit_id', 'sentence', 'start_offset', 'end_offset', 'target_word', 'native_annots',
                          'nonnative_annots', 'native_complex', 'nonnative_complex', 'gold_label', 'gold_prob']
            reader = csv.DictReader(file, fieldnames=fieldnames, delimiter='\t')

            dataset = [sent for sent in reader]

        return dataset