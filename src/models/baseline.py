"""Baseline Model

This module contains the class(es) and functions that implement the CWI baseline model.

"""

from sklearn.linear_model import LogisticRegression
from src.features import length_features as lenfeats
from src.features import syn_and_sense_features as synsenfeats
from src.features import affix_features as affixfeats


class Baseline(object):
    """
    A basic CWI model implementing simple features that serves as baseline.

    """

    def __init__(self, language):
        """Defines the basic properties of the model.

        Args:
            language (str): A two letter word that specifies the language of the data.

        """
        self.language = language
        self.model = LogisticRegression()
        self.affixes=[]
        with open('data/external/greek_and_latin_affixes.txt') as f:
            for line in f:
                self.affixes.append(line.replace("\n", ""))

    def extract_features(self, target_word):
        """Extracts features from a given target word or phrase.

        Args:
            target_word (str): word or phrase candidate.

        Returns:
            list. The values of the extracted features.

        """
        len_chars_norm = lenfeats.character_length(target_word, language=self.language)
        len_tokens = lenfeats.token_length(target_word)
        ##no_syns = synsenfeats.no_synonyms(target_word, self.language)
        #no_sens = synsenfeats.no_senses(target_word, self.language)
        gr_lat = affixfeats.greek_or_latin(target_word, self.affixes)

        return [len_chars_norm, len_tokens, gr_lat]

    def train(self, train_set):
        """Trains the model with the given instances.

        Args:
            train_set (list): A list of dictionaries that contain the information of each instance in the dataset.
                In particular, the target words/phrases and their gold labels.

        """
        X = []  # to store the extracted features
        y = []  # to store the gold labels
        for sent in train_set:
            X.append(self.extract_features(sent['target_word']))
            y.append(sent['gold_label'])

        self.model.fit(X, y)

    def predict(self, test_set):
        """Predicts the label for the given instances.

        Args:
            test_set (list): A list of dictionaries that contain the information of each instance in the dataset.
                In particular, the target words/phrases.

        Returns:
            numpy array. The predicted label for each target word/phrase.

        """
        X = []  # to store the extracted features
        for sent in test_set:
            X.append(self.extract_features(sent['target_word']))

        return self.model.predict(X)
