"""Baseline Model

This module contains the class(es) and functions that implement the CWI baseline model.

"""

from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import Pipeline, FeatureUnion
from src.features import length_features as lenfeats
from src.features import phonetic_features as phonfeats
from src.features.feature_transfomers import Selector, Word_Feature_Extractor


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
        self.features_pipeline = self.join_pipelines()

    def build_pipelines(self):
        """
        Builds all feature pipelines
        Returns pipelines in format suitable for sklearn FeatureUnion
        Args:
            none
        Returns:
            list. list of ('pipeline_name', Pipeline) tuples
        """

        pipe_dict = {}
        pipe_dict['word_features'] = Pipeline([
            ('select', Selector(key="target_word")),
            ('extract', Word_Feature_Extractor(self.language)),
            ('vectorize', DictVectorizer())])

        pipe_dict['bag_of_words'] = Pipeline([
            ('select', Selector(key="target_word")),
            ('vectorize', CountVectorizer())])

        return list(pipe_dict.items())

    def join_pipelines(self):

        pipelines = self.build_pipelines()
        feature_union = Pipeline([('join pipelines', FeatureUnion(transformer_list=pipelines))])

        return feature_union



    def extract_features(self, target_word):
        """Extracts features from a given target word or phrase.

        Args:
            target_word (str): word or phrase candidate.

        Returns:
            list. The values of the extracted features.

        """
        len_chars_norm = lenfeats.character_length(target_word, language=self.language)
        len_tokens = lenfeats.token_length(target_word)
        consonant_freq = phonfeats.consonant_frequency(target_word)
        len_syllables = phonfeats.num_syllables(target_word, language=self.language)

        return [len_chars_norm, len_tokens, len_syllables, consonant_freq]

    def train(self, train_set):
        """Trains the model with the given instances.

        Args:
            train_set (list): A list of dictionaries that contain the information of each instance in the dataset.
                In particular, the target words/phrases and their gold labels.

        """

        X = self.features_pipeline.fit_transform(train_set)
        y = train_set['gold_label']
        self.model.fit(X, y)

    def predict(self, test_set):
        """Predicts the label for the given instances.

        Args:
            test_set (list): A list of dictionaries that contain the information of each instance in the dataset.
                In particular, the target words/phrases.

        Returns:
            numpy array. The predicted label for each target word/phrase.

        """

        X = self.features_pipeline.transform(test_set)

        return self.model.predict(X)
