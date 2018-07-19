"""Crosslingual Model

This module contains the class(es) and functions that implement the CWI crosslingual model.

"""
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import Pipeline, FeatureUnion
from src.features.feature_transfomers import Selector,Advanced_Extractor, Word_Feature_Extractor, Spacy_Feature_Extractor, Sentence_Feature_Extractor


class CrosslingualCWI(object):
    """
    A CWI model implementing features that for crosslingual setting of the task.

    """

    def __init__(self, languages):
        """Defines the basic properties of the model.

        Args:
            languages (list of str): languages of the data that will be used for training.

        """
        self.model = LogisticRegression()

        self.features_pipelines = {}
        for language in languages:
            self.features_pipelines[language] = self.join_pipelines(language)

    def build_pipelines(self, language):
        """
        Builds all feature pipelines
        Returns pipelines in format suitable for sklearn FeatureUnion
        Args:
            language: The language of the data.
        Returns:
            list. list of ('pipeline_name', Pipeline) tuples
        """
        pipe_dict = {}
        pipe_dict['word_features'] = Pipeline([
            ('select', Selector(key="target_word")),
            ('extract', Word_Feature_Extractor(language)),
            ('vectorize', DictVectorizer())])

        # pipe_dict['sent_features'] = Pipeline([
        #     ('select', Selector(key="sentence")),
        #     ('extract', Sentence_Feature_Extractor(language)),
        #     ('vectorize', DictVectorizer())])
        #
        # pipe_dict['bag_of_words'] = Pipeline([
        #     ('select', Selector(key="target_word")),
        #     ('vectorize', CountVectorizer())])
        #
        # # Noun Phrase, BIO Encoding, Hypernym Count. Comment to exclude.
        # # To include BIO Encoding uncomment lines in transform function of
        # # Advanced Features Extractor Class
        # pipe_dict['Advanced_Features']=Pipeline([
        #     ('select', Selector(key=["target_word", "sentence"])),
        #     ('extract', Advanced_Extractor(language)),
        #     ('vectorize', DictVectorizer())])
        #
        # # Spacy feature extraction. Uncomment to use.
        # pipe_dict['spacy_features'] = Pipeline([
        #     ('select', Selector(key=["target_word", "spacy"])),
        #     ('extract', Spacy_Feature_Extractor(language)),
        #     ('vectorize', DictVectorizer())])

        return list(pipe_dict.items())

    def join_pipelines(self, language):

        pipelines = self.build_pipelines(language)
        feature_union = Pipeline([('join pipelines', FeatureUnion(transformer_list=pipelines))])

        return feature_union

    def train(self, train_set):
        """Trains the model with the given instances.

        Args:
            train_set (list): A list of (str, dictionary) tuples that contain the language and information of each
                            instance in the dataset.

        """
        for i, (language, train_data) in enumerate(train_set):
            print(len(train_data))
            if i == 0:
                X = self.features_pipelines[language].fit_transform(train_data)
                y = train_data['gold_label']
                print("X: ", X.size)
                print("y: ", y.size)
            else:
                X_temp = self.features_pipelines[language].fit_transform(train_data)
                y_temp = train_data['gold_label']
                print("X_temp: ", X_temp.size)
                print("y_temp: ", y_temp.size)
                X = np.concatenate(X, X_temp)
                y = np.concatenate(y, y_temp)
        self.model.fit(X, y)

    def predict(self, test_set):
        """Predicts the label for the given instances.

        Args:
            test_set (list): A list of (str, dictionary) tuples that contain the language and information of each
                        instance in the dataset.

        Returns:
            numpy array. The predicted label for each target word/phrase.

        """
        for i, (language, test_data) in enumerate(test_set):
            if i == 0:
                X = self.features_pipelines[language].transform(test_data)
            else:
                np.concatenate(X, self.features_pipelines[language].transform(test_data))

        return self.model.predict(X)
