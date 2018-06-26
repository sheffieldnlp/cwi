# -*- coding: utf-8 -*-
"""
Created on Sun Jun 24 01:22:30 2018

@author: pmfin
"""

"""Baseline Model

This module contains the class(es) and functions that implement the CWI baseline model.

"""

from sklearn.linear_model import LogisticRegression
import scipy.sparse as sp
import numpy as np

from src.features import length_features as lenfeats
from enum import Enum

class Feats(Enum):
    word_length = 1
    num_tokens = 2


class Advanced(object):
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

        self.features_to_use = set([Feats.word_length, Feats.num_tokens])

    def extract_features(self, list_of_targets, list_of_sents):
        """Extracts features from a given target word or phrase.

        Args:
            target_word (str): word or phrase candidate.

        Returns:
            list. The values of the extracted features.

        """
        
        dict_features, dict_to_vect_mapping, vect_shape = self.get_all_features(list_of_targets, list_of_sents)
        vect_features, normalize_mask = self.get_vects_from_dicts(dict_features, dict_to_vect_mapping, vect_shape)
        norm_vect_features = self.normalize(vect_features, normalize_mask)
        
        
#        len_chars_norm = lenfeats.character_length(target_word, language=self.language)
#        len_tokens = lenfeats.token_length(target_word)

        return norm_vect_features
    
    def get_all_features(self, list_of_targets, list_of_sents):
        dict_features = []
        unique_features = set()
        
        for i in range(len(list_of_targets)):
            temp_features = {}
            target = list_of_targets[i]
            sent = list_of_sents[i]
            
            for feature in self.features_to_use:
                if feature == Feats.word_length:
                    temp_features[Feats.word_length] = self.get_word_length(target)
#                    print("word length")
                elif feature == Feats.num_tokens:
                    temp_features[Feats.num_tokens] = self.get_num_tokens(target)
#                    print("num tokens")
                else:
                    # Something went wrong
                    x = 0
                    
            dict_features.append(temp_features)
            
            unique_features.update([key for key, value in temp_features.items()])
        
        dict_to_vect_mapping = {}
        start_i = 0
        for feature in unique_features:
            dict_to_vect_mapping[feature] = start_i
            start_i += 1
            
        num_xs = len(list_of_targets)
        len_features = len(unique_features)
        
        vect_shape = (num_xs,len_features)
        
        return dict_features, dict_to_vect_mapping, vect_shape
    
    def get_word_length(self, target):
        num_tokens = self.get_num_tokens(target)
        
        if num_tokens == 1:
            length = len(target)
        else:
            total_length = 0
            for token in target.split(' '):
                total_length += len(token)
            length = total_length/num_tokens
        
        # Need to normalize to each language
        
        return length
    
    def get_num_tokens(self, tokens):
        num_tokens = len(tokens.split(' '))
        return num_tokens
    
    def get_vects_from_dicts(self, dict_features, dict_to_vect_mapping, vect_shape):
        
        # Could set a dtype?
        matrix = sp.dok_matrix(vect_shape)
        for target_i in range(len(dict_features)):
            features = dict_features[target_i]
            for feature in features:
                feature_j = dict_to_vect_mapping[feature]
                matrix[target_i, feature_j] = dict_features[target_i][feature]     
                
        vect_matrix = matrix.tocsr()
#        vect_matrix = matrix.todense()
        
        normalize_mask = [0,1,0,0,1]
        return vect_matrix, normalize_mask
    
    def normalize(self, vect_features, normalize_mask):
        norm_vect_features = vect_features
        return norm_vect_features

    def train(self, train_set):
        """Trains the model with the given instances.

        Args:
            train_set (list): A list of dictionaries that contain the information of each instance in the dataset.
                In particular, the target words/phrases and their gold labels.

        """

        all_targets = [(sent['target_word']) for sent in train_set]
        all_sents = [(sent['sentence']) for sent in train_set]
        all_gold_labels = [sent['gold_label'] for sent in train_set]
        
        X = self.extract_features(all_targets, all_sents)
        y = all_gold_labels

        self.model.fit(X, y)

    def predict(self, test_set):
        """Predicts the label for the given instances.

        Args:
            test_set (list): A list of dictionaries that contain the information of each instance in the dataset.
                In particular, the target words/phrases.

        Returns:
            numpy array. The predicted label for each target word/phrase.

        """
        all_targets = [(sent['target_word']) for sent in test_set]
        all_sents = [(sent['sentence']) for sent in test_set]
        
        X = self.extract_features(all_targets, all_sents)

        return self.model.predict(X)
