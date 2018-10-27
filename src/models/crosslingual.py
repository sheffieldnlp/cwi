"""Crosslingual Model

    This module contains the class(es) and functions that implement the CWI
    crosslingual model.

"""

from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import Pipeline, FeatureUnion
from src.features.feature_transfomers import Selector, Monolingual_Feature_Extractor, Crosslingual_Feature_Extractor
from src.features.save_load_features import save_features, load_features

class CrosslingualCWI(object):
    """
    A basic CWI model implementing simple features that serves as baseline.

    """

    def __init__(self, language):
        """Defines the basic properties of the model.

        Args:
            language (str): The language of the data.

        """
        
        """ Set this to None or 'All' to just use all available features """
#        self.features_to_use = None
        self.features_to_use = [
                'is_nounphrase',
                'len_tokens_norm',
#                'hypernym_count',
#                'len_chars_norm',
                'len_syllables',
                'len_tokens',
                'consonant_freq',
                'gr_or_lat',
                'is_capitalised',
                'num_complex_punct',
#                'averaged_chars_per_word',
                'sent_length',
                'unigram_prob',
                'char_n_gram_feats',
#                'sent_n_gram_feats',
                'iob_tags',
                'lemma_feats',
                'bag_of_shapes',
                'pos_tag_counts',
                'NER_tag_counts',
                ]
        
        self.model = LogisticRegression(random_state=0)
        self.features_pipeline = self.join_pipelines(language)
        

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

        pipe_dict['bag_of_words'] = Pipeline([
            ('select', Selector(key="target_word")),
            ('vectorize', CountVectorizer())])

        pipe_dict['crosslingual_features'] = Pipeline([
            ('select', Selector(key=["target_word", "spacy", "sentence", 'language', 'dataset_name'])),
            ('extract', Crosslingual_Feature_Extractor(features_to_use=self.features_to_use)),
            ('vectorize', DictVectorizer())])

        return list(pipe_dict.items())

    def join_pipelines(self, language):

        pipelines = self.build_pipelines(language)
        feature_union = Pipeline([('join pipelines', FeatureUnion(transformer_list=pipelines))])

        return feature_union

    def train(self, train_set):
        """Trains the model with the given instances.

        Args:
            train_set (list): A list of dictionaries that contain the information of each instance in the dataset.
                In particular, the target words/phrases and their gold labels.

        """
        X = self.features_pipeline.fit_transform(train_set)
        
#        X = load_features('train', self.language, train_set)
#        # We couldn't find the preloaded features file:
#        if X == None:
#            X = self.features_pipeline.fit_transform(train_set)
#            save_features('train', self.language, train_set, X)
            
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
        
#        X = load_features('test', self.language, test_set)
#        
#        # We couldn't find the preloaded features file:
#        if X == None:
#            X = self.features_pipeline.transform(test_set)
#            save_features('test', self.language, test_set, X)

        return self.model.predict(X)
    