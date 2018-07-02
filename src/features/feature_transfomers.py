"""
Feature Transformers

This module will contain the feature transformation classes that will be used in
our cwi model. This format allows us to extract and vectorize features at different
levels (word, subword, sentence) and of different types (word features, bag of words etc)
simultaneously. Please see the links below for more information on writing your own transformers.

Adapted from:

https://opendevincode.wordpress.com/2015/08/01/building-a-custom-python-scikit-learn-transformer-for-machine-learning/
and http://michelleful.github.io/code-blog/2015/06/20/pipelines/
and http://scikit-learn.org/stable/auto_examples/hetero_feature_union.html#sphx-glr-auto-examples-hetero-feature-union-py

"""

from sklearn.base import BaseEstimator, TransformerMixin
from src.features import length_features as lenfeats
from src.features import phonetic_features as phonfeats
from src.features import affix_features as affeats
from src.features import char_trigram_features as trifeats
from src.features import NGram_char_features as charfeats

class Selector(BaseEstimator, TransformerMixin):
    """
    Transformer to select a column from a dataframe

    """

    def __init__(self, key):
        """
        Defines the column to be extracted from dataframe

        Args:
            key(str): Dataframe column name
        """

        self.key = key

    def fit(self, X, *_):
        return self

    def transform(self, df):
        """
        Extracts the column from dataframe

        Args:
            df(DataFrame): input dataframe from which column is extracted
        """
        return df[self.key]


class Word_Feature_Extractor(BaseEstimator, TransformerMixin):
    """
    Transformer to extract word features from column of target words
    
    """

    def __init__(self, language):
        """
        Define basic properties

        Args:
            language(str): language of input data
        """
        self.language = language

    def fit(self, X, *_):
        return self

    def transform(self, X, *_):

        """Extracts features from a given target word or phrase.

        Args:
            X (Series): Column of target words and phrases

        Returns:
            result (list[dict]): List of row dictionaries containing the values of the extracted features for each row.

        This tranformer should always be followed by a DictionaryVectorizer in any pipeline which uses it.
        """

        result = []
        for target_word in X:

            len_chars_norm = lenfeats.character_length(target_word, language=self.language)
            len_tokens = lenfeats.token_length(target_word)
            consonant_freq = phonfeats.consonant_frequency(target_word)
            len_syllables = phonfeats.num_syllables(target_word, language=self.language)
            gr_or_lat = affeats.greek_or_latin(target_word)
            char_tri_sum, char_tri_avg = trifeats.trigram_stats(target_word, self.language)
            
            char_ngrams = charfeats.getAllCharNGrams(target_word, N=6)

            # dictionary to store the features in, vectorize this with DictionaryVectorizer
            row_dict = {
                    'len_chars_norm': len_chars_norm, 
                    'len_tokens': len_tokens, 
                    'len_syllables': len_syllables,
                    'consonant_freq': consonant_freq,
                    'gr_or_lat': gr_or_lat, 
                    'char_tri_sum': char_tri_sum,
                    'char_tri_avg': char_tri_avg,
                    }
            
            # Need to add these in a loop, since I don't know how many there will be:
            for ngram, count in char_ngrams.items():
                row_dict['char_ngrams__' + ngram] = count
            
#            for i in row_dict:
#                print(i, row_dict[i])
            
            result.append(row_dict)

        return result
