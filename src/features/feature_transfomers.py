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

import numpy as np
import spacy
from spacy.tokenizer import Tokenizer


from sklearn.base import BaseEstimator, TransformerMixin
from src.features import length_features as lenfeats
from src.features import phonetic_features as phonfeats
from src.features import affix_features as affeats
from src.features import char_trigram_features as trifeats
from src.features import NGram_char_features as charfeats
from src.features import sentence_features as sentfeats
from src.features import syn_and_sense_features as synsenfeats
from src.features import morphological_features as morphfeats
from src.features import frequency_index_features as freqixfeats

from src.features import lemma_features as lemmafeats

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

    def __init__(self, language, maxCharNgrams=6, normaliseSynsenFeats=True):
        """
        Define basic properties

        Args:
            language(str): language of input data
            maxCharNgrams(int): Extract 1 to N length Character NGrams and
                                suffixes and prefixes (e.g. 2 = 'ch')
        """
        self.language = language
        self.maxCharNgrams = maxCharNgrams
        self.normaliseSynsenFeats = normaliseSynsenFeats

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

        """Gathering normalisation information from the whole dataset"""
        if (self.language == 'english' or self.language == 'spanish'):
            if self.normaliseSynsenFeats == True:
                self.avg_sense_count = np.mean([synsenfeats.no_synonyms(target_word, self.language) for target_word in X])
                self.avg_syn_count = np.mean([synsenfeats.no_senses(target_word, self.language) for target_word in X])


        for target_word in X:

            len_chars_norm = lenfeats.character_length(target_word, language=self.language)
            len_tokens = lenfeats.token_length(target_word)
            consonant_freq = phonfeats.consonant_frequency(target_word)
            len_syllables = phonfeats.num_syllables(target_word, language=self.language)
            gr_or_lat = affeats.greek_or_latin(target_word)
            char_tri_sum, char_tri_avg = trifeats.trigram_stats(target_word, self.language)
            is_capitalised = morphfeats.is_capitalised(target_word)



            char_ngrams = charfeats.getAllCharNGrams(target_word, self.maxCharNgrams)



            # dictionary to store the features in, vectorize this with DictionaryVectorizer
            row_dict = {
                    'len_chars_norm': len_chars_norm,
                    'len_tokens': len_tokens,
                    'len_syllables': len_syllables,
                    'consonant_freq': consonant_freq,
                    'gr_or_lat': gr_or_lat,
                    'char_tri_sum': char_tri_sum,
                    'char_tri_avg': char_tri_avg,
                    'is_capitalised': is_capitalised
                    }

            if (self.language == 'english' or self.language == 'spanish'):
                syn_count = synsenfeats.no_synonyms(target_word, self.language)
                sense_count = synsenfeats.no_senses(target_word, self.language)

                if self.normaliseSynsenFeats == True: # Normalisation
                    row_dict.update({'syn_count': syn_count/self.avg_syn_count, 'sense_count': sense_count/self.avg_sense_count})
                elif self.normaliseSynsenFeats == False:
                    row_dict.update({'syn_count': syn_count, 'sense_count': sense_count})

            # Need to add these in a loop, since I don't know how many there will be:
            for ngram, count in char_ngrams.items():
                row_dict['char_ngrams__' + ngram] = count

            result.append(row_dict)

        return result

class Spacy_Feature_Extractor(BaseEstimator, TransformerMixin):
    """
    Transformer to extract word features from dataframe of target words and
    spacy docs

    """

    def __init__(self, language):
        """
        Define basic properties

        Args:
            language(str): language of input data
        """
        self.language = language

        # Loading the spacy vocab for tokenisation.
        if self.language == "english":
            self.nlp = spacy.load('en_core_web_lg')
        elif self.language == "spanish":
            self.nlp = spacy.load("es_core_news_md")
        elif self.language == "german":
            self.nlp = spacy.load('de_core_news_sm')
        elif self.language == "french":
            self.nlp = spacy.load('fr_core_news_md')

        # Create the tokeniser
        self.tokenizer = Tokenizer(self.nlp.vocab)

        """Build a Frequency Index reference for spanish language"""
        if self.language == 'spanish':
            self.esp_freq_index = {}
            with open("data/external/spanish_subtitle_words_frequency_indexes.txt", "r", encoding="utf-8") as f:
                for line in f.readlines():
                    wd = line.split(",")[0]
                    FI = int(line.split(",")[1])
                    self.esp_freq_index[wd] = FI

    def fit(self, X, *_):
        return self

    def get_spacy_tokens(self, spacy_sentence, spacy_target_word):
        """
        A function to locate the target phrase spacy tokens in a spacy doc of
        a whole sentence.

        Args:
            spacy_sentence: spacy doc for the context sentence
            spacy_target_word: spacy doc for the target word/phrase only

        Returns:
            spacy_token_list: a list of the spacy tokens for the target phrase,
                              using the information from the context sentence.

        """
        spacy_token_list = []

        for target in self.tokenizer(spacy_target_word):
            for wd in spacy_sentence:
                if target.text == wd.text:
                    spacy_token_list.append(wd)
                    break

        return spacy_token_list


    def transform(self, X, *_):

        """Extracts features from a given target word and context.

        Args:
            X (DataFrame): Columns of target words and spacy docs

        Returns:
            result (list[dict]): List of row dictionaries containing the values of the extracted features for each row.

        This tranformer should always be followed by a DictionaryVectorizer in any pipeline which uses it.
        """

        result = []

        self._avg_target_phrase_len = np.mean([len(x) for x in X["spacy"]])

        for x in X.iterrows():

            # Reference the spacy doc and the target word separately
            spacy_sent = x[1]["spacy"]
            target_word = x[1]["target_word"]

            # Look up the spacy tokens of the target word.
            spacy_tokens = self.get_spacy_tokens(spacy_sent, target_word)

            # Extract features
            len_tokens_norm = len(spacy_tokens)/self._avg_target_phrase_len

            row_dict = {
                    'len_tokens_norm': len_tokens_norm,
                    }

            # Bag-of-Lemmas Feature #TODO there is probably a better way of doing this. Dictionary union?
            lemma_features = lemmafeats.lemmas(spacy_tokens)
            for lemma, count in lemma_features.items():
                row_dict[lemma] = count

            # Spanish Frequency Index feature #TODO there is probably a better way of doing this. Dictionary union?
            if self.language == 'spanish':
                esp_freq_index_features = freqixfeats.frequency_index(spacy_tokens, self.esp_freq_index)
                for k, v in esp_freq_index_features.items():
                    row_dict[k] = v

            result.append(row_dict)

        return result

class Sentence_Feature_Extractor(BaseEstimator, TransformerMixin):
    """
    Transformer to extract sentence features from column of sentences

    """

    def __init__(self, language, maxSentNGram = 3):
        """
        Define basic properties

        Args:
            language(str): language of input data
        """
        self.language = language
        self.maxSentNGram = maxSentNGram

    def fit(self, X, *_):
        return self

    def transform(self, X, *_):

        """Extracts features from a given target sentence.

        Args:
            X (Series): Column of target sentencess

        Returns:
            result (list[dict]): List of row dictionaries containing the values of the extracted features for each row.

        This tranformer should always be followed by a DictionaryVectorizer in any pipeline which uses it.
        """

        result = []
        for target_sent in X:
            sent_length = lenfeats.token_length(target_sent)
            sent_NGrams = sentfeats.getAllSentNGrams(target_sent, self.maxSentNGram)

            # dictionary to store the features in, vectorize this with DictionaryVectorizer
            row_dict = {
                    'sent_length' : sent_length,
                    }

            for ngram, count in sent_NGrams.items():
                row_dict['sent_ngrams__' + ngram] = count

            result.append(row_dict)

        return result
