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
from sklearn import preprocessing
import pandas as pd
import nltk
nltk.download("omw", quiet=True)
import pyphen

from collections import OrderedDict

from sklearn.base import BaseEstimator, TransformerMixin

from src.features import length_features
from src.features import phonetic_features
from src.features import affix_features
from src.features import char_trigram_features
from src.features import NGram_char_features
from src.features import sentence_features
from src.features import syn_and_sense_features
from src.features import word_emb_features
from src.features import syntactic_features
from src.features import morphological_features
from src.features import frequency_index_features
from src.features import stopwords
from src.features import lemma_features
from src.features import frequency_features
from src.features import hypernym_features
from src.features import noun_phrase_features
from src.features import iob_features
from src.features import probability_features
from src.features import file_io


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

class Monolingual_Feature_Extractor(BaseEstimator, TransformerMixin):
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

        if (self.language == 'english'):
            self.u_prob = file_io.read_file('data/external/english_u_prob.csv') #should be in data/external

        if (self.language == 'spanish'):
            self.u_prob = file_io.read_file('data/external/spanish_u_prob.csv')

        #if (self.language == 'german'):
         #   self.u_prob = file_io.read_file('data/external/german_u_prob.csv')
        # Loading the spacy vocab for tokenisation.
        if self.language == "english":
            self.nlp = spacy.load('en_core_web_lg')
        elif self.language == "spanish":
            self.nlp = spacy.load("es_core_news_md")
        elif self.language == "german":
            self.nlp = spacy.load('de_core_news_sm')
        elif self.language == "french":
            self.nlp = spacy.load('fr_core_news_md')

        # load pyphen stuff here #TODO

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
        # Create the tokeniser
        tokenizer = Tokenizer(self.nlp.vocab)

        spacy_token_list = []

        for target in tokenizer(spacy_target_word):
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
            result (list[dict]): List of row dictionaries containing the values
            of the extracted features for each row.
        """

        result = []

        """ Dataset level Statistics """
        """if (self.language == 'english' or self.language == 'spanish'):
            self.avg_sense_count = np.mean([syn_and_sense_features.no_synonyms(target_word, self.language) for target_word in X.target_word])
            self.avg_syn_count = np.mean([syn_and_sense_features.no_senses(target_word, self.language) for target_word in X.target_word])
        """
        # self._avg_target_phrase_len = np.mean([len(x) for x in X["spacy"]])

        """ """
        for ix, row in X.iterrows():

            """ Selecting """
            spacy_sent = row["spacy"]
            target_word = row["target_word"]
            target_sent = row["sentence"]
            spacy_tokens = self.get_spacy_tokens(spacy_sent, target_word)
            language = row['language']

            """ Feature Extraction """
            char_tri_sum, char_tri_avg = char_trigram_features.trigram_stats(target_word, language)
            rare_trigram_count = char_trigram_features.rare_trigram_count(target_word, language)
            is_stopword = stopwords.is_stop(target_word,language)
            #num_pronunciations = phonetic_features.num_pronunciations(target_word, language=language)
            rare_word_count = frequency_features.rare_word_count(target_word, language)

            row_dict = {
                    'char_tri_sum': char_tri_sum,
                    'char_tri_avg': char_tri_avg,
                    'rare_trigram_count': rare_trigram_count,
                    'is_stop':is_stopword,
                    'num_pronunciations':  num_pronunciations,
                    'rare_word_count': rare_word_count,
                    }

            """if(language == 'english' or language == 'spanish'):
                unigram_prob = probability_features.get_unigram_prob(target_word, language, self.u_prob) #also german
                row_dict['unigram_prob'] = unigram_prob
                syn_count = syn_and_sense_features.no_synonyms(target_word, language)
                sense_count = syn_and_sense_features.no_senses(target_word, language)
                row_dict.update({'syn_count': syn_count/self.avg_syn_count, 'sense_count': sense_count/self.avg_sense_count})
                word_vec = word_emb_features.get_word_emb(spacy_tokens, language)
                for i in range(word_vec.shape[0]):
                    row_dict['vec_' + str(i)] = word_vec[i] #word embedding"""

            # Spanish Frequency Index feature
            """if language == 'spanish':
                esp_freq_index_features = frequency_index_features.frequency_index(spacy_tokens, self.esp_freq_index)
                row_dict.update(esp_freq_index_features)
            """
            row_dict = OrderedDict(sorted(row_dict.items(), key=lambda t: t[0]))

            result.append(row_dict)

        return result


class Crosslingual_Feature_Extractor(BaseEstimator, TransformerMixin):
    """
    Transformer to extract crosslingual features

    """

    def __init__(self, language=None):
        """
        Define basic properties

        Args:
            language(str): language of input data
        """
        if language == 'english':
            self.spacy_models = {'english': spacy.load('en_core_web_lg')}
            self.hyph_dictionaries = {'english': pyphen.Pyphen(lang='en')}
            self.unigram_prob_dict = {'english': file_io.read_file('data/external/english_u_prob.csv')}

        elif language == 'spanish':
            self.spacy_models = {'spanish': spacy.load('es_core_news_md')}
            self.hyph_dictionaries = {'spanish': pyphen.Pyphen(lang='es')}
            self.unigram_prob_dict = {'spanish': file_io.read_file('data/external/spanish_u_prob.csv')}

        elif language == 'german':
            self.spacy_models = {'german': spacy.load('de_core_news_sm')}
            self.hyph_dictionaries = {'german': pyphen.Pyphen(lang='de')}
            self.unigram_prob_dict = {'german': file_io.read_file('data/external/german_u_prob.csv')}

        elif language == 'french':
            self.spacy_models = {'french': spacy.load('fr_core_news_md')}
            self.hyph_dictionaries = {'french': pyphen.Pyphen(lang='fr')}
            self.unigram_prob_dict = {'french': file_io.read_file('data/external/french_u_prob.csv')}

        else:
            self.spacy_models = {
                'english': spacy.load('en_core_web_lg'),
                'spanish': spacy.load("es_core_news_md"),
                'german': spacy.load('de_core_news_sm'),
                'french': spacy.load('fr_core_news_md')
            }

            self.hyph_dictionaries = {
                    'english': pyphen.Pyphen(lang='en'),
                    'spanish': pyphen.Pyphen(lang='es'),
                    'german': pyphen.Pyphen(lang='de'),
                    'french': pyphen.Pyphen(lang='fr')
            }

            self.unigram_prob_dict = {
                    'english' : file_io.read_file('data/external/english_u_prob.csv'),
                    'spanish' : file_io.read_file('data/external/spanish_u_prob.csv'),
                    'german' : file_io.read_file('data/external/german_u_prob.csv'),
                    'french': file_io.read_file('data/external/french_u_prob.csv')
            }

    def fit(self, X, *_):
        return self

    def get_spacy_tokens(self, spacy_sentence, target_word, spacy_model):
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
        # Create the tokeniser
        tokenizer = Tokenizer(spacy_model.vocab)

        spacy_token_list = []

        for target in tokenizer(target_word):
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
            result (list[dict]): List of row dictionaries containing the values
            of the extracted features for each row.
        """


        result = []

        """ Dataset level Statistics """
        avg_target_phrase_len = np.mean([len(x) for x in X["spacy"]])

        """ """
        for ix, row in X.iterrows():

            """ Selecting """
            spacy_sent = row["spacy"]
            target_word = row["target_word"]
            target_sent = row["sentence"]
            language = row["language"]
            spacy_tokens = self.get_spacy_tokens(spacy_sent,
                                                 target_word,
                                                 self.spacy_models[language])

            """ Feature Extraction """
            is_nounphrase = noun_phrase_features.is_noun_phrase(spacy_sent, target_word)
            len_tokens_norm = len(spacy_tokens)/avg_target_phrase_len
            hypernym_counts = hypernym_features.hypernym_count(target_word, language)
            len_chars_norm = length_features.character_length(target_word, language=language)
            len_tokens = length_features.token_length(target_word)
            len_syllables = phonetic_features.num_syllables(target_word, language, self.hyph_dictionaries[language])
            consonant_freq = phonetic_features.consonant_frequency(target_word)
            gr_or_lat = affix_features.greek_or_latin(target_word)
            is_capitalised = morphological_features.is_capitalised(target_word)
            num_complex_punct = morphological_features.num_complex_punct(target_word)
            averaged_chars_per_word = length_features.averaged_chars_per_word(target_word, language)
            sent_length = length_features.token_length(target_sent)

            row_dict = {
                'is_nounphrase': is_nounphrase,
                'len_tokens_norm': len_tokens_norm,
                'hypernym_count': hypernym_counts,
                'len_chars_norm': len_chars_norm,
                'len_tokens': len_tokens,
                'consonant_freq': consonant_freq,
                'gr_or_lat': gr_or_lat,
                'is_capitalised': is_capitalised,
                'num_complex_punct': num_complex_punct,
                'averaged_chars_per_word': averaged_chars_per_word,
                'sent_length' : sent_length
                'unigram_prob': probability_features.get_unigram_prob(target_word, language, self.unigram_prob_dict[language])
            }

            long_feat_dict = {
            'char_n_gram_feats': NGram_char_features.getAllCharNGrams(target_word, 6),
            'sent_n_gram_feats': sentence_features.getAllSentNGrams(target_sent, 3),
            'iob_tags': iob_features.iob_tags(spacy_tokens),
            'lemma_feats': lemma_features.lemmas(spacy_tokens),
            'bag_of_shapes': morphological_features.word_shape(spacy_tokens),
            'pos_tag_counts': syntactic_features.get_pos_counts(spacy_tokens),
            'NER_tag_counts': syntactic_features.get_ne_counts(spacy_tokens)
            }

            for feat, value_dict in long_feat_dict.items():
                row_dict.update(value_dict.items())
            #Character NGram Features
            #row_dict.update()

            # Sentence N gram features
            #row_dict.update()

            # iob tags
            #row_dict.update()

            # Bag-of-Lemmas Feature
            #row_dict.update()

            # Bag-of-shapes feature (1 word shape per word in target phrase)
            #row_dict.update()

            # Part-of-speech tag features
            #row_dict.update() #TODO check this is universal POStag

            # Named-Entity tag features
            #row_dict.update()

            row_dict = OrderedDict(sorted(row_dict.items(), key=lambda t: t[0]))

            result.append(row_dict)

        return result
