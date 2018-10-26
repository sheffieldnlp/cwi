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

    def __init__(self, language, ablate=None):
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

        self.ablate = ablate

        # load pyphen stuff here #TODO

        """Build a Frequency Index reference for spanish language"""
        """if self.language == 'spanish':
            self.esp_freq_index = {}
            with open("data/external/spanish_subtitle_words_frequency_indexes.txt", "r", encoding="utf-8") as f:
                for line in f.readlines():
                    wd = line.split(",")[0]
                    FI = int(line.split(",")[1])
                    self.esp_freq_index[wd] = FI
        """

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
        if (self.language == 'english' or self.language == 'spanish'):
            self.avg_sense_count = np.mean([syn_and_sense_features.no_synonyms(target_word, self.language) for target_word in X.target_word])
            self.avg_syn_count = np.mean([syn_and_sense_features.no_senses(target_word, self.language) for target_word in X.target_word])

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
                    #'num_pronunciations':  num_pronunciations,
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
                    row_dict['vec_' + str(i)] = word_vec[i] #word embedding

            # Spanish Frequency Index feature
            if language == 'spanish':
                esp_freq_index_features = frequency_index_features.frequency_index(spacy_tokens, self.esp_freq_index)
                row_dict.update(esp_freq_index_features)
            """
            if self.ablate:
                for key in self.ablate:
                    if key in row_dict:
                        del row_dict[key]

            row_dict = OrderedDict(sorted(row_dict.items(), key=lambda t: t[0]))

            result.append(row_dict)

        return result


class Crosslingual_Feature_Extractor(BaseEstimator, TransformerMixin):
    """
    Transformer to extract crosslingual features

    """

    def __init__(self, language=None, ablate=None, features_to_use=None):
        """
        Define basic properties

        Args:
            language(str): language of input data
            features_to_use: a list of string named features to use
        """
        # This dict contains all available features, along with a list of their 
        # high-computing-power requirements e.g. ['spacy']
        feature_requirements = {
                # TODO: Actually fill in these requirements. At the moment, I'm just putting everything as all requirements.
                'is_nounphrase' : ['spacy'],
                'len_tokens_norm': ['spacy'],
                'hypernym_count': None,
                'len_chars_norm': None,
                'len_tokens': None,
                'len_syllables': ['hyph'],
                'consonant_freq': None,
                'gr_or_lat': ['affix'],
                'is_capitalised': None,
                'num_complex_punct': None,
                'averaged_chars_per_word': None,
                'sent_length': None,
                'unigram_prob': ['unigram_probs'],
                'char_n_gram_feats': None,
                'sent_n_gram_feats': None,
                'iob_tags': ['spacy'],
                'lemma_feats': ['spacy'],
                'bag_of_shapes': ['spacy'],
                'pos_tag_counts': ['spacy'],
                'NER_tag_counts': ['spacy'],
                }
        
        if features_to_use == None or features_to_use == 'all':
            features_to_use = list(feature_requirements.keys())
        
        # Total requirements is a unique list of all the requirements.
        self.total_requirements = set()
        final_features = []
        for feature in features_to_use:
            
            # Making sure that we know about the feature
            if feature in feature_requirements.keys():
                if feature_requirements[feature] is not None:
                    for requirement in feature_requirements[feature]:   
                        self.total_requirements.add(requirement)
                final_features.append(feature)
            else:
                print("{} did not match any of the features in feature_requirements, so was not used.".format(feature))
                
        
        
        self.features_to_use = final_features
        
        self.affixes = {}
        self.spacy_models = {'english':None,'spanish':None,'german':None,'french':None}
        self.hyph_dictionaries = {'english':None,'spanish':None,'german':None,'french':None}
        self.unigram_prob_dict = {'english':None,'spanish':None,'german':None,'french':None}
        
        # So that we're only opening this file once.
        if 'affix' in self.total_requirements:
            self.affixes = affix_features.get_affixes()
        
        
        if language == 'english':
            if 'spacy' in self.total_requirements:
                self.spacy_models = {'english': spacy.load('en_core_web_lg')}
            
            if 'hyph' in self.total_requirements:
                self.hyph_dictionaries = {'english': pyphen.Pyphen(lang='en')}
                
            if 'unigram_probs' in self.total_requirements:
                self.unigram_prob_dict = {'english': file_io.read_file('data/external/english_u_prob.csv')}
            

        elif language == 'spanish':
            if 'spacy' in self.total_requirements:
                self.spacy_models = {'spanish': spacy.load('es_core_news_md')}
                
            if 'hyph' in self.total_requirements:
                self.hyph_dictionaries = {'spanish': pyphen.Pyphen(lang='es')}
                
            if 'unigram_probs' in self.total_requirements:
                self.unigram_prob_dict = {'spanish': file_io.read_file('data/external/spanish_u_prob.csv')}

        elif language == 'german':
            if 'spacy' in self.total_requirements:
                self.spacy_models = {'german': spacy.load('de_core_news_sm')}
                
            if 'hyph' in self.total_requirements:
                self.hyph_dictionaries = {'german': pyphen.Pyphen(lang='de')}
                
            if 'unigram_probs' in self.total_requirements:
                self.unigram_prob_dict = {'german': file_io.read_file('data/external/german_u_prob.csv')}

        elif language == 'french':
            if 'spacy' in self.total_requirements:
                self.spacy_models = {'french': spacy.load('fr_core_news_md')}
                
            if 'hyph' in self.total_requirements:
                self.hyph_dictionaries = {'french': pyphen.Pyphen(lang='fr')}
                
            if 'unigram_probs' in self.total_requirements:    
                self.unigram_prob_dict = {'french': file_io.read_file('data/external/french_u_prob.csv')}

        else:
            if 'spacy' in self.total_requirements:
                self.spacy_models = {
                    'english': spacy.load('en_core_web_lg'),
                    'spanish': spacy.load("es_core_news_md"),
                    'german': spacy.load('de_core_news_sm'),
                    'french': spacy.load('fr_core_news_md')
                }

            if 'hyph' in self.total_requirements:
                self.hyph_dictionaries = {
                        'english': pyphen.Pyphen(lang='en'),
                        'spanish': pyphen.Pyphen(lang='es'),
                        'german': pyphen.Pyphen(lang='de'),
                        'french': pyphen.Pyphen(lang='fr')
                }
            
            if 'unigram_probs' in self.total_requirements:
                self.unigram_prob_dict = {
                        'english' : file_io.read_file('data/external/english_u_prob.csv'),
                        'spanish' : file_io.read_file('data/external/spanish_u_prob.csv'),
                        'german' : file_io.read_file('data/external/german_u_prob.csv'),
                        'french': file_io.read_file('data/external/french_u_prob.csv')
                }
                

                
            
                
        self.ablate = ablate

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
        
        # This is a mapping, which allows the system to not run functions that
        # aren't being used.
        feature_names_to_funcs = {
                'is_nounphrase': noun_phrase_features.is_noun_phrase,
                'len_tokens_norm': length_features.token_length_norm,
                'hypernym_count': hypernym_features.hypernym_count,
                'len_chars_norm': length_features.character_length,
                'len_tokens': length_features.token_length,
                'len_syllables': phonetic_features.num_syllables,
                'consonant_freq': phonetic_features.consonant_frequency,
                'gr_or_lat': affix_features.greek_or_latin,
                'is_capitalised': morphological_features.is_capitalised,
                'num_complex_punct': morphological_features.num_complex_punct,
                'averaged_chars_per_word': length_features.averaged_chars_per_word,
                'sent_length' : length_features.token_length,
                'unigram_prob': probability_features.get_unigram_prob,
                
                # These are long features, maybe worth considering differently?
                'char_n_gram_feats': NGram_char_features.getAllCharNGrams,
                'sent_n_gram_feats': sentence_features.getAllSentNGrams,
                'iob_tags': iob_features.iob_tags,
                'lemma_feats': lemma_features.lemmas,
                'bag_of_shapes': morphological_features.word_shape,
                'pos_tag_counts': syntactic_features.get_pos_counts,
                'NER_tag_counts': syntactic_features.get_ne_counts    
                }

        for ix, row in X.iterrows():

            """ Selecting """
            spacy_sent = row["spacy"]
            target_word = row["target_word"]
            target_sent = row["sentence"]
            language = row["language"]
            
            if 'spacy' in self.total_requirements:
                spacy_tokens = self.get_spacy_tokens(spacy_sent,
                                                     target_word,
                                                     self.spacy_models[language])
            else:
                spacy_tokens = []
                
            func_params = {
                    'is_nounphrase': (spacy_sent, target_word),
                    'len_tokens_norm': (spacy_tokens, avg_target_phrase_len),
                    'hypernym_count': (target_word, language),
                    'len_chars_norm': (target_word, language),
                    'len_tokens': (target_word),
                    'len_syllables': (target_word, language, self.hyph_dictionaries[language]),
                    'consonant_freq':(target_word),
                    'gr_or_lat': (target_word, self.affixes),
                    'is_capitalised': (target_word),
                    'num_complex_punct': (target_word),
                    'averaged_chars_per_word': (target_word, language),
                    'sent_length' : (target_sent),
                    'unigram_prob': (target_word, language, self.unigram_prob_dict[language]),
                    
                    # These are long features, maybe worth considering differently?
                    'char_n_gram_feats': (target_word, 6),
                    'sent_n_gram_feats': (target_sent, 3),
                    'iob_tags': (spacy_tokens),
                    'lemma_feats': (spacy_tokens),
                    'bag_of_shapes': (spacy_tokens),
                    'pos_tag_counts': (spacy_tokens),
                    'NER_tag_counts': (spacy_tokens)    
                    }
                
            
            
            """ Feature Extraction """
            # Long features are those that output whole dictionaries of unknown size
            long_features = [
                    'char_n_gram_feats',
                    'sent_n_gram_feats',
                    'iob_tags',
                    'lemma_feats',
                    'bag_of_shapes',
                    'pos_tag_counts',
                    'NER_tag_counts'
                ]
            
            row_dict = {}
            for feature in self.features_to_use:
                
                feature_params = func_params[feature]
                    
                if feature in long_features:
                    if isinstance(feature_params, tuple):
                        row_result = feature_names_to_funcs[feature](*feature_params).items()
                    else:
                        row_result = feature_names_to_funcs[feature](feature_params).items()
                    row_dict.update(row_result)
                else:
                    if isinstance(feature_params, tuple):
                        row_dict[feature] = feature_names_to_funcs[feature](*feature_params)
                    else:
                        row_dict[feature] = feature_names_to_funcs[feature](feature_params)


            if self.ablate:
                for key in self.ablate:
                    if key in row_dict:
                        del row_dict[key]

            row_dict = OrderedDict(sorted(row_dict.items(), key=lambda t: t[0]))

            result.append(row_dict)

        return result
