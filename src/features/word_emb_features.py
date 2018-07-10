""" Spacy-based features.

This module contains functions to extract the word embeddings from a target phrase
Author: Sanjana Khot

"""
import numpy as np
from collections import Counter

def get_word_emb(target_word_spacy_tokens, lang):
    """Gets the word embedding representing the target word.

    Args:
        target_word_spacy_tokens (list): spacy tokens for the target phrase
        lang : language of the dataset

    Returns:
        word embedding of the phrase

    """
    if (lang == 'english'):
        dim = 300 #will be different for different languages.

    word_vec = np.zeros((dim,)) #if no vector - add vector of zeros
    number_of_vecs = len([token.has_vector for token in target_word_spacy_tokens])
    if number_of_vecs > 0:
        for i in range(len(target_word_spacy_tokens)):
            current_token = target_word_spacy_tokens[i]
            if current_token.has_vector:
                word_vec = np.add(word_vec, current_token.vector)
        word_vec = np.true_divide(word_vec,number_of_vecs) #avg of word embeddings for individual tokens.
    return word_vec