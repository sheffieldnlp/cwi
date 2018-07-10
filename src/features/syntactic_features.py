"""
Syntactic features.

This module contains functions to extract features concerning syntax (POS, NER) from a target word or phrase
Author: Sanjana Khot

"""
from collections import Counter
def get_pos_counts(target_word_spacy_tokens):
    """
    Compute the frequency of consonants in the target word

    Args:
        target_word (str): word or phrase candidate.

    Returns:
        Counter of pos tags of the phrase and their respective counts
    """

    pos = Counter([token.pos_ for token in target_word_spacy_tokens])

    return pos

    
