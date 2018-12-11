"""
Syntactic features.

This module contains functions to extract features concerning syntax (POS, NER) from a target word or phrase
Author: NAME

"""
from collections import Counter
def get_pos_counts(target_word_spacy_tokens):
    """
    Compute the pos counts of the target phrase

    Args:
        target_word (str): word or phrase candidate.

    Returns:
        Counter of pos tags of the phrase and their respective counts
    """
    pos = Counter(["POS|__|" + str(token.pos_) for token in target_word_spacy_tokens])
    return pos

def get_ne_counts(target_word_spacy_tokens):
    """
    Compute the NE counts of the target phrase

    Args:
        target_word (str): word or phrase candidate.

    Returns:
        Counter of NE tags of the phrase and their respective counts
    """
    ents = Counter(["NE|__|" + str(token.ent_type_) for token in target_word_spacy_tokens])
    return ents

    
