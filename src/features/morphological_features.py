"""
Morphological features.

This module contains functions to extract features concerning word shape or morphology

"""
from collections import Counter


def word_shape(target_word_spacy_tokens):
    """Compute the shapes of the words in target phrase

    Args:
        target_word_spacy_tokens (list): spacy tokens for the target phrase

    Returns:
        Dict. Word shapes in the target phrase
    """
    shapes = Counter()
    for token in target_word_spacy_tokens:
        shapes["SHAPE_" + token.shape_] += 1

    return shapes


def is_capitalised(target_word):
    """Binary indicator of capitalisation of the target word
    Args:
        target_word (str): word or phrase candidate
    Returns:
        Boolean: Is the first letter of the phrase uppercase?
    """

    return target_word[0].isupper()


def num_complex_punct(target_word):  # Alison
    """Compute the  number of "complex" punctuation symbols in phrase

    Args:
        target_word (str): word or phrase candidate

    Returns:
        int - the number of complex punctuations

    """
    num_punct = sum(map(target_word.lower().count, "-,;"))

    return num_punct
