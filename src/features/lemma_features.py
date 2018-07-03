""" Length-based features.

This module contains functions to extract the lemmas from a target phrase

"""

from collections import Counter

def lemmas(target_word_spacy_tokens, normalised=True):
    """Computes the lemmas the target word.

    Args:
        target_word_spacy_tokens (list): spacy tokens for the target phrase

    Returns:
        Counter. Lemmas present in the phrase

    """
    lemmas = Counter()

    for token in target_word_spacy_tokens:

        if normalised == True:
            lemmas["LEMMA_" + token.lemma_] += (1 / len(target_word_spacy_tokens))
        else:
            lemmas["LEMMA_" + token.lemma_] += 1

    return lemmas
