"""
Morphological features.

This module contains functions to extract features concerning word shape or morphology

"""
import spacy


def word_shape(target_word, language):
    """Compute the shape of the target word

    Args:
        target_word (str): word or phrase candidate

    Returns:
        str - the shape of the word

    Raises:
        ValueError
    """

    if language == 'english':
        nlp = spacy.load('en')

    elif language == 'spanish':
        nlp = spacy.load('es')

    elif language == 'german':
        nlp = spacy.load('de')

    else:
        raise ValueError("Language specified ({}) not supported.".format(language))

    doc = nlp(u'{}'.format(target_word))

    token = doc[0]

    return token.shape_


def is_capitalised(target_word):
    """Binary indicator of capitalisation of the target word

    Args:
        target_word (str): word or phrase candidate

    Returns:
        Boolean: Is the first letter of the phrase uppercase?

    """

    return target_word[0].isupper()
