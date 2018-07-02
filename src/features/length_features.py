""" Length-based features.

This module contains functions to extract length-based features from a target word or phrase.

"""


def character_length(target_word, normalised=True, language='en'):
    """Computes the (normalised) number of characters in the target word.

    Args:
        target_word (str): word or phrase candidate.
        normalised (bool): to normalise the length or not.
        language (str): the language of the word (only used is normalised is True).
    Returns:
        float. The (normalised) number of characters.

    Raises:
        ValueError
    """

    if normalised:  # TODO: Constant values per language should not be hardcoded.
        if language == 'english':
            avg_word_length = 5.3
        elif language == 'spanish':
            avg_word_length = 6.2
        elif language == 'german':
            avg_word_length = 6.5
        else:
            raise ValueError("Language specified ({}) not supported.".format(language))
    else:
        avg_word_length = 1.0

    return len(target_word) / avg_word_length


def token_length(target_word):
    """Computes the number of tokens in the target word.

    Args:
        target_word (str): word or phrase candidate.

    Returns:
        int. The number of tokens.
    """

    return len(target_word.split(' '))  # TODO: Maybe use a proper tokenizer.
