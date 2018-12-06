""" IOB encoding features.

"""

from collections import Counter
def iob_tags(target_word_spacy_tokens, normalised=True):
    """

    Args:
        Spacy tokens for target phrase


    Returns:
        dictionary.

    """

    iob_tags = Counter()

    for token in target_word_spacy_tokens:

        if normalised == True:
            iob_tags["IOB|__|" + token.ent_iob_] += (1 / len(target_word_spacy_tokens))
        else:
            iob_tags["IOB|__|" + token.ent_iob_] += 1

    return iob_tags
