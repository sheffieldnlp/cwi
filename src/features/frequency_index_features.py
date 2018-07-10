""" Frequency Index Features

"""

from collections import Counter

def frequency_index(target_word_spacy_tokens, frequency_index):
    """Looks up the frequency index (FI: 0-6) of each word in the target phrase

    Args:
        target_word_spacy_tokens (list) : list of spacy objects

    Returns:
        counter. counts of the frequency indexes of the lemmas of the tokens. 

    """
    freq_indexes = Counter()

    for token in target_word_spacy_tokens:

        if token.lemma_ in frequency_index:
            x_features["FI_" + str(frequency_index[token.lemma_])] += (1 / len(target_word_spacy_tokens))
        else:
            x_features["FI_0"] += (1 / len(target_word_spacy_tokens))

    return freq_indexes
