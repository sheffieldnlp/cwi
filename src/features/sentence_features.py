# -*- coding: utf-8 -*-
"""
Created on Tue Jul  3 19:40:37 2018

@author: pmfin
"""

from src.features import word_normalization as wn
from collections import Counter

def getAllSentNGrams(target_sent, maxSentNGram):
    """Extracts NGrams up to maxSentNgram 

    Args:
        target_sent(str): string containing words
        maxSentNgram(int): highest NGram level to extract (e.g. 3 = unigrams, 
                                                            bigrams & trigrams)

    Returns:
        result (Counter()): Counter dictionary of all the NGrams
    """
    result = Counter()
    norm_tokens = wn.normalize_sent(target_sent)
    for i in range(maxSentNGram):
        n = i + 1
        result.update(getSentNGrams(norm_tokens, n))
    return result

def getSentNGrams(norm_tokens, N):
    """Extracts NGrams

    Args:
        norm_tokens(list[str]): list of normalized tokens
        N(int): NGram to extract (e.g. 3 = trigrams)

    Returns:
        result (list[str]): List of the Ngrams
    """
    tokens_to_process = norm_tokens
    if N > 1:
        tokens_to_process.insert(0, "__START__")
        tokens_to_process.append("__END__")
        
    result = ['+'.join(tokens_to_process[i:i+N]) for i in range(len(tokens_to_process) - N + 1)]
    result = [str(N)+"-Gram|__|"+x for x in result]

    return result

    
if __name__ == '__main__':
    sent = "What is going on in this place? How did I get here? Why are there so many clowns?"
    print(getAllSentNGrams(sent, 3))