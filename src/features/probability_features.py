"""
Unigram probability features.

This module contains functions to extract probability features from a target word or phrase
Author: Sanjana Khot

"""
from collections import Counter
import math
def get_unigram_prob(target_phrase, language, u_prob):
    """
    Compute the unigram probability of the target phrase

    Args:
        target_word (str): word or phrase candidate.

    Returns:
        unigram probability (float)
    """
    prob = 1.0
    words = target_phrase.split(' ')
    for word in words:
        if word in u_prob:
            prob *= u_prob[word]
        else:
            if(language == 'english'):
                prob *= 8.611840246918683e-07 #lowest prob
            if(language == 'spanish'):
                prob *= 5.189817577912136e-06
            if(language == 'german'):
                prob *= 1.1258243849058473e-06
            if(language == 'french'):
                prob *= 1.9025884963829795e-08
    return math.log(prob)

    
