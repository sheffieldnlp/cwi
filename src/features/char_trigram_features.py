""" Character trigram-based features.

This module contains functions to find the frequencies of the trigrams contained in a target word.

"""
from nltk import ngrams
import pickle

with open('data/external/german_tr_freqs.pkl', 'rb') as gtf:
                german_tf = pickle.load(gtf)
with open('data/external/english_tr_freqs.pkl', 'rb') as etf:
                english_tf = pickle.load(etf)
with open('data/external/spanish_tr_freqs.pkl', 'rb') as stf:
                spanish_tf = pickle.load(stf)

def trigram_stats(target_word, language):
    """Computes the sum of the absolute frequencies of the character trigrams in the target word as well as their average.

    Args:
        target_word (str): word or phrase candidate.
        language (str): target language

    Returns:
        (int, int). Sum and average of the frequencies in a large corpus of the trigrams of the target word.
    
    Raises:
        Value Error
    """
    if language == 'english':
       freqs = english_tf
    elif language == 'spanish':
        freqs = spanish_tf
    elif language == 'german':
        freqs = german_tf
    else:
        raise ValueError("Language specified ({}) not supported.".format(language))
        
    fr_sum = 0    
    target_trigrams = ['{}{}{}'.format(t[0],t[1], t[2]) for t in list(ngrams(target_word, 3))]
    for tr in target_trigrams:
        fr_sum += freqs[tr]
    average = 0 if len(target_trigrams) == 0 else fr_sum/len(target_trigrams)
    return (fr_sum, average)
        
        
