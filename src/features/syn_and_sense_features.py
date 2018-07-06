""" Synonym and sense-based features

This module contains functions to extract features relating to the senses and synonyms of a target word or phrase.

"""
from nltk.corpus import wordnet as wn

def no_synonyms(target_word, language):
    """Computes the number of synonyms for the target word.

    Args:
        target_word (str): word or phrase candidate.
        language (str): the language of the word

    Returns:
        int. The number of synonyms.
    """
    synonyms = set()
    for synset in wn.synsets(target_word, lang = language[:3]):
            for lemma in synset.lemma_names(language[:3]):
                synonyms.add(lemma)
    return(len(synonyms))
    
def no_senses(target_word, language):
    """Computes the number of senses for the target word.

    Args:
        target_word (str): word or phrase candidate.
        language (str): the language of the word

    Returns:
        int. The number of senses.
    """
    return(len(wn.synsets(target_word, lang = language[:3])))
        
        
