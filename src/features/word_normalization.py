# -*- coding: utf-8 -*-
"""
Created on Tue Jul  3 19:56:01 2018

@author: pmfin
"""

import string

# Since a bunch of my feature extractors needed normalized tokens, thought it made
# sense to create some utility functions.

def strip_punctuation(sent):
    translator = str.maketrans('', '', string.punctuation)
    result = sent.translate(translator)
    return result

def tokenize(sent):
    # TODO: Probably want a better tokenizer here
    tokens = sent.split(' ')
    return tokens

def normalize(word):
    # TODO: Again, this is really basic punctuation removal and decapitalization.
    # Maybe we should have something that detects if the word is a number and then
    # replaces it with "__NUM__" or something?
    normalized = word.lower()
    return normalized

def normalize_sent(sent):
    stripped = strip_punctuation(sent)
    tokens = tokenize(stripped)
    result = [normalize(word) for word in tokens]
    return result