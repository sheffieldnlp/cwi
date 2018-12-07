# -*- coding: utf-8 -*-
"""
Created on Mon Jul  2 17:54:45 2018

@author: pmfin
"""

from collections import Counter
from nltk import ngrams

def normalizeWord(word):
    result = word.lower()
    return result

# This is probably irrelevant, as we're using count vectorizers:
def countCharacterNgrams(target, N):
    ngram_counts = Counter()
    tokens = target.split(' ')
    num_tokens = len(tokens)
    if num_tokens == 1:
        ngram_counts.update(getCharacterNgrams(tokens))
    else:
        for token in tokens:
            ngram_counts.update(countCharacterNgrams(token, N))

    return ngram_counts

def getNGramsPrefixesSuffixes(token, index, N):
    if index == 0:
        result = ["PREFIX|__|" + token[ index : index + N ], "AFFIX|__|"+token[ index : index + N ]]
    elif index == len(token) - N:
        result = ["SUFFIX|__|" + token[ index : index + N ], "AFFIX|__|"+token[ index : index + N ]]
    else:
        result = ["AFFIX|__|"+ token[ index : index + N ]]
    return result

def getCharacterNgrams(token, N):
    word = normalizeWord(token)
    char_ngrams = Counter()
    for i in range(len(word) - N + 1):
        char_ngrams.update(getNGramsPrefixesSuffixes(word, i, N))
    return char_ngrams

def getBigramsAndTrigrams(token):
    bigrams = getCharacterNgrams(token, 2)
    trigrams = getCharacterNgrams(token, 3)
    result = bigrams + trigrams
    return result

def getAllCharNGrams(token, N=2):
    resultngrams = []
    result = Counter()
    for n in range(N):
        resultngrams += getCharacterNgrams(token, n+1)

    result.update(resultngrams)
    result = dict(result)

    return result

def getPrefixesAndSuffixes(target, min_length, max_length):
    prefix_counts = Counter()
    suffix_counts = Counter()
    tokens = target.split(' ')
    num_tokens = len(tokens)
    if num_tokens == 1:
        word = normalizeWord(tokens[0])

        prefixes = []
        suffixes = []
        for j in range(min_length, max_length + 1):
            prefix = word[:j]
            suffix = word[-j:]
            prefixes.append(prefix)
            suffixes.append(suffix)
        prefix_counts.update(prefixes)
        suffix_counts.update(suffixes)
    else:
        for token in tokens:
            prefixes, suffixes = getPrefixesAndSuffixes(token, min_length, max_length)
            prefix_counts.update(prefixes)
            suffix_counts.update(suffixes)

    return prefix_counts, suffix_counts

def getCorpusCharNgramsFromTarget(target, corpus_NGram_counts):
    # For different values of N:
    ngram_counts = Counter()

    tokens = target.split(' ')
    num_tokens = len(tokens)
    if num_tokens == 1:
        word = normalizeWord(tokens[0])
        all_ngrams = []
        for N, ngrams_ in corpus_NGram_counts.items():
            word_ngrams = getCharacterNgrams(word, N)
            for word_ngram in word_ngrams:
                if word_ngram in ngrams_:
                    all_ngrams.append(word_ngram)
        ngram_counts.update(all_ngrams)
    else:
        for token in tokens:
            ngram_counts.update(getCorpusCharNgramsFromTarget(token, corpus_NGram_counts))


    return ngram_counts

# This is a preprocessing function
def getCorpusNgrams(corpus_text, min_length, max_length):
    words = corpus_text
    corpus_dict = {}
    for n in range(min_length, max_length+1):
        corpus_dict[n] = getCharacterNgrams(words, n)

    return corpus_dict

if __name__ == '__main__':
    word = "Extravagantly"
    print(getAllCharNGrams(word, N=4))

#    targets = ["The cat is eating the table","Huzzah","Happy","What the hell", "Whoops"]
#    corpustext = "Don't read the test code!"
#    corpus = getCorpusNgrams(corpustext, 1, 4)
#
#
#    for target in targets:
#        corpus_ngrams = getCorpusCharNgramsFromTarget(target, corpus)
#        char_trigrams = getCharacterNgrams(target, 3)
#        char_bigrams = getCharacterNgrams(target, 2)
#        prefixes, suffixes = getPrefixesAndSuffixes(target, 1, 4)
#
#
#
#        print("{}\nCorpus Ngrams:\n{}\n\nTrigrams:\n{}\n\nBigrams:\n{}\n\nPrefixes:\n{}\n\nSuffixes:\n{}".format(target, corpus_ngrams, char_trigrams, char_bigrams, prefixes, suffixes))
