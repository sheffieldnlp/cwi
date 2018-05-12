import pickle
import re
import os
import random
from string import punctuation
from collections import Counter
from collections import defaultdict



unigram_counts_english = pickle.load(open('English-model.pkl',"rb"))
total_word_count = len(unigram_counts_english.keys())

wordRE = re.compile('[a-zA-Z]+|[^a-zA-Z]')

#unigram_counts_english = defaultdict(int)
#unigram_counts_spanish = defaultdict(int)



with open('news.'+i+'.en.shuffled','r',encoding="utf8") as infile:
    for sent in infile:
        for word in re.findall(r"[\w']+".lower()): #for word in re.findall(r"[^\s]+", sent)
            unigram_counts_english[word]+=1
            total_word_count+=1
