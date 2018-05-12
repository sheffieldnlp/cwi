
import re
import os
import random
from string import punctuation
from collections import Counter
from collections import defaultdict

import pickle


def strip_punctuation(s):
    return ''.join(c for c in s if c not in punctuation)

s = "model"+".pkl"
       
wordRE = re.compile('[a-zA-Z]+|[^a-zA-Z]')
unigram_counts_english = defaultdict(int)
unigram_counts_spanish = defaultdict(int)
w = ['2007','2008','2009','2010','2011']
total_word_count=0

for i in w:
    with open('news.'+i+'.en.shuffled','r',encoding="utf8") as infile:
        for sent in infile:
            for word in re.findall(r"[\w']+",sent.lower()): #for word in re.findall(r"[^\s]+", sent)
                unigram_counts_english[word]+=1
                total_word_count+=1

for key, value in unigram_counts_english.items():
    unigram_counts_english[key] = value / total_word_count
    


s = "English-model"+".pkl"
pickle.dump(unigram_counts_english,open(s,"wb"))



total_word_count=0
for i in w:
    with open('news.'+i+'.es.shuffled','r',encoding="utf8") as infile:
        for sent in infile:
            for word in re.findall(r"[\w']+",sent.lower()): #for word in re.findall(r"[^\s]+", sent)
                unigram_counts_spanish[word]+=1
                total_word_count=+1



for key, value in unigram_counts_spanish.items():
    unigram_counts_spanish[key] = value / total_word_count


s = "Spanish-model"+".pkl"
pickle.dump(unigram_counts_spanish,open(s,"wb"))

#u2 = defaultdict(int)
#with open('news2008.en.shuffled','r',encoding="utf8") as infile:
    #for sent in infile:
        #for word in wordRE.findall(sent.lower()): #for word in re.findall(r"[^\s]+", sent)
            #u2[strip_punctuation(word)]+=1

#u3 = defaultdict(int)
#with open('news2008.en.shuffled','r',encoding="utf8") as infile:
    #for sent in infile:
        #for word in wordRE.findall(sent.lower()): #for word in re.findall(r"[^\s]+", sent)
            #u3[strip_punctuation(word)]+=1
