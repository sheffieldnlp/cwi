import pickle
import re
import os
import random
from string import punctuation
from collections import Counter
from collections import defaultdict
import csv

def read_dataset(file_path):
    with open(file_path, encoding="utf8") as file:
        fieldnames = ['hit_id', 'sentence', 'start_offset', 'end_offset', 'target_word', 'native_annots',
                          'nonnative_annots', 'native_complex', 'nonnative_complex', 'gold_label', 'gold_prob']
        reader = csv.DictReader(file, fieldnames=fieldnames, delimiter='\t')
        dataset = [sent for sent in reader]

    return dataset


unigram_counts_english = pickle.load(open('English-model.pkl',"rb"))
total_word_count = len(unigram_counts_english.keys())

wordRE = re.compile('[a-zA-Z]+|[^a-zA-Z]')


with open('news-commentary-v6.en','r',encoding="utf8") as infile:
    for sent in infile:
        for word in re.findall(r"[\w']+",sent.lower()): #for word in re.findall(r"[^\s]+", sent)
            unigram_counts_english[word]+= (1/total_word_count)
            total_word_count+=1




with open('europarl-v6.en','r',encoding="utf8") as infile:
    for sent in infile:
        for word in re.findall(r"[\w']+",sent.lower()): #for word in re.findall(r"[^\s]+", sent)
            unigram_counts_english[word]+= (1/total_word_count)
            total_word_count+=1



#File = ['News_Train.tsv','WikiNews_Train.tsv','Wikipedia_Train.tsv']
#for i in File:
    #with open('english/original/'+i,'r',encoding="utf8") as infile:
        #for sent in infile:
            #for word in re.findall(r"[\w']+",sent.lower()): #for word in re.findall(r"[^\s]+", sent)
                #unigram_counts_english[word]+= (1/total_word_count)
                #total_word_count+=1
            



s = "English-model"+".pkl"
pickle.dump(unigram_counts_english,open(s,"wb"))

#trainset = read_dataset('English_Train.tsv')
#for sent in trainset:
   #unigram_counts_english[sent['target_word'].lower()]+= (1/total_word_count)
   #total_word_count+=1


unigram_counts_spanish = pickle.load(open('Spanish-model.pkl',"rb"))
total_word_count = len(unigram_counts_spanish.keys())



with open('news-commentary-v6.es','r',encoding="utf8") as infile:
    for sent in infile:
        for word in re.findall(r"[\w']+",sent.lower()): #for word in re.findall(r"[^\s]+", sent)
            unigram_counts_spanish[word]+= (1/total_word_count)
            total_word_count+=1




with open('europarl-v6.es','r',encoding="utf8") as infile:
    for sent in infile:
        for word in re.findall(r"[\w']+",sent.lower()): #for word in re.findall(r"[^\s]+", sent)
            unigram_counts_spanish[word]+= (1/total_word_count)
            total_word_count+=1
#print(unigram_counts_spanish)


#trainset = read_dataset('Spanish_Train.tsv')
#for sent in trainset:
   #unigram_counts_spanish[sent['target_word'].lower()]+= (1/total_word_count)
   #total_word_count+=1


#File = ['News_Train.tsv','WikiNews_Train.tsv','Wikipedia_Train.tsv']
#for i in File:
    #with open('spanish/original/'+i,'r',encoding="utf8") as infile:
        #for sent in infile:
            #for word in re.findall(r"[\w']+",sent.lower()): #for word in re.findall(r"[^\s]+", sent)
                #unigram_counts_english[word]+= (1/total_word_count)
                #total_word_count+=1
              
s = "Spanish-model"+".pkl"
pickle.dump(unigram_counts_spanish,open(s,"wb"))
