from sklearn.linear_model import LogisticRegression

import nltk
nltk.download("wordnet")
import re
from collections import Counter
from nltk.corpus import wordnet as wn
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import RandomForestRegressor
import pandas as pd
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint
import numpy as np



class Baseline(object):

    def __init__(self, language):
        self.language = language
        # from 'Multilingual and Cross-Lingual Complex Word Identification' (Yimam et. al, 2017)
        if language == 'english':
            self.avg_word_length = 5.3
        else:  # spanish
            self.avg_word_length = 6.2

        #self.model = LogisticRegression()
        #self.model = DecisionTreeRegressor()
        # Decision Tree Hyperperemeters Range for optimization
        self.param_distribs = {
            'min_samples_split': randint(low=2, high=10),
            'min_samples_leaf': randint(low=50, high=200),
            }
        self.model= RandomForestRegressor(random_state=42)
        #self.model = DecisionTreeRegressor(random_state=42)
        self.rnd_search = RandomizedSearchCV(self.model, param_distributions=self.param_distribs,
                                n_iter=15, cv=5, scoring='neg_mean_squared_error', random_state=42)
        #print(self.rnd_search)


    def consonant_sum(self,word):
        consonants = list("bcdfghjklmnpqrstvexz")
        number_of_consonants = sum(word.count(c) for c in consonants)
        return number_of_consonants

    def vowel_sum(self,word):
        #https://stackoverflow.com/questions/7736211/python-counting-the-amount-of-vowels-or-consonants-in-a-user-input-word
        vowels = list("aeiouy")
        number_of_vowels = sum(word.count(c) for c in vowels)
        return number_of_vowels
        
        


    def extract_features(self, word,sent):
        len_chars = len(word) / self.avg_word_length
        len_tokens = len(word.split(' '))
        senses = len(wn.synsets(word))
        sylable = self.syllables(word)
        consonant_sums = self.consonant_sum(word)
        vowel_sums = self.vowel_sum(word)
        #Counter(re.sub("[^\w']"," ",sent).split())
        #x = str(len_chars)+ ',' + str(len_tokens) + ',' + str(senses) + ',' + str(sylable)
        return [str(len_chars),str(len_tokens),str(senses),str(sylable),str(consonant_sums),str(vowel_sums)]
    



    def syllables(self,word):
        #referred from stackoverflow.com/questions/14541303/count-the-number-of-syllables-in-a-word
        count = 0
        vowels = 'aeiouy'
        word = word.lower()
        if word[0] in vowels:
            count +=1
        for index in range(1,len(word)):
            if word[index] in vowels and word[index-1] not in vowels:
                count +=1
        if word.endswith('e'):
            count -= 1
        if word.endswith('le'):
            count+=1
        if count == 0:
            count +=1
        return count
        
        
    def train(self, trainset):
        X = []
        y = []
        sentences = []
        bigrams = []
        #f = open('helloworld.txt','w',encoding="utf8")
        for sent in trainset:
            X.append(self.extract_features(sent['target_word'],sent['sentence']))
            y.append([float(sent['gold_label'])])
     
        #self.model.fit(X, y)
        #np.asarray(y
        #print(X)
        #print(y)
        #print(np.array(X))
        #print(y)
        print(len(X))
        print(len(y))
        #print(np.array(y,object).shape)
        self.rnd_search.fit(np.array(X,object),np.array(pd.DataFrame(y,columns=['Gold_Labels']).values.ravel(),object))
        print(self.rnd_search.cv_results_)
        print(self.rnd_search.best_estimator_)
        print(self.rnd_search.best_params_)
        #print(self.model.fit(df1, df2))
        

    def test(self, testset):
        X = []
        for sent in testset:
            X.append(self.extract_features(sent['target_word'],sent['sentence']))
        #df1=pd.DataFrame(X,columns=['len_chars','len_tokens','senses','syllabules'])
    
        return self.rnd_search.predict(X).round() #self.model.predict(df1)
