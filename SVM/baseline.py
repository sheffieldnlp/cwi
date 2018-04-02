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
from sklearn import svm, datasets
from sklearn.model_selection import GridSearchCV



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
        #parameters = {'kernel':('linear', 'rbf'), 'C':[1, 10],'epsilon':[0.1,0.5,1]}
        parameters = [
              {'C': [1, 10, 100], 'kernel': ['linear']},
              {'C': [1, 10, 100], 'gamma': [0.1,0.001, 0.0001], 'kernel': ['rbf','sigmoid','poly']},
             ]
        self.model = svm.SVC()
        self.clf = GridSearchCV(self.model, parameters)
        #self.model= RandomForestRegressor(n_estimators=1000, criterion='mse', min_samples_split=100, min_samples_leaf=100, max_features='auto')

    def extract_features(self, word,sent):
        len_chars = len(word) / self.avg_word_length
        len_tokens = len(word.split(' '))
        senses = len(wn.synsets(word))
        sylable = self.syllables(word)
        #Counter(re.sub("[^\w']"," ",sent).split())
        return [len_chars, len_tokens,senses,sylable]

    def bigram_LM(self,sentence_x, smoothing=0.0):
        unique_words = len(unigram_counts.keys()) + 2 # For the None paddings
        x_bigrams = nltk.bigrams(re.sub("[^\w']"," ",sentence_x).split(), pad_left=True, pad_right=True)
        prob_x = 1.0
        for bg in x_bigrams:
            if bg[0] == None:
                prob_bg = (bigram_counts[bg]+smoothing)/(len(brown.sents())+smoothing*unique_words)
            else:
                prob_bg = (bigram_counts[bg]+smoothing)/(unigram_counts[bg[0]]+smoothing*unique_words)
            prob_x = prob_x *prob_bg
            #print(str(bg)+":"+str(prob_bg))
        return prob_x

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
                #bigrams.extend(nltk.bigrams(re.sub("[^\w']"," ",sent['sentence']).split(), pad_left=True, pad_right=True))
                #prob_x = self.bigram_LM(sent['sentence'],1)
            #print(sent['sentence'])
            #f.write('\n' + sent['target_word'])
            #print(sent)
            X.append(self.extract_features(sent['target_word'],sent['sentence']))
            y.append(sent['gold_label'])
        #f.close()
        #print(X)
        #bigram_counts = Counter(bigrams)
        #print(bigrams)
        #df1=pd.DataFrame(X,columns=['len_chars','len_tokens','senses','syllabules'])
        #df1.index.name = 'ID'
        #df2 = pd.DataFrame(y,columns=['Label'])
        #df2.index.name = 'ID'
        #self.model.fit(X, y)
        self.clf.fit(X, y)
        print(self.clf.cv_results_.keys())
        print(self.clf.cv_results_.values())
        print('Best Estimator')
        print(self.clf.best_estimator_)
        print('Best Parameter')
        print(self.clf.best_params_)
        #sorted(self.clf.cv_results_.keys())
        #print(self.model.fit(df1, df2))
        

    def test(self, testset):
        X = []
        for sent in testset:
            X.append(self.extract_features(sent['target_word'],sent['sentence']))
        #df1=pd.DataFrame(X,columns=['len_chars','len_tokens','senses','syllabules'])
    
        return self.model.predict(X).round() #self.model.predict(df1)
