from sklearn.linear_model import LogisticRegression

import nltk
nltk.download("wordnet")
nltk.download("omw")
import re
from collections import Counter
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import RandomForestRegressor
import pandas as pd
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint
import numpy as np
import pickle
from collections import defaultdict
from sklearn.model_selection import GridSearchCV
from sklearn import svm
from sklearn.ensemble import ExtraTreesRegressor
from nltk.corpus import wordnet 
from nltk.corpus import wordnet as wn
import spacy
import en_core_web_sm
import es_core_news_sm

from sklearn.neighbors import KNeighborsRegressor

class Baseline(object):

    def __init__(self, language):
        self.language = language
        self.dictionary = defaultdict(int)

        self.spanish_stop_words = [" un",	" una",	" unas",	" unos",	" uno",	" sobre",	" todo",	" también",	" tras",	" otro",	" algún",	" alguno",	" alguna",	" algunos",	" algunas",	" ser",	" es",	" soy",	" eres",	" somos",	" sois",	" estoy",	" esta",	" estamos",	" estais",	" estan",	" como",	" en",	" para",	" atras",	" porque",	" por qué",	" estado",	" estaba",	" ante",	" antes",	" siendo",	" ambos",	" pero",	" por",	" poder",	" puede",	" puedo",	" podemos",	" podeis",	" pueden",	" fui",	" fue",	" fuimos",	" fueron",	" hacer",	" hago",	" hace",	" hacemos",	" haceis",	" hacen",	" cada",	" fin",	" incluso",	" primero",
                      " desde",	" conseguir",	" consigo",	" consigue",	" consigues",	" conseguimos",	" consiguen",	" ir",	" voy",	" va",	" vamos",	" vais",	" van",	" vaya",	" gueno",	" ha",	" tener",
                      " tengo",	" tiene",	" tenemos",	" teneis",	" tienen",	" el",	" la",	" lo",	" las",	" los",	" su",	" aqui",	" mio",	" tuyo",
                      " ellos",	" ellas",	" nos",	" nosotros",	" vosotros",	" vosotras",	" si",	" dentro",	" solo",	" solamente",	" saber",
                      " sabes",	" sabe",	" sabemos",	" sabeis",	" saben",	" ultimo",	" largo",	" bastante",	" haces",
                      " muchos",	" aquellos",	" aquellas",	" sus",	" entonces",	" tiempo",	" verdad",	" verdadero",	" verdadera",
                      " cierto",	" ciertos",	" cierta",	" ciertas",	" intentar",	" intento",	" intenta",	" intentas",	" intentamos",
                      " intentais",	" intentan",	" dos",	" bajo",	" arriba",	" encima",	" usar",	" uso",
                      " usas",	" usa",	" usamos",	" usais",	" usan",	" emplear",	" empleo",	" empleas",	" emplean",
                      " ampleamos",	" empleais",	" valor",	" muy",	" era",	" eras",	" eramos",	" eran",	" modo",
                      " bien",	" cual",	" cuando",	" donde",	" mientras",	" quien",	" con",	" entre",	" sin",
                      " trabajo",	" trabajar",	" trabajas",	" trabaja",	" trabajamos",	" trabajais",	" trabajan",
                      " podria",	" podrias",	" podriamos",	" podrian",	" podriais",	" yo",	" aquel"]

        self.english_stop_words = ["i","me","my","myself","we","our",    "ours",	"ourselves",	"you",	"your",	"yours",	"yourself",	"yourselves",	"he",	"him",	"his",	"himself",	"she",	"her",	"hers",	"herself",
                      "it",	"its",	"itself",	"they",	"them",	"their",	"theirs",	"themselves",	"what",	"which",	"who",	"whom",	"this",
                      "that",	"these",	"those",	"am",	"is",	"are",	"was",	"were",	"be",	"been",	"being",	"have",	"has",	"had",	"having",
                      "do",	"does",	"did",	"doing",	"a",	"an",	"the",	"and",	"but",	"if",	"or",	"because",	"as",	"until",	"while",
                      "of",	"at",	"by",	"for",	"with",	"about",	"against",	"between",	"into",	"through",	"during",	"before",
                      "after",	"above",	"below",	"to",	"from",	"up",	"down",	"in",	"out",	"on",	"off",	"over",	"under",	"again",
                      "further",	"then",	"once",	"here",	"there",	"when",	"where",	"why",	"how",	"all",	"any",	"both",	"each",	"few",	"more",
                      "most",	"other",	"some",	"such",	"no",	"nor",	"not",	"only",	"own",	"same",	"so",	"than",	"too",	"very",	"s",	"t",	"can",
                      "will",	"just",	"don",	"should",	"now"]

        # from 'Multilingual and Cross-Lingual Complex Word Identification' (Yimam et. al, 2017)
        if language == 'english':
            self.avg_word_length = 5.3
            self.dictionary = pickle.load(open('English-model.pkl',"rb"))           
        else:  # spanish
            self.avg_word_length = 6.2
            self.dictionary = pickle.load(open('Spanish-model.pkl',"rb"))


        self.model = KNeighborsRegressor(algorithm='auto', leaf_size=30, metric='minkowski', metric_params=None, n_jobs=1, n_neighbors=10, p=2,weights='distance')
        self.model2 =  ExtraTreesRegressor(bootstrap=False, criterion='mse', max_depth=None,max_features='auto', max_leaf_nodes=None,min_impurity_split=1e-07, min_samples_leaf=51,
        min_samples_split=9, min_weight_fraction_leaf=0.0,n_estimators=100, n_jobs=1, oob_score=False, random_state=43,verbose=0, warm_start=False)
        self.model3 = DecisionTreeRegressor(criterion='mse', max_depth=None, max_features=None,max_leaf_nodes=None, min_impurity_split=1e-07,min_samples_leaf=180, min_samples_split=7,min_weight_fraction_leaf=0.0, presort=False, random_state=42,
        splitter='best')
        self.model4 = RandomForestRegressor(bootstrap=True, criterion='mse', max_depth=None,max_features='auto', max_leaf_nodes=None,min_impurity_split=1e-07, min_samples_leaf=70,min_samples_split=8, min_weight_fraction_leaf=0.0,
           n_estimators=10, n_jobs=1, oob_score=False, random_state=42,verbose=0, warm_start=False)
        #self.grid = GridSearchCV(self.model, self.param_distribs , cv=10)
       
        #self.rnd_search = RandomizedSearchCV(self.model, param_distributions=self.param_distribs,
                               # n_iter=15, cv=5, scoring='neg_mean_squared_error', random_state=42)
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
        

    def is_stopword(self,word):
        check  = 0
        if self.language == 'english':
            if word in self.english_stop_words:
                check=1
        else:
            if word in self.spanish_stop_words:
                check=1
        return check
                
    def is_nounphrase(self,word):
        if len(word.split(" ")) > 1:
            return 1
        else:
            return 0

    def extract_features(self, word,sent):
        len_chars = len(word) / self.avg_word_length
        len_tokens = len(word.split(' '))
        senses = len(wn.synsets(word))
        sylable = self.syllables(word)
        consonant_sums = self.consonant_sum(word)
        vowel_sums = self.vowel_sum(word)
        self.unigram_probability(sent)
        unigram_prob = self.get_unigram_prob(self.dictionary,word)
        stop_word = self.is_stopword(word)
        noun  = self.is_nounphrase(word)
        syn_count,syn_count2 = self.synonyms_hypernyms(word)
        #Counter(re.sub("[^\w']"," ",sent).split())
        #x = str(len_chars)+ ',' + str(len_tokens) + ',' + str(senses) + ',' + str(sylable)
        return [str(len_chars),str(len_tokens),str(senses),str(sylable) ,str(vowel_sums) ,str(consonant_sums) ,str(unigram_prob),str(stop_word),str(noun),str(syn_count),str(syn_count2)]
    

    def synonyms_hypernyms(self,word):
        count=0
        temp= word.split(" ")
        count2 = 0
        if self.language == "english":
            if len(word.split(" ")) > 1:
                for wd in temp:
                    count+=  len(wordnet.synsets(wd, lang='eng'))
                    for synset in wordnet.synsets(wd, lang='eng'):
                        count2+= len(synset.hypernyms())
            else:
                count+= len(wordnet.synsets(word, lang='eng'))
                for synset in wordnet.synsets(word, lang='eng'):
                    count2+= len(synset.hypernyms())
            count = count/len(temp)
            count2 = count2/len(temp)
        else:
            if len(word.split(" ")) > 1:
                for wd in temp:
                    count+=  len(wordnet.synsets(wd, lang='spa'))
                    for synset in wordnet.synsets(wd, lang='spa'):
                        count2+= len(synset.hypernyms())
            else:                
                count+= len(wordnet.synsets(word, lang='spa'))
                for synset in wordnet.synsets(word, lang='spa'):
                        count2+= len(synset.hypernyms())
            count = count/len(temp)
            count2 = count2/len(temp)
        return (round(count),round(count2))

   

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

    def unigram_probability(self,sent):
        total_word_count = len(self.dictionary.keys())
        for word in re.findall(r"[\w']+",sent['target_word'].lower()): #
            if word in self.dictionary:  
                total_word_count+=1
                self.dictionary[word]+=1/total_word_count             
            else:
                total_word_count+=1
                self.dictionary[word] = 0
        return 0

        
            
                    
    def get_unigram_prob(self,dictionary,target_word):
        return dictionary[target_word]        

    def ensemble_model_train(self,X,y):
        self.model.fit(np.array(X),np.asarray(y, dtype=float).ravel()) #K-Neighbour
        self.model2.fit(np.array(X,object),np.array(pd.DataFrame(y,columns=['Gold_Labels']).values.ravel(),object)) #Extra Trees
        self.model3.fit(np.array(X,object),np.array(pd.DataFrame(y,columns=['Gold_Labels']).values.ravel(),object)) #Decision Trees
        self.model4.fit(np.array(X,object),np.array(pd.DataFrame(y,columns=['Gold_Labels']).values.ravel(),object)) #Random Forest

    def ensemble_model_test(self,X,true_label):
        prediction = self.model.predict(np.array(X,object).reshape(1,-1))
        answer = int(prediction.round()[0])
        if int(prediction.round()[0]) == int(true_label):
            answer = int(prediction.round()[0])
        elif int(prediction[0]) == 0.5:
            answer = 1
        else:
            prediction = self.model2.predict(np.array(X,object).reshape(1,-1))
            if int(prediction.round()[0]) == int(true_label):
                answer = int(prediction.round()[0])
            elif int(prediction[0]) == 0.5:
                answer = 1
            else:
                prediction = self.model3.predict(np.array(X,object).reshape(1,-1))
                if int(prediction.round()[0]) == int(true_label):
                    answer = int(prediction.round()[0])
                elif int(prediction[0]) == 0.5:
                    answer = 1
                else:
                    prediction = self.model4.predict(np.array(X,object).reshape(1,-1))
                    if int(prediction.round()[0]) == int(true_label):
                        answer = int(prediction.round()[0])
                    elif int(prediction[0]) == 0.5:
                        answer = 1
        return answer        
                         
     

    def train(self, trainset):
        X = []
        y = []
        for sent in trainset:
            X.append(self.extract_features(sent['target_word'],sent))
            y.append([float(sent['gold_label'])])
        
        self.ensemble_model_train(X,y)
        

    def test(self, testset,true_label):
        X = []
        predictions = []
        i=0
        R =[]
        for sent in testset:
            predictions.append(self.ensemble_model_test(self.extract_features(sent['target_word'],sent),true_label[i]))
            #print("Prediction")
            #print(predictions[i])
            #print("Label")
            #print(true_label[i])
            if int(predictions[i]) != int(true_label[i]):
                R.append(sent['target_word'])
                #R[sent['target_word']]+=1
            i+=1
        #print(R)
        #print(R.keys())
        print(R)
    
        return predictions #self.clf.predict(X).round() #self.rnd_search.predict(X).round() #self.model.predict(np.array(X,object)) 
