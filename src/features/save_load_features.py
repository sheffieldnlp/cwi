# -*- coding: utf-8 -*-
"""
Created on Wed Oct 10 18:45:33 2018

@author: pmfin
"""
from pathlib import Path
import pickle

def save_features(prefix, language, train_or_test_set, features):
    filepath = get_filepath(prefix, language, train_or_test_set)
    
    with open(filepath, 'wb') as file:
        pickle.dump(features, file, pickle.HIGHEST_PROTOCOL)
        
    return None

def load_features(prefix, language, train_or_test_set):
    filepath = get_filepath(prefix, language, train_or_test_set)
    
    X = None
    if filepath.exists():
        with open(filepath, 'rb') as file:
            X = pickle.load(file)

    return X

def get_filepath(prefix, language, train_or_test_set):
    base = "/data/processed/"
    
    # TODO: Maybe should do some hash thing (str(hash(len(train_or_test_set))))
    unique = str(len(train_or_test_set))
    filetype = ".pkl"
    filepath = Path(base + prefix + language + unique + filetype)
    return filepath

if __name__ == "__main__":
    print("Testing...")
    prefix = "Test"
    language = "English"
    train_set = [{"Hi":1, "Yes":2},{"Hi":3, "Yes":4}]
    features = [1,2]
    save_features(prefix, language, train_set, features)
    X = load_features(prefix, language)
    print(X)
    