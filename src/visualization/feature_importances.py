# -*- coding: utf-8 -*-
"""
Created on Fri Nov  2 13:53:23 2018

@author: pmfin
"""

import pickle
from operator import itemgetter

def save_model_importances(model, feature_pipeline, dest):
    
    coefs = model.coef_ 
    feature_names = feature_pipeline.named_steps['join pipelines'].get_feature_names()

    feat_coef_dict = {}
    for coef, feature in zip(coefs[0,:], feature_names):
        feat_coef_dict[feature] = coef
        
    with open(dest, 'wb') as f:
        pickle.dump(feat_coef_dict, f, pickle.HIGHEST_PROTOCOL)
        
    return None


def load_model_importances(dest):
    
    with open(dest, 'rb') as f:
        result = pickle.load(f)
        
    return result

def highest_x_importances(feat_coef_dict, top_x):
    sorted_dict = sorted(feat_coef_dict.items(), key=itemgetter(1), reverse=True)
    ordered_high = sorted_dict[:top_x]
    low = sorted_dict[-top_x:]
    ordered_low = list(reversed(low))
    return ordered_high, ordered_low

def print_x_importances(dest, top_x):
    importances = load_model_importances(dest)
    high, low = highest_x_importances(importances, top_x)
    high_str = get_formatted_str(high)
    low_str = get_formatted_str(low)
    printable_list = "Highest Coefficients:\n{}\nLowest Coefficients:\n{}".format(high_str, low_str)
    print(printable_list)
    
def get_formatted_str(li):
    result = ""
    rank = 1
    for item in li:
        name = str(item[0])
        coef = str(item[1])
        append = "{:>4}: {:>40}\t{:>12}\n".format(rank, name, coef)
        result += append
        rank += 1
    return result
        

if __name__ == "__main__":
    importances_dest = "data/interim/importances.pkl"
#    importances = load_model_importances(importances_dest)
#    high, low = highest_x_importances(importances, 10)
#    print(high)
    
    print_x_importances(importances_dest, 25)
    