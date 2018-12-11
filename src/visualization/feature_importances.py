# -*- coding: utf-8 -*-
"""
Created on Fri Nov  2 13:53:23 2018

@author: pmfin
"""

import pickle
from operator import itemgetter

def save_model_importances(model, feature_pipeline, dest):
    
    coefs = model.coef_
    class_map = model.classes_
    
    feature_names = feature_pipeline.named_steps['join pipelines'].get_feature_names()

    feat_coef_dict = {}
    for coef, feature in zip(coefs[0,:], feature_names):
        feat_coef_dict[feature] = coef
        
    with open(dest, 'wb') as f:
        pickle.dump((feat_coef_dict, class_map), f, pickle.HIGHEST_PROTOCOL)
        
    return None


def load_model_importances(dest):
    
    with open(dest, 'rb') as f:
        important_tup = pickle.load(f)
        importances = important_tup[0]
        class_map = important_tup[1]
        
    return importances, class_map

def highest_x_importances(feat_coef_dict, top_x):
    sorted_dict = sorted(feat_coef_dict.items(), key=itemgetter(1), reverse=True)
    ordered_high = sorted_dict[:top_x]
    low = sorted_dict[-top_x:]
    ordered_low = list(reversed(low))
    return ordered_high, ordered_low

def print_x_importances(dest, top_x):
    importances, class_map = load_model_importances(dest)
    high, low = highest_x_importances(importances, top_x)
    high_str = get_formatted_str(high)
    low_str = get_formatted_str(low)
    printable_list = "Highest Coefficients:\n{}\nLowest Coefficients:\n{}".format(high_str, low_str)
    class_map_str = ""
    first_coef = False
    for gold_label in class_map:
        if first_coef == False:
            coef_text = "(Positive Coefficients)"
            first_coef = True
        else:
            coef_text = "(Negative Coefficients)"
            
        if gold_label == 0:
            annotation = "Non-complex\t" + coef_text
        else:
            annotation = "Complex\t" + coef_text
        class_map_str += "\n{} - {}".format(str(gold_label), annotation)
    
    print("Class Map: {}\n".format(class_map_str))
    print(printable_list)
    
def get_formatted_str(li):
    result = ""
    rank = 1
    for item in li:
        name = str(item[0])
        source = name.split(sep="__")[0]
        featandspecific = name.split(sep="|__|")
        if source == "bow":
            featname = "Bag"
            specific = '\"{}\"'.format(name.split(sep="__")[1])
        else:
            featname = featandspecific[1]
            specific = '\"{}\"'.format(featandspecific[2])
        
        coef = str(item[1])
        append = "{:>4}: {:>11}{:>20}{:>24}\t{:>12}\n".format(rank, source, featname, specific, coef)
        result += append
        rank += 1
    return result
        

if __name__ == "__main__":
    importances_dest = "data/interim/importances.pkl"
#    importances = load_model_importances(importances_dest)
#    high, low = highest_x_importances(importances, 10)
#    print(high)
    
    print_x_importances(importances_dest, 25)
    