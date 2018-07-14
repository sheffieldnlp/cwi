"""
Frequency features.

This module contains functions to extract features concerning the frequency of a target word or phrase in external lists
of the most frequent words

"""

import re

# import lists of frequent words
wordRE = re.compile(r"[\w]+")

# English: from https://github.com/first20hours/google-10000-english/blob/master/README.md
top_eng = []
with open("data/external/frequent_english_words.txt",  encoding="utf8") as infile:
    for line in infile:
        for word in wordRE.findall(line.lower()):  
            top_eng.append(word)
                    
# Spanish from https://en.wiktionary.org/wiki/User:Matthias_Buchmeier/Spanish_frequency_list-1-5000   
top_span = []
with open("data/external/frequent_spanish_words.txt", encoding = "latin-1") as infile:
    for line in infile:
        for word in wordRE.findall(line.lower()):  
            top_span.append(word)
            
            
#German top 3000 from http://germanvocab.com/
top_ger = []
with open("data/external/frequent_spanish_words.txt") as infile:
    for line in infile:
        for word in wordRE.findall(line.lower()):  
            top_ger.append(word)




def rare_word_count(target_word, language): # Alison

    """
    Count the number of "rare" tokens in the target word

    Args:
        target_word (str): word or phrase candidate

    Returns:
        int - the number of pronunciations of the target word if language = English
        int 0 else
        
        
    Raises:
        Value Error
    
    """
    if language == 'english':
        uncommon = 0  
        for token in target_word.split(' '):
            if token.lower() not in top_eng[0:3000]:                   
                uncommon += 1
     
    elif language == 'spanish':           
        uncommon = 0 
        for token in target_word.split(' '):
            if token.lower() in top_span:
                pass
            elif token+"s" in top_span:
                pass
            elif token+"es" in top_span:
                pass
            elif token[:-1] in top_span:
                pass
            elif token[:-2] in top_span:
                pass
            else:
                uncommon += 1
                
    elif language == 'german':
        uncommon = 0  
        for token in target_word.split(' '):
            if token.lower() not in top_ger:                   
                uncommon += 1
                
    else:
        raise ValueError("Language specified ({}) not supported.".format(language))
         
    return(uncommon)