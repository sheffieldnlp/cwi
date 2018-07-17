spanish_stop_words = []
with open('data/external/Spanish_StopWords.txt','r',encoding="utf-8") as infile:
    for line in infile:
       spanish_stop_words.append(line.replace("\n", ""))

english_stop_words = []
with open('data/external/English_StopWords.txt','r',encoding="utf-8") as infile:
    for line in infile:
        english_stop_words.append(line.replace("\n", ""))

german_stop_words = []
with open('data/external/german_stopwords_full.txt','r',encoding="utf-8") as infile:
    for line in infile:
        german_stop_words.append(line.replace("\n", ""))
german_stop_words.pop(0)


def is_stop(target_word,language):
    v=0
    total_len =  len(target_word.split(" "))
    if total_len > 1:
        for word in target_word.split(" "):
            if language == 'english':
                if word in english_stop_words:
                    v+=1
            elif language == 'spanish':
                if word in spanish_stop_words:
                    v+=1
            else:
                if word in german_stop_words:
                    v+=1
        if (v/total_len) > 0.2:
            v=1
        else:
            v=0
    else:
        if language == 'english':
            if target_word in english_stop_words:
                v=1
            elif  language == 'spanish':
                if target_word in spanish_stop_words:
                    v=1
            else:
                if target_word in german_stop_words:
                    v=1       
    return v
