from nltk.corpus import wordnet
from googletrans import Translator

def hypernym_count(word, language):

    """
    NEEDS DESCRIPTION
    """

    # from the class
    translator = Translator()
    trans_count = 0
    #

    temp= word.split(" ")
    count2 = 0

    if language == "english":
        if len(word.split(" ")) > 1:
            for wd in temp:
                for synset in wordnet.synsets(wd, lang='eng'):
                    count2+= len(synset.hypernyms())
        else:
            for synset in wordnet.synsets(word, lang='eng'):
                count2+= len(synset.hypernyms())
        count2 = count2/len(temp)

    elif language == 'spanish':
        if len(word.split(" ")) > 1:
            for wd in temp:
                for synset in wordnet.synsets(wd, lang='spa'):
                    count2+= len(synset.hypernyms())
        else:
            for synset in wordnet.synsets(word, lang='spa'):
                    count2+= len(synset.hypernyms())
        count2 = count2/len(temp)

    else: #german, french

        # trans_count+=1
        # if  trans_count < 350:
        #     translated_word=translator.translate(word, dest='en')
        # else:
        #     translator=Translator()
        #     translated_word=translator.translate(word, dest='en')
        #     trans_count=0
        #
        # word = translated_word.text
        # temp= word.split(" ")
        # if len(temp) > 1:
        #     for wd in temp:
        #         for synset in wordnet.synsets(wd, lang='eng'):
        #             count2+= len(synset.hypernyms())
        # else:
        #     for synset in wordnet.synsets(word, lang='eng'):
        #         count2+= len(synset.hypernyms())
        # count2 = count2/len(temp)
        
        count = 0




    return (round(count2))
