""" Noun-phrase features.

"""
def is_noun_phrase(spacy_sentence, target_word):
    """ check if the target phrase is a noun phrase

    Args:
        Spacy tokens for target phrase


    Returns:
        Bool. (1,0)

    """

    t=[]
    for chunk in spacy_sentence.noun_chunks:
        t.append(chunk.text)

    if (target_word in t) and (len(target_word.split(" ")) > 1):
        return 1
    else:
        return 0
