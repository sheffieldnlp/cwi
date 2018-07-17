""" Affix-based features.

This module contains functions to detect whether a target word or phrase contains Latin or Greek affixes.

"""
affixes = []
<<<<<<< HEAD
with open('data/external/greek_and_latin_affixes.txt', encoding="latin-1") as f:
=======
with open('data/external/greek_and_latin_affixes.txt', encoding='latin-1') as f:
>>>>>>> upstream/master
        for line in f:
            affixes.append(line.replace("\n", ""))

def greek_or_latin(target_word):
    """Computes whether the target word contains Greek or Latin affixes.

    Args:
        target_word (str): word or phrase candidate.

    Returns:
        int. 1 if a Greek or Latin affix is contained in the target word, 0 if not.
    """
    return int(any(affix in target_word for affix in affixes))
