""" Affix-based features.

This module contains functions to detect whether a target word or phrase contains Latin or Greek affixes.

"""
from pathlib import Path
import os

def get_affixes():
    affixes = []
    base = Path(os.path.abspath('.'))
    path_to_affixes = Path('data/external/greek_and_latin_affixes.txt')
    final_path = base.joinpath(path_to_affixes)

    with open(final_path, encoding='latin-1') as f:
        for line in f:
            affixes.append(line.replace("\n", ""))
    return affixes
    

def greek_or_latin(target_word, affixes=None):
    """Computes whether the target word contains Greek or Latin affixes.

    Args:
        target_word (str): word or phrase candidate.

    Returns:
        int. 1 if a Greek or Latin affix is contained in the target word, 0 if not.
    """
    if affixes is None:
        affixes = get_affixes()
    return int(any(affix in target_word for affix in affixes))

if __name__ == '__main__':
    affixes = get_affixes()
