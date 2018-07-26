"""

This module contains file input/output operations.

"""
import csv

def read_file(file_name):
    reader = csv.reader(open(file_name, 'r'))
    u_prob = {}
    for (word,prob) in reader:
        u_prob[word] = float(prob)
    return u_prob