# -*- coding: utf-8 -*-
"""
Created on Fri Nov  2 19:26:45 2018

@author: NAME
"""
from sklearn.pipeline import Pipeline

class NamedPipeline(Pipeline):
    def get_feature_names(self):
        # gets the last element of the list
        return self.steps[-1][1].get_feature_names()