# -*- coding: utf-8 -*-
"""
Created on Tue Jan 22 11:58:27 2019

@author: baruch
"""

#Sample
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier

model = AdaBoostClassifier(base_estimator = DecisionTreeClassifier(max_depth=2), n_estimators=4)