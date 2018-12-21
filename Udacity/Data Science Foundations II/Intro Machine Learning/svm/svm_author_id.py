#!/usr/bin/python

"""
    This is the code to accompany the Lesson 2 (SVM) mini-project.

    Use a SVM to identify emails from the Enron corpus by their authors:
    Sara has label 0
    Chris has label 1
"""

import sys
from time import time
sys.path.append("../tools/")
from email_preprocess import preprocess
import numpy as np


### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()




#########################################################
### your code goes here ###

from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

#features_train = features_train[:len(features_train)/100] ##diminuir a amostra para 1%
#labels_train = labels_train[:len(labels_train)/100] ##diminuir a amostra para 1%
clf = SVC(C=10000.0, kernel='rbf')
t0 = time()
clf.fit(features_train, labels_train)
print "training_time:", round(time()-t0, 3), "s"
t0 = time()
pred = clf.predict(features_test)
print "prediction time:", round(time()-t0, 3), "s"
acc = accuracy_score(pred, labels_test)
print acc

#confusion matrix para saber quanto o programa previu que emails eram do Chris
cm = confusion_matrix(labels_test, pred)
print cm


####
#Previsao dos registros 10, 26 e 50
#answer10 = pred[10]
#answer26 = pred[26]
#answer50 = pred[50]
#print "Resposta 10:", answer10
#print "Resposta 26:", answer26
#print "Resposta 50:", answer50
####

#########################################################
