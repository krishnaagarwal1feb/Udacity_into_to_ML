#!/usr/bin/python
import numpy
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


### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()
before = time()

# You can continue with the rest of your code here
#########################################################
### your code goes here ###
#features_train = features_train[:len(features_train)/100] 
#labels_train = labels_train[:len(labels_train)/100] 
from sklearn.svm import SVC
clf = SVC(kernel = 'rbf',C=10000)
clf.fit(features_train,labels_train)
mid = time()
pred = clf.predict(features_test)
#########################################################
"""
after = time()
pred_time = after - mid
from sklearn.metrics import accuracy_score
acc = accuracy_score(labels_test, pred)
print acc
print "pred time : "
print  pred_time
print "lets see the answers = "
print clf.predict([features_test[50]])
"""
counter = sum(x for x in pred)
print counter 
