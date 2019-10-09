#!/usr/bin/python


"""
    Starter code for the evaluation mini-project.
    Start by copying your trained/tested POI identifier from
    that which you built in the validation mini-project.

    This is the second step toward building your POI identifier!

    Start by loading/formatting the data...
"""

import pickle
import sys
sys.path.append("../tools/")
from feature_format import featureFormat, targetFeatureSplit

data_dict = pickle.load(open("../final_project/final_project_dataset.pkl", "r") )

### add more features to features_list!
features_list = ["poi", "salary"]

data = featureFormat(data_dict, features_list)
labels, features = targetFeatureSplit(data)



### your code goes here 


from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score 

clf = DecisionTreeClassifier()
"""clf.fit(features,labels)
acc = accuracy_score(labels, clf.predict(features))
print acc
"""
from sklearn.model_selection import train_test_split
features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size = 0.3,random_state = 42)

clf.fit(features_train,labels_train)
y_pred = clf.predict(features_test)
print "accuracy score when we split the data in train-test :"
print accuracy_score(labels_test, y_pred )

print "[Q1] How many POIs are predicted for the test set for your POI identifier?"
print "sol-1 :" , sum(y_pred)


print "we can have a look at the confusion matrix :"
from sklearn.metrics import confusion_matrix,precision_score,recall_score
cf = confusion_matrix(labels_test, y_pred)
print cf
length_test = len(labels_test)
print "[Q2] How many people total are in your test set?"
print length_test
print "[Q3] If your identifier predicted 0. (not POI) for everyone in the test set, what would its accuracy be?"
print "[A3]", labels_test.count(0) / float(length_test)

print "precision score : " , precision_score(labels_test,y_pred)
print "recall score : " , recall_score(labels_test, y_pred)