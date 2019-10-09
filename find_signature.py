#!/usr/bin/python

import pickle
import numpy
numpy.random.seed(42)


### The words (features) and authors (labels), already largely processed.
### These files should have been created from the previous (Lesson 10)
### mini-project.
words_file = "../text_learning/your_word_data.pkl" 
authors_file = "../text_learning/your_email_authors.pkl"
word_data = pickle.load( open(words_file, "r"))
authors = pickle.load( open(authors_file, "r") )



### test_size is the percentage of events assigned to the test set (the
### remainder go into training)
### feature matrices changed to dense representations for compatibility with
### classifier functions in versions 0.15.2 and earlier
from sklearn import model_selection
features_train, features_test, labels_train, labels_test = model_selection.train_test_split(word_data, authors, test_size=0.1, random_state=42)

from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer(sublinear_tf=True, max_df=0.5,
                             stop_words='english')
features_train = vectorizer.fit_transform(features_train)
features_test  = vectorizer.transform(features_test).toarray()
li =  vectorizer.get_feature_names()
#print li[33614]

### a classic way to overfit is to use a small number
### of data points and a large number of features;
### train on only 150 events to put ourselves in this regime
features_train = features_train[:150].toarray()
labels_train   = labels_train[:150]



### your code goes here
from sklearn.tree import DecisionTreeClassifier
clf = DecisionTreeClassifier()
clf.fit(features_train, labels_train)

from sklearn.metrics import accuracy_score
acc = accuracy_score(labels_test, clf.predict(features_test))
print acc
numpy_arr = clf.feature_importances_
mx = max(numpy_arr)
print mx
print numpy.where(numpy_arr == mx)
"""as this maximum feature imp terms are causing my decision tree to overfit
i will try to remove these points of max feat imp from my tfidf vectorizer 
Thus this point with feat imp > 0.2 can sevrve as an outlier """
print numpy.where(numpy_arr > 0.2)
print li[21323]
"""
we have removed all the outliers here - "sshacklensf", "cgermannsf"
and now the training data has started to overfit with 
accuracy of 81.68%"""
