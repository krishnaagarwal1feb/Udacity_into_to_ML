#!/usr/bin/python

import matplotlib.pyplot as plt
from prep_terrain_data import makeTerrainData
from class_vis import prettyPicture

features_train, labels_train, features_test, labels_test = makeTerrainData()


### the training data (features_train, labels_train) have both "fast" and "slow"
### points mixed together--separate them so we can give them different colors
### in the scatterplot and identify them visually
grade_fast = [features_train[ii][0] for ii in range(0, len(features_train)) if labels_train[ii]==0]
bumpy_fast = [features_train[ii][1] for ii in range(0, len(features_train)) if labels_train[ii]==0]
grade_slow = [features_train[ii][0] for ii in range(0, len(features_train)) if labels_train[ii]==1]
bumpy_slow = [features_train[ii][1] for ii in range(0, len(features_train)) if labels_train[ii]==1]


#### initial visualization
plt.xlim(0.0, 1.0)
plt.ylim(0.0, 1.0)
plt.scatter(bumpy_fast, grade_fast, color = "b", label="fast")
plt.scatter(grade_slow, bumpy_slow, color = "r", label="slow")
plt.legend()
plt.xlabel("bumpiness")
plt.ylabel("grade")
plt.show()
################################################################################


### your code here!  name your classifier object clf if you want the 
### visualization code (prettyPicture) to show you the decision boundary

from sklearn.metrics import accuracy_score
#%% ada boost algo 
from sklearn.ensemble import AdaBoostClassifier
clf = AdaBoostClassifier(n_estimators=100, random_state=0)
clf.fit(features_train, labels_train)  
y_pred = clf.predict(features_test)
adaboost_accuracy = accuracy_score(labels_test, y_pred)

#%% random forest 
from sklearn.ensemble import RandomForestClassifier
clfrand = RandomForestClassifier(n_estimators=100, random_state=0)
clfrand.fit(features_train, labels_train)  
rf_pred = clfrand.predict(features_test)
random_forest_acc = accuracy_score(labels_test,rf_pred)
#%% knn algo 
from sklearn.neighbors import KNeighborsClassifier
clf_knn = KNeighborsClassifier(n_neighbors=5)
clf_knn.fit(features_train, labels_train)  
knn_pred = clf_knn.predict(features_test)
knn_acc = accuracy_score(labels_test, knn_pred)  
"""here tunable parameters will have a large impact on model selection
and performance """

#%%
try:
    prettyPicture(clf, features_test, labels_test)
except NameError:
    pass
