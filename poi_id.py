"""I recommend not using dataframes here 
as they were not much compaitaible the time udacity was working on this project
better do all the calculations and analysis using 
dict of dicts only - the given format """
#%%imports
import matplotlib.pyplot as plt
import pandas as pd
import sys
import pickle
import numpy as np
sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
#%% data and feature imports 
### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".    
email_features_list=['from_messages','from_poi_to_this_person',
    'from_this_person_to_poi','shared_receipt_with_poi','to_messages',]

financial_features_list=['bonus','deferral_payments',
    'deferred_income','director_fees',
    'exercised_stock_options','expenses',
    'loan_advances','long_term_incentive',
    'other','restricted_stock',
    'restricted_stock_deferred','salary',
    'total_payments','total_stock_value',]
features_list = ['poi']+email_features_list + financial_features_list 

### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)
#%% outlier detection
def PlotOutlier(data_dict, ax, ay):
    data = featureFormat(data_dict, [ax,ay,'poi'])
    for point in data:
        x = point[0]
        y = point[1]
        poi=point[2]
        if poi:
            color='blue'
        else:
            color='green'

        plt.scatter( x, y, color=color )
    plt.xlabel(ax)
    plt.ylabel(ay)
    plt.show()

PlotOutlier(data_dict, 'from_poi_to_this_person','from_this_person_to_poi')
PlotOutlier(data_dict, 'total_payments', 'total_stock_value')
PlotOutlier(data_dict, 'from_messages','to_messages')
PlotOutlier(data_dict, 'salary','bonus')


### Task 2: Remove outliers
data_dict.pop('TOTAL', 0)
data_dict.pop('BAXTER JOHN C',0)

#%% Fill in the NaN payment and stock values with zero 

### Store to my_dataset for easy export below.
my_dataset = {k: {k2: 0 if v2 == 'NaN' else v2 for k2, v2 in v.items()} \
                    for k, v in data_dict.items()}
#we filled nan values with zeros in the data dictionary itself 
df = pd.DataFrame.from_dict(my_dataset, orient='index')

#%% Task 3: Create new feature(s)
"""
since i awnt the fraction of to_messages/total_msgs
and from_messages/total_msgs
thus we will creat e a function which will compute fraction between 
two features in the data dictionary 
"""
def Fraction(poi_messages, all_messages):
    #to and from / all messages would be our 2 extra features 
    fraction = 0
    if poi_messages == 0 or all_messages == 0:
        return 0
    fraction = float(poi_messages)/float(all_messages)
    return fraction

#now to create all messages and from and to messsages for every person 
submit_dict = {}
for name in my_dataset:
    data_point = my_dataset[name]
    from_poi = data_point["from_poi_to_this_person"]
    to_poi = data_point["from_this_person_to_poi"]
    
    fraction_from_poi= Fraction(from_poi,data_point["from_messages"])
    fraction_to_poi = Fraction( to_poi, data_point["to_messages"] )
    
    data_point["fraction_from_poi"] = fraction_from_poi
    submit_dict[name]={"from_poi_to_this_person":fraction_from_poi,
                       "from_this_person_to_poi":fraction_to_poi}
    data_point["fraction_to_poi"] = fraction_to_poi

# now as our feature construction is finished we will move on to selecting the
#best features needed
my_feature_list=features_list+['fraction_from_poi','fraction_to_poi']
#%% selecting best features 
from sklearn.feature_selection import SelectKBest, f_classif

def getkbest(features_list, k):
    data=featureFormat(my_dataset, features_list)
    labels, features = targetFeatureSplit(data)
    selection=SelectKBest(k=k).fit(features,labels)
    scores=selection.scores_
    unsorted_pairs = zip(features_list[1:], scores)
    sorted_pairs=list(reversed(sorted(unsorted_pairs, key=lambda x: x[1])))
    selection_best = dict(sorted_pairs[:k])
    return selection_best

num=12 
best_features = getkbest( my_feature_list, num)
print ('Selected features and their scores: ', best_features)
my_feature_list = ['poi'] + best_features.keys()
print ("{0} selected features: {1}\n".format(len(my_feature_list) - 1, my_feature_list[1:]))


"""after selecting the best 12 features i am going to load my data in
a specific format which is req to train and test my model"""
#%% Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, my_feature_list, sort_keys = True)
labels, features = targetFeatureSplit(data)

from sklearn import preprocessing
scaler=preprocessing.MinMaxScaler()
features=scaler.fit_transform(features)
"""
the income and other values were at diff scale so that is why we are usig the
min max scalar to normalize the values
we are not transforming the result value as it was already in 0s and 1s
"""
### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

#%% Provided to give you a starting point. Try a variety of classifiers.
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline
#from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier

#%%finished importing all the modules
clf_dectree = make_pipeline(StandardScaler(), PCA(),
                      DecisionTreeClassifier(max_depth=6, min_samples_leaf=2, min_samples_split=7,random_state=42))

clf_gauss =  make_pipeline(StandardScaler(),PCA(), GaussianNB())
clf_svc = make_pipeline(StandardScaler(), PCA() , SVC())
#clf_mulnb = make_pipeline(StandardScaler(), PCA() , MultinomialNB())
clf_logreg = make_pipeline(StandardScaler(), PCA() , LogisticRegression())
clf_kmeans = make_pipeline(StandardScaler(), PCA(), KMeans() )
clf_rf = make_pipeline(StandardScaler(), PCA(), RandomForestClassifier())

#%% accuracy precision and recall scores

from sklearn.metrics import precision_score, recall_score,confusion_matrix,classification_report
from sklearn.model_selection import train_test_split
def compute(clf,features, labels,num=100):
    #print clf
    acc = []
    pre = []
    rec = []
    for trial in range(num):
        features_train, features_test, labels_train, labels_test = \
        train_test_split(features, labels, test_size=0.3, random_state=42)
        clf = clf.fit(features_train, labels_train)
        pred = clf.predict(features_test)
        acc.append(clf.score(features_test, labels_test))
        pre.append(precision_score(labels_test, pred,average='micro' ))
        rec.append(recall_score(labels_test, pred, average='micro'))
    print 'accuracy score :',np.mean(acc)
    print 'precision:  ' , np.mean(pre)
    print 'recall: ' , np.mean(rec)
    clf_rep = classification_report(labels_test, pred) 
    cf = confusion_matrix(labels_test, pred)
    print cf
    print clf_rep
"""
    pre_mean = np.mean(pre)
    rec_mean = np.mean(rec)
    acc_mean = np.mean(acc)
    print "precision : " + pre_mean
    print "accuracy : " + acc_mean
    print "recall : " + rec_mean
    cf = confusion_matrix(labels_test, pred)
    print "confusion matrix : " + cf
    classif_rep = classification_report(labels_test,pred)
    print "classification report : " + classif_rep
    return pre_mean, rec_mean, cf, classif_rep
"""
#%% callig the compute to check which classifier suits my results better 
"""
print 'KMeans: ',compute(clf_kmeans, features, labels)
print 'svc :' , compute(clf_svc,features,labels)
print 'random forest : ',  compute(clf_rf,features,labels)
print 'random forest : ', compute(clf_rf,features,labels)
print 'logistic regression :', compute(clf_logreg,features,labels)
print 'gaussian nb :', compute(clf_gauss,features,labels)
print 'decsion tree :', compute(clf_dectree,features,labels)
"""
#BASED ON ABOVE RESULTS WE CHOOSE SVC or random forest or log reg , or gaussNB
#running tester.py for all these , we can evaluate our models and found out that GaussianNB works for us
#%%fine tuning parameters of each classifier 
### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validationStratifiedShuffleSplit.html

#SUPPOSE WE GOT GAUSSIAN NB AS SUITABLE MODEL
#select k best removed from pipeline --
from sklearn.model_selection import GridSearchCV, StratifiedShuffleSplit
k_range = [6, 8, 10, 12]
PCA_range = [2, 3, 4, 5, 6]
pipeline = make_pipeline( SelectKBest(k=10), StandardScaler(), PCA(), GaussianNB())

parameters_gnb = {
        #'k__k' : k_range,
        'pca__n_components' : PCA_range}

cv = StratifiedShuffleSplit(n_splits = 20,random_state = 42)
gs_gnb = GridSearchCV(pipeline, parameters_gnb, n_jobs = -1, cv=cv, scoring="f1")

gs_gnb.fit(features, labels)
clf = gs_gnb.best_estimator_
#%%..
print 'best estimator: ',gs_gnb.best_estimator_
print 'best parameter: ',gs_gnb.best_params_
#%% Example starting point. Try investigating other evaluation techniques!

### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.
from tester import dump_classifier_and_data
dump_classifier_and_data(clf, my_dataset, features_list)
