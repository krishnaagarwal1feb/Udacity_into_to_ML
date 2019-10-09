#!/usr/bin/python 

""" 
    Skeleton code for k-means clustering mini-project.
"""



import pandas as pd
import pickle
import numpy
import matplotlib.pyplot as plt
import sys
sys.path.append("../tools/")
from feature_format import featureFormat, targetFeatureSplit




def Draw(pred, features, poi, mark_poi=False, name="image.png", f1_name="feature 1", f2_name="feature 2"):
    """ some plotting code designed to help you visualize your clusters """

    ### plot each cluster with a different color--add more colors for
    ### drawing more than five clusters
    colors = ["b", "c", "k", "m", "g"]
    for ii, pp in enumerate(pred):
        plt.scatter(features[ii][0], features[ii][1], color = colors[pred[ii]])

    ### if you like, place red stars over points that are POIs (just for funsies)
    if mark_poi:
        for ii, pp in enumerate(pred):
            if poi[ii]:
                plt.scatter(features[ii][0], features[ii][1], color="r", marker="*")
    plt.xlabel(f1_name)
    plt.ylabel(f2_name)
    plt.savefig(name)
    plt.show()



### load in the dict of dicts containing all the data on each person in the dataset
data_dict = pickle.load( open("../final_project/final_project_dataset.pkl", "r") )
### there's an outlier--remove it! 
data_dict.pop("TOTAL", 0)
"""
at start everything is in form of objects so dropna would not work until we apply astype('float')
drop na and then sort the df 
but -- WHY COULDN'T I SORT MY PANDAS DATAFRAME acc to the numeric values 
"""
df = pd.DataFrame(data_dict)

"""FOR EXERCISED STOCK OPITONS -- """
eso = df.loc['exercised_stock_options'].astype('float64').dropna()
print("exercised stock options  :  maximum value = ",max(eso),",  Minimum value = ",min(eso))

"""FOR SALARY -- """
sal = df.loc['salary'].astype('float64').dropna()
print("max sal : ", max(sal), "  min sal : ",min(sal))

### the input features we want to use 
### can be any key in the person-level dictionary (salary, director_fees, etc.) 
feature_1 = "salary"
feature_2 = "exercised_stock_options"
feature_3 = "total_payments"
feature_4 = "from_messages"
poi  = "poi"
features_list = [poi, feature_1, feature_4]         #insert feature_3 here when needed 
data = featureFormat(data_dict, features_list )
poi, finance_features = targetFeatureSplit( data )


### in the "clustering with 3 features" part of the mini-project,
### you'll want to change this line to 
### for f1, f2 in finance_features:
### (as it's currently written, the line below assumes 2 features)
for f1, f2 in finance_features:
    plt.scatter( f1, f2)
plt.show()


##feature sclaing below :
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit_transform(finance_features)
print scaler.transform([[200000.,1000000.]])


### cluster here; create predictions of the cluster labels
### for the data and store them to a list called pred
from sklearn.cluster import KMeans
clf = KMeans(n_clusters=2)      #your choice 
clf.fit(finance_features)
pred = clf.predict(finance_features)


# i can use fit_transform here so that feature scaling is applied automatically 
    
### rename the "name" parameter when you change the number of features
### so that the figure gets saved to a different file
try:
    Draw(pred, finance_features, poi, mark_poi=False, name="clusters.pdf", f1_name=feature_1, f2_name=feature_2)
except NameError:
    print "no predictions object named pred found, no clusters to plot"
