#!/usr/bin/python
import pandas as pd
import pickle
import sys
import matplotlib.pyplot
sys.path.append("../tools/")
from feature_format import featureFormat, targetFeatureSplit

### read in data dictionary, convert to numpy array
data_dict = pickle.load( open("../final_project/final_project_dataset.pkl", "r") )

"""print data_dict.keys()
to get the keys in key-value pairs, that is names of people in this case
"""

data_dict.pop('TOTAL', 0)
"""
data_dict.pop('SKILLING JEFFREY K',0)
data_dict.pop('LAY KENNETH L',0)

these two people above are biggest bosses of enron - they are poi for us , so we should keep them 
"""
features = ["salary", "bonus"]
data = featureFormat(data_dict, features)

df = pd.DataFrame(data_dict)

"""
print df.loc['salary',:].astype('float64').idxmax(axis=1)
this line above helped me find the outlier when 'TOTAL' was found 
"""
""" 
now we need to get 2 ppl who got bonus>5e+06 and salary>1e+06
    print [ name for name in df.columns if df.loc['bonus', name] > 5*10**6 and df.loc['salary', name] > 10**6 and df.loc['bonus',name]!='NaN']
so we used this code to pop out 2 outilers , now 2 more are present which are influencing our reg model to errors1
maybe we can use the 10 percent removal technique to remove outliers but herem we have all the data visualised so we dont need it 
"""
### your code below
for point in data:
    salary = point[0]
    bonus = point[1]
    matplotlib.pyplot.scatter( salary, bonus )

matplotlib.pyplot.xlabel("salary")
matplotlib.pyplot.ylabel("bonus")
matplotlib.pyplot.show()

