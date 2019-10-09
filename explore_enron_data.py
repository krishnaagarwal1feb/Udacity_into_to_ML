#!/usr/bin/python

""" 
    Starter code for exploring the Enron dataset (emails + finances);
    loads up the dataset (pickled dict of dicts).

    The dataset has the form:
    enron_data["LASTNAME FIRSTNAME MIDDLEINITIAL"] = { features_dict }

    {features_dict} is a dictionary of features associated with that person.
    You should explore features_dict as part of the mini-project,
    but here's an example to get you started:

    enron_data["SKILLING JEFFREY K"]["bonus"] = 5600000
    
"""

import pickle

enron_data = pickle.load(open("../final_project/final_project_dataset.pkl", "rb"))
#146 by 21 
pois = sum(v.get("poi", 0) == 1 for v in enron_data.values())
#to acces a dict of dict - accessing its key-value pairs  --
enron_data['PRENTICE JAMES']["total_stock_value"]
enron_data['COLWELL WESLEY']['from_this_person_to_poi']

#ROLES of below three person during the fraud time 
#the CEO
enron_data['SKILLING JEFFREY K']['total_payments']
#the CFO
enron_data['FASTOW ANDREW S']['total_payments']
#the CHAIRMAN
enron_data['LAY KENNETH L']['total_payments']

"""how many folks have quantified salary (is not a NaN)"""
counter = sum(each["salary"] !='NaN' for each in enron_data.values())
valid_email = sum(each["email_address"] !='NaN' 
                  for each in enron_data.values())
total_pay_nan = sum(each["total_payments"] == 'NaN'
                    for each in enron_data.values())
total_pay_nan_poi = sum((each['poi'] and each['total_payments']=='NaN')
                        for each in enron_data.values())