#!/usr/bin/python
# coding=utf-8

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

enron_data = pickle.load(open("../final_project/final_project_dataset.pkl", "r"))


#How many registers there is in our dataset?
print "Number of registers: {}".format(len(enron_data.keys()))

#For each POI, how many attributes available?
print "Number of attributes: {}".format(len(enron_data.values()[0]))

#How many POIs are there in E+F dataset
count=0
for i in enron_data:
    if enron_data[i]['poi'] == True:
        count+=1
print "Number of POIs: {}".format(count)

#How many POIs are there in total
poi_text = "D:/DataScience/Courses/Udacity/Data Science Foundations II/Intro Machine Learning/final_project/poi_names.txt"
poi_names = open(poi_text, 'r')
names_list = poi_names.readlines()
print "Total of POIs: {}".format(len(names_list[2:]))
poi_names.close()

#Whats the name of the features?
print "List of features: {}".format(list(enron_data.values()[0]))


#Total `stock` value from James Prentice
print "Total stock value from James Prentice: {}".format(enron_data['PRENTICE JAMES']['total_stock_value'])

#How many e-mails Wesley Colwell sent to POIs
print "Total emails sent to POI from Wesley Colwell: {}".format(enron_data['COLWELL WESLEY']['from_this_person_to_poi'])

#Total exercided stock options from Jeffrey K SKILLING
print "Total exercised stock options from Jeffrey K Skilling: {}".format(enron_data['SKILLING JEFFREY K']['exercised_stock_options'])

#Between Lay, Skilling and Fastow, who earned more money
print "Lay: {}".format(enron_data['LAY KENNETH L']['total_payments'])
print "Skilling: {}".format(enron_data['SKILLING JEFFREY K']['total_payments'])
print "Fastow: {}".format(enron_data['FASTOW ANDREW S']['total_payments'])

# How many folks in this dataset have a quantified salary?
# What about a known email address?
count_salary = 0
count_email = 0
for key in enron_data.keys():
    if enron_data[key]['salary'] != 'NaN':
        count_salary +=1
    if enron_data[key]['email_address'] != 'NaN':
        count_email +=1
print "Registers with quantified salary: {}".format(count_salary)
print "Registers with know email address: {}".format(count_email)

# How many people in the E+F dataset (as it currently exists) have “NaN” for their total payments?
# What percentage of people in the dataset as a whole is this?
count_NaN = 0
for key in enron_data.keys():
    if enron_data[key]['total_payments'] == 'NaN':
        count_NaN +=1
print "Total of NaN in total_payments: {}".format(count_NaN)
print float(count_NaN)/len(enron_data.keys())


# How many POIs in the E+F dataset (as it currently exists) have “NaN” for their total payments?
# What percentage of people in the dataset as a whole is this?
count_poi_NaN = 0
for key in enron_data.keys():
    if enron_data[key]['total_payments'] == 'NaN' and enron_data[key]['poi'] == True:
        count_poi_NaN +=1
print "Total of POI with NaN in total_payments: {}".format(count_poi_NaN)
print float(count_poi_NaN)/len(enron_data.keys())
