#!/usr/bin/python

import pickle
import sys
import matplotlib.pyplot
sys.path.append("../tools/")
from feature_format import featureFormat, targetFeatureSplit


### read in data dictionary, convert to numpy array
data_dict = pickle.load( open("../final_project/final_project_dataset.pkl", "r") )
data_dict.pop('TOTAL', 0) #dropping the 'TOTAL' key from dict
features = ["salary", "bonus"]
data = featureFormat(data_dict, features)


### your code below

for point in data:
    salary = point[0]
    bonus = point[1]
    matplotlib.pyplot.scatter( salary, bonus )

matplotlib.pyplot.xlabel("salary")
matplotlib.pyplot.ylabel("bonus")
matplotlib.pyplot.show()

#To find outlier
for key, value in data_dict.items():
    if value['bonus'] == data.max():
        print key

#Two outliers that received Bonus > 5kk and salary > 1kk
from pprint import pprint
outliers_salary = []
for key in data_dict:
    val = data_dict[key]['salary']
    if val == 'NaN':
        continue
    outliers_salary.append((key,int(val)))

pprint(sorted(outliers_salary,key=lambda x:x[1],reverse=True)[:5])

outliers_bonus = []
for key in data_dict:
    val = data_dict[key]['bonus']
    if val == 'NaN':
        continue
    outliers_bonus.append((key,int(val)))

pprint(sorted(outliers_bonus,key=lambda x:x[1],reverse=True)[:5])
