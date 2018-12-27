#!/usr/bin/python

import sys
import pickle
import os
import numpy as np
from time import time
from sklearn.model_selection import train_test_split


sys.path.append("../tools/")
os.chdir(r"D:\DataScience\Courses\Udacity\Data Science Foundations II\Intro Machine Learning\tools")
from feature_format import featureFormat, targetFeatureSplit
os.chdir(r"D:\DataScience\Courses\Udacity\Data Science Foundations II\Intro Machine Learning\final_project")
from tester import dump_classifier_and_data


### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)

### Exploring the Dataset
# How many registers are in our dataset?
print len(data_dict) #146

# How many of them are POI? # 18
count_poi = 0
key_poi = []

for i in data_dict:
    if data_dict[i]['poi']==1:
        count_poi +=1
        key_poi.append(i)
print count_poi
print key_poi

# What are the features of our dataset?
features_list = str(data_dict['HANNON KEVIN P'].keys())
print features_list #21

# Let's check the correlation from some features and see if we find something interesting
from matplotlib import pyplot

#Check #1
features_1 = ['salary', 'from_this_person_to_poi']
data_1 = featureFormat(data_dict, features_1)

for point in data_1:
    salary = point[0]
    from_this_person_to_poi = point[1]
    pyplot.scatter( salary, from_this_person_to_poi )
print "\nPrinting Salary x Messages to POI"
pyplot.xlabel("salary")
pyplot.ylabel("from_this_person_to_poi")
pyplot.show()

#Checking the outlier
for i in data_dict:
    if data_dict[i]['salary'] == data_1.max():
        print "Name:", i #TOTAL
        
# Removing TOTAL
data_dict.pop('TOTAL', None)


#Check #2
features_2 = ['salary', 'bonus']
data_2 = featureFormat(data_dict, features_2)

for point in data_2:
    salary = point[0]
    bonus = point[1]
    pyplot.scatter( salary, bonus)
print "\nPrinting Salary x Bonus"
pyplot.xlabel("salary")
pyplot.ylabel("bonus")
pyplot.show() #Still more 3 outliers considering salary and 1 outlier considering bonus

#Checking bonus outlier
for i in data_dict:
    if data_dict[i]['bonus'] == data_2.max():
        print "Name:", i, data_dict[i]['bonus'] # LAVORATO JOHN J
        
#Checking if he is POI
print (data_dict['LAVORATO JOHN J']['poi'] == 1) # Not POI


#Checking salary outlier
outlier_salary = []        
for i in data_dict:
    n = data_dict[i]['salary']
    if n == 'NaN':
        continue
    outlier_salary.append((i,n))
    
three_firsts = sorted(outlier_salary,key=lambda x:x[1],reverse=True)[:3] 
print three_firsts #SKILLING JEFFREY K, LAY KENNETH L, FREVERT MARK A

# Checking if they are POIs
for i in three_firsts:
    if data_dict[i[0]]['poi'] == 1:
        print i[0] #SKILLING JEFFREY K, LAY KENNETH L are POIs
        
# Lets check average salary and bonus
salary = []
bonus = []

for i in data_dict:
    if data_dict[i]['salary'] != 'NaN':
        salary.append(data_dict[i]['salary'])
for i in data_dict:
    if data_dict[i]['bonus'] != 'NaN':
        bonus.append(data_dict[i]['bonus'])
        
print "Average salary:", np.mean(salary)
print "Average bonus:", np.mean(bonus)

#This average is not what I expected... considering the TOTAL had 26.704.229, dividing it into 146 the result is different.
#Checking 'NaN' values
salary_nan = 0
bonus_nan = 0

for i in data_dict:
    if data_dict[i]['salary'] == 'NaN':
        salary_nan += 1
for i in data_dict:
    if data_dict[i]['bonus'] == 'NaN':
        bonus_nan += 1
print "Amount of NaN salary:", salary_nan # 51
print "Amount of NaN bonus:", bonus_nan # 64


#Check #3
features_3 = ['total_payments', 'total_stock_value']
data_3 = featureFormat(data_dict, features_3)

for point in data_3:
    total_payments = point[0]
    total_stock_value = point[1]
    pyplot.scatter(total_payments, total_stock_value )
print "\nPrinting Total Payments x Total Stock Value"
pyplot.xlabel("total_payments")
pyplot.ylabel("total_stock_value")
pyplot.show()

# Another big outlier here for total_payments lets check
for i in data_dict:
    if data_dict[i]['total_payments'] == data_3.max():
        print "Name:", i, data_dict[i]['total_payments'], data_dict[i]['total_stock_value'] # LAY KENNETH L

my_dataset = data_dict

#Creating new features... the fraction of e-mails sent/received to/from POIs per total e-mails
#I have found the functions below in the git hub from user 'zelite'. Lines 152 until 206. I did some changes, but the main logic is from him.
def get_total_list(key1, key2):
    'combine 2 lists in one, assign NaN to 0'
    new_list = []
    for i in data_dict: #data_dict
        # assign NaN to 0
        if data_dict[i][key1] == 'NaN' or data_dict[i][key2] == 'NaN': #data_dict
            new_list.append(0.)
        elif data_dict[i][key1]>=0: #data_dict
            new_list.append(float(data_dict[i][key1]) + float(data_dict[i][key2]))
    return new_list

# get the total poi related emails:
total_poi_emails = get_total_list('from_this_person_to_poi', 'from_poi_to_this_person')

# get the total emails
total_emails = get_total_list('to_messages', 'from_messages')

def fraction_list(list1, list2):
    'divide one list by another'
    fraction = []
    for i in range(0,len(list1)):
        if list2[i] == 0.0:
            fraction.append(0.0)
        else:
            fraction.append(float(list1[i])/float(list2[i]))
            #print fraction
    return fraction

# get the fraction of poi emails
fraction_poi_emails = fraction_list(total_poi_emails, total_emails)

# add this new feature to my_dataset

count = 0
for i in data_dict: #data_dict
    data_dict[i]['fraction_poi_emails'] = fraction_poi_emails[count] #data_dict
    count += 1

# test
print 'LAY KENNETH:', my_dataset['LAY KENNETH L']['fraction_poi_emails']

# let's test if this feature has any correlation with POIs
       
new_features_list = ['poi', 'fraction_poi_emails']
data = featureFormat(data_dict, new_features_list) #data_dict

for point in data:
    poi = point[0]
    fraction_poi_emails = point[1]
    if point[0] == 1:
        pyplot.scatter(poi, fraction_poi_emails, color = 'r')
    else:
        pyplot.scatter(poi, fraction_poi_emails, color = 'b')

pyplot.xlabel("poi")
pyplot.ylabel("fraction_poi_emails")
pyplot.show()


poi_fraction = []
not_poi_fraction = []

for i in data_dict: #data_dict
    if data_dict[i]['poi'] == 1:
        poi_fraction.append(data_dict[i]['fraction_poi_emails'])
    else:
        not_poi_fraction.append(data_dict[i]['fraction_poi_emails'])
        
print "Poi fraction:", np.mean(poi_fraction)
print "Not Poi fraction:", np.mean(not_poi_fraction)


#POI and Not-POI comparison
salary_notpoi = []
salary_poi = []
bonus_notpoi = []
bonus_poi = []

for i in data_dict:
    if data_dict[i]['salary'] != 'NaN' and data_dict[i]['poi'] == 1:
        salary_poi.append(data_dict[i]['salary'])
    elif data_dict[i]['salary'] != 'NaN' and data_dict[i]['poi'] == 0:
        salary_notpoi.append(data_dict[i]['salary'])
for i in data_dict:
    if data_dict[i]['bonus'] != 'NaN' and data_dict[i]['poi'] == 1:
        bonus_poi.append(data_dict[i]['bonus'])
    elif data_dict[i]['bonus'] != 'NaN' and data_dict[i]['poi'] == 0:
        bonus_notpoi.append(data_dict[i]['bonus'])

#create means from lists
mean_salary_notpoi =  np.mean(salary_notpoi)
mean_salary_poi = np.mean(salary_poi)
mean_bonus_notpoi = np.mean(bonus_notpoi)
mean_bonus_poi = np.mean(bonus_poi)


# create plot for salary
n_groups = 1
fig, ax = pyplot.subplots()
index = np.arange(n_groups)
bar_width = 1
opacity = 0.8
 
rects1 = pyplot.bar(index, mean_salary_notpoi, bar_width,
                 alpha=opacity,
                 color='b',
                 label='Not POIs mean Salary')
 
rects2 = pyplot.bar(index + bar_width, mean_salary_poi, bar_width,
                 alpha=opacity,
                 color='r',
                 label='POIs mean Salary')
 
pyplot.xlabel('Group')
pyplot.ylabel('Mean Salary')
pyplot.title('Salary comparison between POIs and Not POIs')
ax.axhline(284088, color="green", label='Dataset mean Salary')
pyplot.legend()
pyplot.tight_layout()
pyplot.show()

# create plot for bonus
n_groups = 1
fig, ax = pyplot.subplots()
index = np.arange(n_groups)
bar_width = 1
opacity = 0.8
 
rects1 = pyplot.bar(index, mean_bonus_notpoi, bar_width,
                 alpha=opacity,
                 color='b',
                 label='Not POIs mean Bonus')
 
rects2 = pyplot.bar(index + bar_width, mean_bonus_poi, bar_width,
                 alpha=opacity,
                 color='r',
                 label='POIs mean Bonus')
 
pyplot.xlabel('Group')
pyplot.ylabel('Mean Bonus')
pyplot.title('Bonus comparison between POIs and Not POIs')
ax.axhline(1201773, color="green", label='Dataset mean Bonus')
pyplot.legend()
pyplot.tight_layout()
pyplot.show()

print "Salary difference from POI and Not POI:", (mean_salary_poi/mean_salary_notpoi)
print "Bonus difference from POI and Not POI:", (mean_bonus_poi/mean_bonus_notpoi)


#Checking other features with missing values
to_messages_nan = 0
deferral_payments_nan = 0
total_payments_nan = 0
exercised_stock_options_nan = 0
restricted_stock_nan = 0
shared_receipt_with_poi_nan = 0
restricted_stock_deferred_nan = 0
total_stock_value_nan = 0
expenses_nan = 0
loan_advances_nan = 0
from_messages_nan = 0
other_nan = 0
from_this_person_to_poi_nan = 0
director_fees_nan = 0
deferred_income_nan = 0
long_term_incentive_nan = 0
from_poi_to_this_person_nan = 0


for i in data_dict:
    if data_dict[i]['to_messages'] == 'NaN':
        to_messages_nan += 1
for i in data_dict:
    if data_dict[i]['deferral_payments'] == 'NaN':
        deferral_payments_nan += 1
for i in data_dict:
    if data_dict[i]['total_payments'] == 'NaN':
        total_payments_nan += 1
for i in data_dict:
    if data_dict[i]['exercised_stock_options'] == 'NaN':
        exercised_stock_options_nan += 1
for i in data_dict:
    if data_dict[i]['restricted_stock'] == 'NaN':
        restricted_stock_nan += 1
for i in data_dict:
    if data_dict[i]['shared_receipt_with_poi'] == 'NaN':
        shared_receipt_with_poi_nan += 1
for i in data_dict:
    if data_dict[i]['restricted_stock_deferred'] == 'NaN':
        restricted_stock_deferred_nan += 1
for i in data_dict:
    if data_dict[i]['total_stock_value'] == 'NaN':
        total_stock_value_nan += 1
for i in data_dict:
    if data_dict[i]['expenses'] == 'NaN':
        expenses_nan += 1
for i in data_dict:
    if data_dict[i]['loan_advances'] == 'NaN':
        loan_advances_nan += 1
for i in data_dict:
    if data_dict[i]['from_messages'] == 'NaN':
        from_messages_nan += 1
for i in data_dict:
    if data_dict[i]['other'] == 'NaN':
        other_nan += 1
for i in data_dict:
    if data_dict[i]['from_this_person_to_poi'] == 'NaN':
        from_this_person_to_poi_nan += 1
for i in data_dict:
    if data_dict[i]['director_fees'] == 'NaN':
        director_fees_nan += 1
for i in data_dict:
    if data_dict[i]['deferred_income'] == 'NaN':
        deferred_income_nan += 1
for i in data_dict:
    if data_dict[i]['long_term_incentive'] == 'NaN':
        long_term_incentive_nan += 1             
for i in data_dict:
    if data_dict[i]['from_poi_to_this_person'] == 'NaN':
        from_poi_to_this_person_nan += 1


print "Amount of NaN to_messages:", to_messages_nan
print "Amount of NaN deferral_payments:", deferral_payments_nan
print "Amount of NaN total_payments:", total_payments_nan
print "Amount of NaN exercised_stock_options:", exercised_stock_options_nan
print "Amount of NaN restricted_stock:", restricted_stock_nan
print "Amount of NaN shared_receipt_with_poi:", shared_receipt_with_poi_nan
print "Amount of NaN restricted_stock_deferred:", restricted_stock_deferred_nan
print "Amount of NaN total_stock_value:", total_stock_value_nan
print "Amount of NaN expenses:", expenses_nan
print "Amount of NaN loan_advances:", loan_advances_nan
print "Amount of NaN from_messages:", from_messages_nan
print "Amount of NaN other:", other_nan
print "Amount of NaN from_this_person_to_poi:", from_this_person_to_poi_nan
print "Amount of NaN director_fees:", director_fees_nan
print "Amount of NaN deferred_income:", deferred_income_nan
print "Amount of NaN long_term_incentive:", long_term_incentive_nan
print "Amount of NaN from_poi_to_this_person:", from_poi_to_this_person_nan       
        





#Try a varity of classifiers
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedShuffleSplit


#Validation function
PERF_FORMAT_STRING = "\
\tAccuracy: {:>0.{display_precision}f}\tPrecision: {:>0.{display_precision}f}\t\
Recall: {:>0.{display_precision}f}\tF1: {:>0.{display_precision}f}\tF2: {:>0.{display_precision}f}"
RESULTS_FORMAT_STRING = "\tTotal predictions: {:4d}\tTrue positives: {:4d}\tFalse positives: {:4d}\
\tFalse negatives: {:4d}\tTrue negatives: {:4d}"

def test_classifier(clf, dataset, feature_list, folds = 1000):
    data = featureFormat(dataset, feature_list, sort_keys = True)
    labels, features = targetFeatureSplit(data)
    
    cv = StratifiedShuffleSplit(1000, random_state = 42)
    true_negatives = 0
    false_negatives = 0
    true_positives = 0
    false_positives = 0
    for train_idx, test_idx in cv.split(features, labels):
        features_train = []
        features_test  = []
        labels_train   = []
        labels_test    = []
        for ii in train_idx:
            features_train.append( features[ii] )
            labels_train.append( labels[ii] )
        for jj in test_idx:
            features_test.append( features[jj] )
            labels_test.append( labels[jj] )
        
        ### fit the classifier using training set, and test on test set
        clf.fit(features_train, labels_train)
        predictions = clf.predict(features_test)
        for prediction, truth in zip(predictions, labels_test):
            if prediction == 0 and truth == 0:
                true_negatives += 1
            elif prediction == 0 and truth == 1:
                false_negatives += 1
            elif prediction == 1 and truth == 0:
                false_positives += 1
            elif prediction == 1 and truth == 1:
                true_positives += 1
            else:
                print "Warning: Found a predicted label not == 0 or 1."
                print "All predictions should take value 0 or 1."
                print "Evaluating performance for processed predictions:"
                break
    try:
        total_predictions = true_negatives + false_negatives + false_positives + true_positives
        accuracy = 1.0*(true_positives + true_negatives)/total_predictions
        precision = 1.0*true_positives/(true_positives+false_positives)
        recall = 1.0*true_positives/(true_positives+false_negatives)
        f1 = 2.0 * true_positives/(2*true_positives + false_positives+false_negatives)
        f2 = (1+2.0*2.0) * precision*recall/(4*precision + recall)
        print clf
        print PERF_FORMAT_STRING.format(accuracy, precision, recall, f1, f2, display_precision = 5)
        print RESULTS_FORMAT_STRING.format(total_predictions, true_positives, false_positives, false_negatives, true_negatives)
        print ""
    except:
        print "Got a divide by zero when trying out:", clf
        print "Precision or recall may be undefined due to a lack of true positive predicitons."

###########################

features_list = ['poi', 'salary', 'to_messages', 'deferral_payments', 'total_payments', 'exercised_stock_options', 'bonus', 
                 'restricted_stock', 'shared_receipt_with_poi', 'restricted_stock_deferred', 'total_stock_value', 'expenses',
                 'loan_advances', 'from_messages', 'other', 'from_this_person_to_poi', 'director_fees', 'deferred_income',
                 'long_term_incentive', 'from_poi_to_this_person'] #removed e-mail address

### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = True) #my_dataset
labels, features = targetFeatureSplit(data)

#split data into train and test
features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size = 0.30, random_state =42)


print "Number of training points: ", len(features_train)
print "Number of features: ", len(features_list)

#SVM
svm = LinearSVC()
t0 = time()
svm.fit(features_train, labels_train)
print "training_time:", round(time()-t0, 3), "s"
t0 = time()
pred = svm.predict(features_test)
print "prediction time:", round(time()-t0, 3), "s"
test_classifier(svm, my_dataset, features_list)

#Naive Bayes
nb = GaussianNB()
t0 = time()
nb.fit(features_train, labels_train)
print "training_time:", round(time()-t0, 3), "s"
t0 = time()
pred = nb.predict(features_test)
print "prediction time:", round(time()-t0, 3), "s"
test_classifier(nb, my_dataset, features_list)


#DecisionTree
dtc = DecisionTreeClassifier()
t0 = time()
dtc.fit(features_train, labels_train)
print "training_time:", round(time()-t0, 3), "s"
t0 = time()
pred = dtc.predict(features_test)
print "prediction time:", round(time()-t0, 3), "s"
test_classifier(dtc, my_dataset, features_list)
importances = dtc.feature_importances_
indices = np.argsort(importances)[::-1]
print 'Feature Ranking: '
for i in range(10):
    print "{} feature {} ({})".format(i+1,features_list[i+1],importances[indices[i]])
 

#RandomForest
rfc = RandomForestClassifier()
t0 = time()
rfc.fit(features_train, labels_train)
print "training time:", round(time()-t0, 3), "s"
t0 = time()
pred = rfc.predict(features_test)
print "prediction time:", round(time()-t0, 3), "s"
test_classifier(rfc, my_dataset, features_list)



#Using the 5 more important features accordling to DecisionTree
five_top_features_list = ['poi', 'salary', 'to_messages', 'deferral_payments', 'total_payments', 'exercised_stock_options']

data = featureFormat(my_dataset, five_top_features_list, sort_keys = True) #my_dataset
labels, features = targetFeatureSplit(data)

features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size = 0.3, random_state =42)


#SVM
svm = LinearSVC()
t0 = time()
svm.fit(features_train, labels_train)
print "training_time:", round(time()-t0, 3), "s"
t0 = time()
pred = svm.predict(features_test)
print "prediction time:", round(time()-t0, 3), "s"
test_classifier(svm, my_dataset, five_top_features_list)


#Naive Bayes
nb = GaussianNB()
t0 = time()
nb.fit(features_train, labels_train)
print "training_time:", round(time()-t0, 3), "s"
t0 = time()
pred = nb.predict(features_test)
print "prediction time:", round(time()-t0, 3), "s"
test_classifier(nb, my_dataset, five_top_features_list)


#DecidionTree
dtc = DecisionTreeClassifier()
t0 = time()
dtc.fit(features_train, labels_train)
print "training_time:", round(time()-t0, 3), "s"
t0 = time()
pred = dtc.predict(features_test)
print "prediction time:", round(time()-t0, 3), "s"
test_classifier(dtc, my_dataset, five_top_features_list)

#RandomForest
rfc = RandomForestClassifier()
t0 = time()
rfc.fit(features_train, labels_train)
print "training time:", round(time()-t0, 3), "s"
t0 = time()
pred = rfc.predict(features_test)
print "prediction time:", round(time()-t0, 3), "s"
test_classifier(rfc, my_dataset, five_top_features_list)

#Using the 3 more important features accordling to DecisionTree
three_top_features_list = ['poi', 'salary', 'to_messages', 'deferral_payments']

data = featureFormat(my_dataset, three_top_features_list, sort_keys = True) #my_dataset
labels, features = targetFeatureSplit(data)

features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size = 0.3, random_state =42)


#SVM
svm = LinearSVC()
t0 = time()
svm.fit(features_train, labels_train)
print "training_time:", round(time()-t0, 3), "s"
t0 = time()
pred = svm.predict(features_test)
print "prediction time:", round(time()-t0, 3), "s"
test_classifier(svm, my_dataset, three_top_features_list)


#Naive Bayes
nb = GaussianNB()
t0 = time()
nb.fit(features_train, labels_train)
print "training_time:", round(time()-t0, 3), "s"
t0 = time()
pred = nb.predict(features_test)
print "prediction time:", round(time()-t0, 3), "s"
test_classifier(nb, my_dataset, three_top_features_list)
#DecidionTree
dtc = DecisionTreeClassifier()
t0 = time()
dtc.fit(features_train, labels_train)
print "training_time:", round(time()-t0, 3), "s"
t0 = time()
pred = dtc.predict(features_test)
print "prediction time:", round(time()-t0, 3), "s"
test_classifier(dtc, my_dataset, three_top_features_list)


#RandomForest
rfc = RandomForestClassifier()
t0 = time()
rfc.fit(features_train, labels_train)
print "training time:", round(time()-t0, 3), "s"
t0 = time()
pred = rfc.predict(features_test)
print "prediction time:", round(time()-t0, 3), "s"
test_classifier(rfc, my_dataset, three_top_features_list)

#Using the more important feature accordling to DecisionTree
top_features_list = ['poi', 'salary']

data = featureFormat(my_dataset, top_features_list, sort_keys = True) #my_dataset
labels, features = targetFeatureSplit(data)

features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size = 0.3, random_state =42)


#SVM
svm = LinearSVC()
t0 = time()
svm.fit(features_train, labels_train)
print "training_time:", round(time()-t0, 3), "s"
t0 = time()
pred = svm.predict(features_test)
print "prediction time:", round(time()-t0, 3), "s"
test_classifier(svm, my_dataset, top_features_list)


#Naive Bayes
nb = GaussianNB()
t0 = time()
nb.fit(features_train, labels_train)
print "training_time:", round(time()-t0, 3), "s"
t0 = time()
pred = nb.predict(features_test)
print "prediction time:", round(time()-t0, 3), "s"
test_classifier(nb, my_dataset, top_features_list)


#DecidionTree
dtc = DecisionTreeClassifier()
t0 = time()
dtc.fit(features_train, labels_train)
print "training_time:", round(time()-t0, 3), "s"
t0 = time()
pred = dtc.predict(features_test)
print "prediction time:", round(time()-t0, 3), "s"
test_classifier(dtc, my_dataset, top_features_list)


#RandomForest
rfc = RandomForestClassifier()
t0 = time()
rfc.fit(features_train, labels_train)
print "training time:", round(time()-t0, 3), "s"
t0 = time()
pred = rfc.predict(features_test)
print "prediction time:", round(time()-t0, 3), "s"
test_classifier(rfc, my_dataset, top_features_list)

#Using the 7 more important features accordling to DecisionTree
seven_top_features_list = ['poi', 'salary', 'to_messages', 'deferral_payments', 'total_payments', 'exercised_stock_options', 'bonus',
                          'restricted_stock']

data = featureFormat(my_dataset, seven_top_features_list, sort_keys = True) #my_dataset
labels, features = targetFeatureSplit(data)

features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size = 0.3, random_state =42)

#SVM
svm = LinearSVC()
t0 = time()
svm.fit(features_train, labels_train)
print "training_time:", round(time()-t0, 3), "s"
t0 = time()
pred = svm.predict(features_test)
print "prediction time:", round(time()-t0, 3), "s"
test_classifier(svm, my_dataset, seven_top_features_list)


#Naive Bayes
nb = GaussianNB()
t0 = time()
nb.fit(features_train, labels_train)
print "training_time:", round(time()-t0, 3), "s"
t0 = time()
pred = nb.predict(features_test)
print "prediction time:", round(time()-t0, 3), "s"
test_classifier(nb, my_dataset, seven_top_features_list)


#DecidionTree
dtc = DecisionTreeClassifier()
t0 = time()
dtc.fit(features_train, labels_train)
print "training_time:", round(time()-t0, 3), "s"
t0 = time()
pred = dtc.predict(features_test)
print "prediction time:", round(time()-t0, 3), "s"
test_classifier(dtc, my_dataset, seven_top_features_list)


#RandomForest
rfc = RandomForestClassifier()
t0 = time()
rfc.fit(features_train, labels_train)
print "training time:", round(time()-t0, 3), "s"
t0 = time()
pred = rfc.predict(features_test)
print "prediction time:", round(time()-t0, 3), "s"
test_classifier(rfc, my_dataset, seven_top_features_list)



#Using the 10 more important features accordling to DecisionTree
ten_top_features_list = ['poi', 'salary', 'to_messages', 'deferral_payments', 'total_payments', 'exercised_stock_options', 'bonus',
                          'restricted_stock', 'shared_receipt_with_poi', 'restricted_stock_deferred', 'total_stock_value']

data = featureFormat(my_dataset, ten_top_features_list, sort_keys = True) #my_dataset
labels, features = targetFeatureSplit(data)

features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size = 0.3, random_state =42)

#SVM
svm = LinearSVC()
t0 = time()
svm.fit(features_train, labels_train)
print "training_time:", round(time()-t0, 3), "s"
t0 = time()
pred = svm.predict(features_test)
print "prediction time:", round(time()-t0, 3), "s"
test_classifier(svm, my_dataset, ten_top_features_list)


#Naive Bayes
nb = GaussianNB()
t0 = time()
nb.fit(features_train, labels_train)
print "training_time:", round(time()-t0, 3), "s"
t0 = time()
pred = nb.predict(features_test)
print "prediction time:", round(time()-t0, 3), "s"
test_classifier(nb, my_dataset, ten_top_features_list)


#DecidionTree
dtc = DecisionTreeClassifier()
t0 = time()
dtc.fit(features_train, labels_train)
print "training_time:", round(time()-t0, 3), "s"
t0 = time()
pred = dtc.predict(features_test)
print "prediction time:", round(time()-t0, 3), "s"
test_classifier(dtc, my_dataset, ten_top_features_list)

#RandomForest
rfc = RandomForestClassifier()
t0 = time()
rfc.fit(features_train, labels_train)
print "training time:", round(time()-t0, 3), "s"
t0 = time()
pred = rfc.predict(features_test)
print "prediction time:", round(time()-t0, 3), "s"
test_classifier(rfc, my_dataset, ten_top_features_list)


#Trying with new feature

new_features_list = ['poi', 'fraction_poi_emails', 'salary']

data = featureFormat(data_dict, new_features_list, sort_keys = True) #data_dict
labels, features = targetFeatureSplit(data)
features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size = 0.3, random_state =42)


#SVM
svm = LinearSVC()
t0 = time()
svm.fit(features_train, labels_train)
print "training_time:", round(time()-t0, 3), "s"
t0 = time()
pred = svm.predict(features_test)
print "prediction time:", round(time()-t0, 3), "s"
test_classifier(svm, my_dataset, new_features_list)


#Naive Bayes
nb = GaussianNB()
t0 = time()
nb.fit(features_train, labels_train)
print "training_time:", round(time()-t0, 3), "s"
t0 = time()
pred = nb.predict(features_test)
print "prediction time:", round(time()-t0, 3), "s"
test_classifier(nb, my_dataset, new_features_list)

#DecidionTree
dtc = DecisionTreeClassifier()
t0 = time()
dtc.fit(features_train, labels_train)
print "training_time:", round(time()-t0, 3), "s"
t0 = time()
pred = dtc.predict(features_test)
print "prediction time:", round(time()-t0, 3), "s"
test_classifier(dtc, my_dataset, new_features_list)

#RandomForest
rfc = RandomForestClassifier()
t0 = time()
rfc.fit(features_train, labels_train)
print "training time:", round(time()-t0, 3), "s"
t0 = time()
pred = rfc.predict(features_test)
print "prediction time:", round(time()-t0, 3), "s"
test_classifier(rfc, my_dataset, new_features_list)




from sklearn.feature_selection import SelectKBest
from sklearn.model_selection import GridSearchCV

#Lets try improve our results with SelectKBest
#Preparing data...
features_list = ['poi', 'salary', 'to_messages', 'deferral_payments', 'total_payments', 'exercised_stock_options', 'bonus', 
                 'restricted_stock', 'shared_receipt_with_poi', 'restricted_stock_deferred', 'total_stock_value', 'expenses',
                 'loan_advances', 'from_messages', 'other', 'from_this_person_to_poi', 'director_fees', 'deferred_income',
                 'long_term_incentive', 'from_poi_to_this_person']
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)

# Creating selector
selector = SelectKBest(k=8)
selectedFeatures = selector.fit(features,labels)
feature_names = [features_list[i] for i in selectedFeatures.get_support(indices=True)]
print 'Selector: ', feature_names
# ['poi', 'total_payments', 'exercised_stock_options', 'bonus', 'restricted_stock_deferred', 'director_fees', 'deferred_income']


kbest_features_list = ['poi', 'deferral_payments', 'total_payments', 'exercised_stock_options', 'bonus', 
                       'restricted_stock_deferred', 'director_fees', 'deferred_income']
data = featureFormat(my_dataset, kbest_features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)
features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size = 0.3, random_state =42)

#SVM
svm = LinearSVC()
t0 = time()
svm.fit(features_train, labels_train)
print "training_time:", round(time()-t0, 3), "s"
t0 = time()
pred = svm.predict(features_test)
print "prediction time:", round(time()-t0, 3), "s"
test_classifier(svm, my_dataset, kbest_features_list)


#Naive Bayes
nb = GaussianNB()
t0 = time()
nb.fit(features_train, labels_train)
print "training_time:", round(time()-t0, 3), "s"
t0 = time()
pred = nb.predict(features_test)
print "prediction time:", round(time()-t0, 3), "s"
test_classifier(nb, my_dataset, kbest_features_list)


#DecidionTree
dtc = DecisionTreeClassifier()
t0 = time()
dtc.fit(features_train, labels_train)
print "training_time:", round(time()-t0, 3), "s"
t0 = time()
pred = dtc.predict(features_test)
print "prediction time:", round(time()-t0, 3), "s"
test_classifier(dtc, my_dataset, kbest_features_list)

#RandomForest
rfc = RandomForestClassifier()
t0 = time()
rfc.fit(features_train, labels_train)
print "training time:", round(time()-t0, 3), "s"
t0 = time()
pred = rfc.predict(features_test)
print "predicting time:", round(time()-t0, 3), "s"
test_classifier(rfc, my_dataset, kbest_features_list)



#Tunning SVM LinearSVC and Naive Bayes with GridSeachCV
# I will use the top 7 features from SelectKBest
kbest_features_list = ['poi', 'deferral_payments', 'total_payments', 'exercised_stock_options', 'bonus', 
                       'restricted_stock_deferred', 'director_fees', 'deferred_income']

data = featureFormat(my_dataset, kbest_features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)

features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size = 0.3, random_state =42)

#DecisionTree
t0 = time()
param_grid = {
             'max_depth':[4, 8, 16, 32, 80, 120, 150, 200, 500],
                 'min_samples_split':[0.1, 0.5, 2, 4, 8, 16, 32],
                     'min_samples_leaf':[0.1, 0.5, 1, 2, 5, 10, 20],
                         'max_features':[1, 2, 3, 4, 5, 6, 7]
                          }
dtc = GridSearchCV(DecisionTreeClassifier(), param_grid)
dtc = dtc.fit(features_train, labels_train)
print "done in %0.3fs" % (time() - t0)
print "Best estimator:"
print dtc.best_estimator_

#Testing with Best parameters estimated
#DecidionTree
dtc = DecisionTreeClassifier(class_weight=None, criterion='gini', max_depth=8,
            max_features=2, max_leaf_nodes=None, min_impurity_decrease=0.0,
            min_impurity_split=None, min_samples_leaf=2,
            min_samples_split=0.1, min_weight_fraction_leaf=0.0,
            presort=False, random_state=None, splitter='best')
t0 = time()
dtc.fit(features_train, labels_train)
print "training_time:", round(time()-t0, 3), "s"
t0 = time()
pred = dtc.predict(features_test)
print "prediction time:", round(time()-t0, 3), "s"
test_classifier(dtc, my_dataset, kbest_features_list)




#RandomForest
t0 = time()
param_grid = { 'bootstrap': [True],
    'max_depth': [80, 90, 100, 110],
    'max_features': [2, 3],
    'min_samples_leaf': [3, 4, 5],
    'min_samples_split': [8, 10, 12],
    'n_estimators': [100, 200, 300, 1000]}
rfc = GridSearchCV(RandomForestClassifier(), param_grid)
rfc = rfc.fit(features_train, labels_train)
print "done in %0.3fs" % (time() - t0)
print "Best estimator:"
print rfc.best_estimator_

#RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
 #           max_depth=50, max_features='auto', max_leaf_nodes=None,
  #          min_impurity_decrease=0.0, min_impurity_split=None,
   #         min_samples_leaf=1, min_samples_split=5,
    #        min_weight_fraction_leaf=0.0, n_estimators=200, n_jobs=None,
     #       oob_score=False, random_state=None, verbose=0,
      #      warm_start=False)


#RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
 #           max_depth=80, max_features=2, max_leaf_nodes=None,
  #          min_impurity_decrease=0.0, min_impurity_split=None,
   #         min_samples_leaf=3, min_samples_split=8,
    #        min_weight_fraction_leaf=0.0, n_estimators=100, n_jobs=None,
     #       oob_score=False, random_state=None, verbose=0,
      #      warm_start=False)


rfc = RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
            max_depth=30, max_features='auto', max_leaf_nodes=None,
            min_impurity_decrease=0.0, min_impurity_split=None,
            min_samples_leaf=1, min_samples_split=3,
            min_weight_fraction_leaf=0.0, n_estimators=20, n_jobs=None,
            oob_score=False, random_state=None, verbose=0,
            warm_start=False)
t0 = time()
rfc.fit(features_train, labels_train)
print "training time:", round(time()-t0, 3), "s"
t0 = time()
pred = rfc.predict(features_test)
print "predicting time:", round(time()-t0, 3), "s"
test_classifier(rfc, my_dataset, kbest_features_list)





#Comparing both classificers best result:
# data to plot
n_groups = 4
values_DecisionTree = (0.82, 0.34, 0.36, 0.35)
values_RandomForest = (0.88, 0.59, 0.3, 0.4)
 
# create plot
fig, ax = pyplot.subplots()
index = np.arange(n_groups)
bar_width = 0.35
opacity = 0.8
 
rects1 = pyplot.bar(index, values_DecisionTree, bar_width,
                 alpha=opacity,
                 color='b',
                 label='DecisionTree')
 
rects2 = pyplot.bar(index + bar_width, values_RandomForest, bar_width,
                 alpha=opacity,
                 color='g',
                 label='RandomForest')
 
pyplot.xlabel('Scores')
pyplot.ylabel('Values')
pyplot.title('Scores by Classificer')
pyplot.xticks(index + bar_width, ('Accuracy', 'Precision', 'Recall', 'F1 Score'))
ax.axhline(0.3, color="red", label='Performance Required')
pyplot.legend()
pyplot.tight_layout()
pyplot.show()

#Preparing the chosen classifier...

#Using the 5 more important features accordling to DecisionTree
features_list = ['poi', 'salary', 'to_messages', 'deferral_payments', 'total_payments', 'exercised_stock_options']

data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)

features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size = 0.3, random_state =42)

#DecidionTree
clf = DecisionTreeClassifier()
t0 = time()
clf.fit(features_train, labels_train)
print "training_time:", round(time()-t0, 3), "s"
t0 = time()
pred = clf.predict(features_test)
print "prediction time:", round(time()-t0, 3), "s"
test_classifier(clf, my_dataset, features_list)


### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

dump_classifier_and_data(clf, my_dataset, features_list)
