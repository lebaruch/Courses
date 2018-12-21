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
    for i in my_dataset:
        # assign NaN to 0
        if my_dataset[i][key1] == 'NaN' or my_dataset[i][key2] == 'NaN':
            new_list.append(0.)
        elif my_dataset[i][key1]>=0:
            new_list.append(float(my_dataset[i][key1]) + float(my_dataset[i][key2]))
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
#my_dataset = data_dict
count = 0
for i in my_dataset:
    my_dataset[i]['fraction_poi_emails'] = fraction_poi_emails[count]
    count += 1

# test
print 'LAY KENNETH:', my_dataset['LAY KENNETH L']['fraction_poi_emails']

# let's test if this feature has any correlation with POIs
new_features_list = ['poi', 'fraction_poi_emails']
data = featureFormat(my_dataset, new_features_list)

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

for i in my_dataset:
    if my_dataset[i]['poi'] == 1:
        poi_fraction.append(my_dataset[i]['fraction_poi_emails'])
    else:
        not_poi_fraction.append(my_dataset[i]['fraction_poi_emails'])
        
print "Poi fraction:", np.mean(poi_fraction)
print "Not Poi fraction:", np.mean(not_poi_fraction)
    


### Extract features and labels from dataset for local testing
features_list = ['poi', 'salary', 'to_messages', 'deferral_payments', 'total_payments', 'exercised_stock_options', 'bonus', 
                 'restricted_stock', 'shared_receipt_with_poi', 'restricted_stock_deferred', 'total_stock_value', 'expenses',
                 'loan_advances', 'from_messages', 'other', 'from_this_person_to_poi', 'director_fees', 'deferred_income',
                 'long_term_incentive', 'from_poi_to_this_person'] #removed e-mail address

data = featureFormat(data_dict, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)

### Task 4: Try a varity of classifiers

# Provided to give you a starting point. Try a variety of classifiers.
# Trying Naive Bayes first
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score


#split data into train and test
features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size = 0.3, random_state =42)

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
print "Accuracy with SVM * feat: ", accuracy_score(labels_test, pred)
print "Precision with SVM * feat: ", precision_score(labels_test, pred)
print "Recall with SVM * feat: ", recall_score(labels_test, pred)
print "F1 Score with SVM * feat: ", f1_score(labels_test, pred)


#Naive Bayes
nb = GaussianNB()
t0 = time()
nb.fit(features_train, labels_train)
print "training_time:", round(time()-t0, 3), "s"
t0 = time()
pred = nb.predict(features_test)
print "prediction time:", round(time()-t0, 3), "s"
print "Accuracy with Naive Bayes * feat: ", accuracy_score(labels_test, pred)
print "Precision with Naive Bayes * feat: ", precision_score(labels_test, pred)
print "Recall with Naive Bayes * feat: ", recall_score(labels_test, pred)
print "F1 Score with Naive Bayes * feat: ", f1_score(labels_test, pred)


#DecidionTree
dtc = DecisionTreeClassifier()
t0 = time()
dtc.fit(features_train, labels_train)
print "training_time:", round(time()-t0, 3), "s"
t0 = time()
pred = dtc.predict(features_test)
print "prediction time:", round(time()-t0, 3), "s"
print "Accuracy with Decision Tree * feat: ", accuracy_score(labels_test, pred)
print "Precision with Decision Tree * feat: ", precision_score(labels_test, pred)
print "Recall with Decision Tree * feat: ", recall_score(labels_test, pred)
print "F1 Score with Decision Tree * feat: ", f1_score(labels_test, pred)
importances = dtc.feature_importances_
indices = np.argsort(importances)[::-1]
print 'Feature Ranking: '
for i in range(7):
    print "{} feature {} ({})".format(i+1,features_list[i+1],importances[indices[i]])


#RandomForest
rfc = RandomForestClassifier()
t0 = time()
rfc.fit(features_train, labels_train)
print "training time:", round(time()-t0, 3), "s"
t0 = time()
pred = rfc.predict(features_test)
print "predicting time:", round(time()-t0, 3), "s"
print "Accuracy with RandomForest * feat: ", accuracy_score(labels_test, pred)
print "Precision with RandomForest * feat: ", precision_score(labels_test, pred)
print "Recall with RandomForest * feat: ", recall_score(labels_test, pred)
print "F1 Score with RandomForest * feat: ", f1_score(labels_test, pred)



#Using the 5 more important features accordling to DecisionTree
five_top_features_list = ['salary', 'to_messages', 'deferral_payments', 'total_payments', 'exercised_stock_options']

data = featureFormat(data_dict, five_top_features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)


#SVM
svm = LinearSVC()
t0 = time()
svm.fit(features_train, labels_train)
print "training_time:", round(time()-t0, 3), "s"
t0 = time()
pred = svm.predict(features_test)
print "prediction time:", round(time()-t0, 3), "s"
print "Accuracy with SVM top5 feat: ", accuracy_score(labels_test, pred)
print "Precision with SVM top 5 feat: ", precision_score(labels_test, pred)
print "Recall with SVM top 5 feat: ", recall_score(labels_test, pred)
print "F1 Score with SVM top 5 feat: ", f1_score(labels_test, pred)


#Naive Bayes
nb = GaussianNB()
t0 = time()
nb.fit(features_train, labels_train)
print "training_time:", round(time()-t0, 3), "s"
t0 = time()
pred = nb.predict(features_test)
print "prediction time:", round(time()-t0, 3), "s"
print "Accuracy with Naive Bayes top 5 feat: ", accuracy_score(labels_test, pred)
print "Precision with Naive Bayes top 5 feat: ", precision_score(labels_test, pred)
print "Recall with Naive Bayes top 5 feat: ", recall_score(labels_test, pred)
print "F1 Score with Naive Bayes top 5 feat: ", f1_score(labels_test, pred)


#DecidionTree
dtc = DecisionTreeClassifier()
t0 = time()
dtc.fit(features_train, labels_train)
print "training_time:", round(time()-t0, 3), "s"
t0 = time()
pred = dtc.predict(features_test)
print "prediction time:", round(time()-t0, 3), "s"
print "Accuracy with Decision Tree top 5 feat: ", accuracy_score(labels_test, pred)
print "Precision with Decision Tree top 5 feat: ", precision_score(labels_test, pred)
print "Recall with Decision Tree top 5 feat: ", recall_score(labels_test, pred)
print "F1 Score with Decision Tree top 5 feat: ", f1_score(labels_test, pred)
importances = dtc.feature_importances_
indices = np.argsort(importances)[::-1]
print 'Feature Ranking: '
for i in range(7):
    print "{} feature {} ({})".format(i+1,features_list[i+1],importances[indices[i]])


#RandomForest
rfc = RandomForestClassifier()
t0 = time()
rfc.fit(features_train, labels_train)
print "training time:", round(time()-t0, 3), "s"
t0 = time()
pred = rfc.predict(features_test)
print "predicting time:", round(time()-t0, 3), "s"
print "Accuracy with RandomForest top 5 feat: ", accuracy_score(labels_test, pred)
print "Precision with RandomForest top 5 feat: ", precision_score(labels_test, pred)
print "Recall with RandomForest top 5 feat: ", recall_score(labels_test, pred)
print "F1 Score with RandomForest top 5 feat: ", f1_score(labels_test, pred)




#Trying with new features

new_features_list = ['poi', 'fraction_poi_emails', 'salary']

data = featureFormat(my_dataset, new_features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)

#SVM
svm = LinearSVC()
t0 = time()
svm.fit(features_train, labels_train)
print "training_time:", round(time()-t0, 3), "s"
t0 = time()
pred = svm.predict(features_test)
print "prediction time:", round(time()-t0, 3), "s"
print "Accuracy with SVM new feat: ", accuracy_score(labels_test, pred)
print "Precision with SVM new feat: ", precision_score(labels_test, pred)
print "Recall with SVM new feat: ", recall_score(labels_test, pred)
print "F1 Score with SVM new feat: ", f1_score(labels_test, pred)


#Naive Bayes
nb = GaussianNB()
t0 = time()
nb.fit(features_train, labels_train)
print "training_time:", round(time()-t0, 3), "s"
t0 = time()
pred = nb.predict(features_test)
print "prediction time:", round(time()-t0, 3), "s"
print "Accuracy with Naive Bayes new feat: ", accuracy_score(labels_test, pred)
print "Precision with Naive Bayes new feat: ", precision_score(labels_test, pred)
print "Recall with Naive Bayes new feat: ", recall_score(labels_test, pred)
print "F1 Score with Naive Bayes new feat: ", f1_score(labels_test, pred)


#DecidionTree
dtc = DecisionTreeClassifier()
t0 = time()
dtc.fit(features_train, labels_train)
print "training_time:", round(time()-t0, 3), "s"
t0 = time()
pred = dtc.predict(features_test)
print "prediction time:", round(time()-t0, 3), "s"
print "Accuracy with Decision Tree new feat: ", accuracy_score(labels_test, pred)
print "Precision with Decision Tree new feat: ", precision_score(labels_test, pred)
print "Recall with Decision Tree new feat: ", recall_score(labels_test, pred)
print "F1 Score with Decision Tree new feat: ", f1_score(labels_test, pred)


#RandomForest
rfc = RandomForestClassifier()
t0 = time()
rfc.fit(features_train, labels_train)
print "training time:", round(time()-t0, 3), "s"
t0 = time()
pred = rfc.predict(features_test)
print "predicting time:", round(time()-t0, 3), "s"
print "Accuracy with RandomForest new feat: ", accuracy_score(labels_test, pred)
print "Precision with RandomForest new feat: ", precision_score(labels_test, pred)
print "Recall with RandomForest new feat: ", recall_score(labels_test, pred)
print "F1 Score with RandomForest new feat: ", f1_score(labels_test, pred)




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
features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size = 0.3, random_state =42)

# Creating selector
selector = SelectKBest(k=7)
selectedFeatures = selector.fit(features,labels)
feature_names = [features_list[i] for i in selectedFeatures.get_support(indices=True)]
print 'Selector: ', feature_names
# ['poi', 'total_payments', 'exercised_stock_options', 'bonus', 'restricted_stock_deferred', 'director_fees', 'deferred_income']


#SVM
svm = LinearSVC()
t0 = time()
svm.fit(features_train, labels_train)
print "training_time:", round(time()-t0, 3), "s"
t0 = time()
pred = svm.predict(features_test)
print "prediction time:", round(time()-t0, 3), "s"
print "Accuracy with SVM with SelectKBest: ", accuracy_score(labels_test, pred)
print "Precision with SVM with SelectKBest: ", precision_score(labels_test, pred)
print "Recall with SVM with SelectKBest: ", recall_score(labels_test, pred)
print "F1 Score with SVM with SelectKBest: ", f1_score(labels_test, pred)


#Naive Bayes
nb = GaussianNB()
t0 = time()
nb.fit(features_train, labels_train)
print "training_time:", round(time()-t0, 3), "s"
t0 = time()
pred = nb.predict(features_test)
print "prediction time:", round(time()-t0, 3), "s"
print "Accuracy with Naive Bayes with SelectKBest: ", accuracy_score(labels_test, pred)
print "Precision with Naive Bayes with SelectKBest: ", precision_score(labels_test, pred)
print "Recall with Naive Bayes with SelectKBest: ", recall_score(labels_test, pred)
print "F1 Score with Naive Bayes with SelectKBest: ", f1_score(labels_test, pred)


#DecidionTree
dtc = DecisionTreeClassifier()
t0 = time()
dtc.fit(features_train, labels_train)
print "training_time:", round(time()-t0, 3), "s"
t0 = time()
pred = dtc.predict(features_test)
print "prediction time:", round(time()-t0, 3), "s"
print "Accuracy with Decision Tree with SelectKBest: ", accuracy_score(labels_test, pred)
print "Precision with Decision Tree with SelectKBest: ", precision_score(labels_test, pred)
print "Recall with Decision Tree with SelectKBest: ", recall_score(labels_test, pred)
print "F1 Score with Decision Tree with SelectKBest: ", f1_score(labels_test, pred)


#RandomForest
rfc = RandomForestClassifier()
t0 = time()
rfc.fit(features_train, labels_train)
print "training time:", round(time()-t0, 3), "s"
t0 = time()
pred = rfc.predict(features_test)
print "predicting time:", round(time()-t0, 3), "s"
print "Accuracy with RandomForest with SelectKBest: ", accuracy_score(labels_test, pred)
print "Precision with RandomForest with SelectKBest: ", precision_score(labels_test, pred)
print "Recall with RandomForest with SelectKBest: ", recall_score(labels_test, pred)
print "F1 Score with RandomForest with SelectKBest: ", f1_score(labels_test, pred)



#Tunning RandomForest and DecisionTree with GridSeachCV
# I will use the top 5 features
five_top_features_list = ['salary', 'to_messages', 'deferral_payments', 'total_payments', 'exercised_stock_options']

data = featureFormat(data_dict, five_top_features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)

#RandomForest
t0 = time()
param_grid = {
             'n_estimators':[10, 100, 1000, 5000],
                 'max_features':[1, 2, 3, 4, 5],
                     'criterion':['gini', 'entropy']
            }
rfc = GridSearchCV(RandomForestClassifier(), param_grid)
rfc = rfc.fit(features_train, labels_train)
print "done in %0.3fs" % (time() - t0)
print "Best estimator:"
print rfc.best_estimator_

#Testing with Best parameters estimated
rfc = RandomForestClassifier(n_estimators = 10, criterion = 'gini', max_features = 3)
t0 = time()
rfc.fit(features_train, labels_train)
print "training time:", round(time()-t0, 3), "s"
t0 = time()
pred = rfc.predict(features_test)
print "predicting time:", round(time()-t0, 3), "s"
print "Accuracy with RandomForest with adjusted parameters: ", accuracy_score(labels_test, pred)
print "Precision with RandomForest with adjusted parameters: ", precision_score(labels_test, pred)
print "Recall with RandomForest with adjusted parameters: ", recall_score(labels_test, pred)
print "F1 Score with RandomForest with adjusted parameters: ", f1_score(labels_test, pred)
#bad result :()


#Let's try the same with DecisionTree
#DecisionTree
t0 = time()
param_grid = {
             'min_samples_split': [2, 3, 4, 5, 6, 7, 8],
                 'max_depth': [1, 2, 3, 4, 5, 6, 7, 8],
                     'max_features': range(1,5)
            }
dtc = GridSearchCV(DecisionTreeClassifier(), param_grid)
dtc = dtc.fit(features_train, labels_train)
print "done in %0.3fs" % (time() - t0)
print "Best estimator:"
print dtc.best_estimator_

dtc = DecisionTreeClassifier(min_samples_split = 2, max_depth = 1, max_features = 1)
t0 = time()
dtc.fit(features_train, labels_train)
print "training_time:", round(time()-t0, 3), "s"
t0 = time()
pred = dtc.predict(features_test)
print "prediction time:", round(time()-t0, 3), "s"
print "Accuracy with Decision Tree with adjusted parameters: ", accuracy_score(labels_test, pred)
print "Precision with Decision Tree with adjusted parameters: ", precision_score(labels_test, pred)
print "Recall with Decision Tree with adjusted parameters: ", recall_score(labels_test, pred)
print "F1 Score with Decision Tree with adjusted parameters: ", f1_score(labels_test, pred)


#Comparing both classificers best result:
# data to plot
n_groups = 4
values_RandomForest = (0.91, 0.67, 0.4, 0.5)
values_DecisionTree = (0.84, 0.33, 0.4, 0.36)
 
# create plot
fig, ax = pyplot.subplots()
index = np.arange(n_groups)
bar_width = 0.35
opacity = 0.8
 
rects1 = pyplot.bar(index, values_RandomForest, bar_width,
                 alpha=opacity,
                 color='b',
                 label='RandomForest')
 
rects2 = pyplot.bar(index + bar_width, values_DecisionTree, bar_width,
                 alpha=opacity,
                 color='g',
                 label='DecisionTree')
 
pyplot.xlabel('Scores')
pyplot.ylabel('Values')
pyplot.title('Scores by Classificer')
pyplot.xticks(index + bar_width, ('Accuracy', 'Precision', 'Recall', 'F1 Score'))
ax.axhline(0.3, color="red")
pyplot.legend()
pyplot.tight_layout()
pyplot.show()

#Preparing the chosen classifier...
features_list = ['salary', 'to_messages', 'deferral_payments', 'total_payments', 'exercised_stock_options']
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)

clf = RandomForestClassifier()
t0 = time()
clf.fit(features_train, labels_train)
print "training time:", round(time()-t0, 3), "s"
t0 = time()
pred = clf.predict(features_test)
print "predicting time:", round(time()-t0, 3), "s"
print "Accuracy with RandomForest top 5 feat: ", accuracy_score(labels_test, pred)
print "Precision with RandomForest top 5 feat: ", precision_score(labels_test, pred)
print "Recall with RandomForest top 5 feat: ", recall_score(labels_test, pred)
print "F1 Score with RandomForest top 5 feat: ", f1_score(labels_test, pred)

#training time: 0.016 s
#predicting time: 0.003 s
#Accuracy with RandomForest top 5 feat:  0.9090909090909091
#Precision with RandomForest top 5 feat:  0.6666666666666666
#Recall with RandomForest top 5 feat:  0.4
#F1 Score with RandomForest top 5 feat:  0.5




### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

dump_classifier_and_data(clf, my_dataset, features_list)
