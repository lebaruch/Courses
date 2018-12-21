#!/usr/bin/python


"""
    Starter code for the evaluation mini-project.
    Start by copying your trained/tested POI identifier from
    that which you built in the validation mini-project.

    This is the second step toward building your POI identifier!

    Start by loading/formatting the data...
"""

import pickle
import sys
sys.path.append("../tools/")
from feature_format import featureFormat, targetFeatureSplit

data_dict = pickle.load(open("../final_project/final_project_dataset.pkl", "r") )

### add more features to features_list!
features_list = ["poi", "salary"]

data = featureFormat(data_dict, features_list)
labels, features = targetFeatureSplit(data)



### your code goes here 
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split

clf = DecisionTreeClassifier()
clf.fit(features, labels)
print clf.score(features, labels)

# Spliting the dataset in training and test (70/30) with random_state = 42
features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size=0.3, random_state=42)

clf2 = DecisionTreeClassifier()
clf2.fit(features_train, labels_train)
print clf2.score(features_test, labels_test)

# How many POIs are in the test set for your POI identifier?
pred2 = clf2.predict(features_test)
print "Ammount of POIs in our test set:", sum(pred2)

# How many people total are in your test set?

print "Ammount of people in our test set:", len(pred2)

# What's the accuracy of the model if it predicted all not POI? True positive / (True positive + false negative)

print "Accuracy of the model for not POI:", (25.0 / (25.0 + 4.0))

# Precision and recall can help illuminate your performance better. 
# Use the precision_score and recall_score available in sklearn.metrics to compute those quantities.

# What’s the precision?
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score

print "Precision:", precision_score(labels_test, pred2)

# What’s the recall? 
print "Recall:", recall_score(labels_test, pred2)

# Here are some made-up predictions and true labels for a hypothetical test set; 
# fill in the following boxes to practice identifying true positives, false positives, true negatives, and false negatives. 
# Let’s use the convention that “1” signifies a positive result, and “0” a negative. 
predictions = [0, 1, 1, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 1, 1, 0, 1, 0, 1] 
true_labels = [0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 1, 0, 1, 1, 1, 0, 1, 0, 0]

# How many true positives?
# Answer: 6 (6 times 1 is on predictions and trues_labels with the same index)

# How many true negatives?
# Answer: 9 (9 times 1 is on predictions)

# How many false positives?
# Answer: 3 (3 times prediction has 1 and true_labels has 0, considering the same index)

# How many false negatives?
# Answer: 2 (2 times true_labels has 1 and predictions has 0, considering the same index)


# What's the precision of this classifier?
print "Precision of this classifier:", precision_score(true_labels, predictions)


# What's the recall of this classifier?
print "Recall of this classifier:", recall_score(true_labels, predictions)