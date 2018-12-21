# -*- coding: utf-8 -*-

import pickle
from sklearn.preprocessing import MinMaxScaler
import numpy as np

data_dict = pickle.load(open("D:/DataScience/Intro_Machine_Learning/final_project/final_project_dataset.pkl", "r"))

### there's an outlier--remove it!
data_dict.pop("TOTAL", 0)

# Apply feature scaling to your k-means clustering code from the last lesson,
# on the “salary” and “exercised_stock_options” features (use only these two features).
# What would be the rescaled value of a "salary" feature that had an original value of $200,000,
# and an "exercised_stock_options" feature of $1 million? (Be sure to represent these numbers as floats, not integers!)

salary = []
ex_stok = []
for users in data_dict:
    val = data_dict[users]["salary"]
    if val == 'NaN':
        continue
    salary.append(float(val))
    val = data_dict[users]["exercised_stock_options"]
    if val == 'NaN':
        continue
    ex_stok.append(float(val))

salary = [min(salary),200000.0,max(salary)]
ex_stok = [min(ex_stok),1000000.0,max(ex_stok)]

print salary
print ex_stok

salary = np.array([[e] for e in salary])
ex_stok = np.array([[e] for e in ex_stok])

scaler_salary = MinMaxScaler()
scaler_stok = MinMaxScaler()

rescaled_salary = scaler_salary.fit_transform(salary)
rescaled_stock = scaler_salary.fit_transform(ex_stok)

print rescaled_salary
print rescaled_stock
