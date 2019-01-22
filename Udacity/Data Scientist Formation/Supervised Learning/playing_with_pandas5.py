# -*- coding: utf-8 -*-
"""
Created on Wed Jan 16 11:18:10 2019

@author: baruch
"""

import pandas as pd

# create a very simple dataframe
df = pd.DataFrame({'col 1':[1, 2, 3], 'col 2': [4, 5, 6]})

# add the third column
df['col 3'] = [7, 8, 9, 10]

# the end result
print(df)