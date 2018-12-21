# -*- coding: utf-8 -*-
"""
Created on Thu Dec 20 17:44:54 2018

@author: baruch
"""


def rename_files():
    import os
    file_list = os.listdir(r"D:\DataScience\Courses\Udacity\Programming Foundations with Python\prank") #Creat list with all file_names from the path
    saved_path = os.getcwd() # current path
    os.chdir(r"D:\DataScience\Courses\Udacity\Programming Foundations with Python\prank") #change to path we need to get the job done
    table = str.maketrans(dict.fromkeys('0123456789'))
    for f in file_list:
        os.rename(f, f.translate(table))
    os.chdir(saved_path) # go back to current directory
    
      
rename_files()
