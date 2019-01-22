# -*- coding: utf-8 -*-
"""
Created on Tue Dec 25 17:46:37 2018

@author: HP
"""

import urllib.request

def read_text():
    quotes = open(r"C:\Users\HP\Desktop\movie_quotes.txt")
    contents_of_file = quotes.read()
    print(contents_of_file)
    quotes.close()
    check_words(contents_of_file)

def check_words(text_to_check):
    url = ("http://www.wdylike.appspot.com/?q=" + text_to_check)
    with urllib.request.urlopen(url) as response:
        html = response.read()
    print(html)

read_text()
    