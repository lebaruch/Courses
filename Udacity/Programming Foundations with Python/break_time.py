# -*- coding: utf-8 -*-
"""
Created on Thu Dec 20 16:57:31 2018

@author: baruch
"""

### Program that reminds to take a break from time to time ###

import webbrowser
import time

total_breaks = 3
break_count = 0

print(("This program started on: {} ").format(time.ctime())) # Print the time of this program started for the first time
while (break_count < total_breaks):
    time.sleep(108000)  # It will open the browser each 3 hours, 10800 seconds
    webbrowser.open("https://www.youtube.com/watch?v=6k8es2BNloE")
    break_count = break_count + 1