#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 23 18:01:47 2020

@author: ananyabanerjee
"""

import runpy
import sys

def main():
    print("Welcome to my code")
    
    
    type_of_smoothing=sys.argv[1]
    if type_of_smoothing=="no_smoothing":
        print("No smoothing initiated")
        runpy.run_path('no_smoothing.py')
    
    elif type_of_smoothing=="add_one_smoothing":
        print("Add one smoothing initiated")
        runpy.run_path('add_one_smoothing.py')
    
    else:
        print("Good Turing Discounting initiated")
        runpy.run_path('good_turing_discounting.py')
    
    
if __name__=="__main__":
    main()


