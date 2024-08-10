# -*- coding: utf-8 -*-
"""
Created on Sat Oct  1 11:59:32 2022

@author: Roya Arian, royaarian101@gmail.com
"""
import numpy as np
def preparing(x, y):
    """
    This function prepares the data when the input is a dictionary and the output is an numpy array 
    the data is is primarily a dictionary in which each key of the dictionary refers to one subject
    because we want to split the data based on subjects.
    but for training the modles we need numpy array
    
    """
    
    data  = []
    label = []
    for i in x:
        for j in range(len(x[i])):
            data.append(x[i][j])
            label.append(y[i])
    
    data = np.reshape(data, np.shape(data))    
    return data, label