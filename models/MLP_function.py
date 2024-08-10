# -*- coding: utf-8 -*-
"""
Created on Tue Jun 20 08:59:12 2023

proposed mlp_function 

@author: Roya Arian, royaarian101@gmail.com
"""

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, BatchNormalization, Dropout    

def mlp_func(input_img, n_class):
        
    # Creating model
    model = Sequential()
 
    ##################################
    model.add(Dense(1000, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.1,seed = 2019))

    model.add(Dense(500, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.4,seed = 2019))  

    model.add(Dense(250, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.4,seed = 2019))  
    
    model.add(Dense(100, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.4,seed = 2019))
    
    model.add(Dense(50, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.4,seed = 2019))
    
    model.add(Dense(10, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.1,seed = 2019))
    
    model.add(Dense(n_class, activation='softmax'))
    
    
    return model

