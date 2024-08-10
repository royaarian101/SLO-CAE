# -*- coding: utf-8 -*-
"""
Created on Wed Feb 19 11:59:24 2020

@author: Roya Arian
"""

""" Preprocessing
"""
import numpy as np
from keras.preprocessing.image import ImageDataGenerator


def Augmentation (x_train, labels_train):
    
    """
    This function Augments train images of each fold due to the limitation of data
    
    x_train: train images befor augmentation
    labels_train: train labels befor augmentation
    
    x: Augmented images
    labels: Augmented labels
    """

    # augmentation
    batch=np.zeros_like(x_train, dtype=np.float32)
    batch_label=np.zeros_like(labels_train, dtype=np.float32)

    datagen = ImageDataGenerator(
        rotation_range= 5, # rotation
        width_shift_range= [-3, 3], # horizontal shift
        height_shift_range= [-5, 5] , # vertical shift
        zoom_range= 0.2,
        vertical_flip= True , # vertical flip
        brightness_range= [0.2, 1.5],
        fill_mode='constant',
          )




    for i in range(len(x_train)):
        x1=x_train[i,:,:,:].copy()
        x1=x1.reshape((1, ) + x1.shape)
        x = datagen.flow(x1, batch_size=1, seed=2020) # to make the result reproducible
        
        
        batch[i,:,:,:] = x.next() 
        batch_label[i] = labels_train[i]

    ###################################################################
    # Final data
    ###################################################################
    
    x = np.concatenate([x_train,batch])
    
    labels = np.concatenate([labels_train,batch_label])
       
    ############################
        
    ############################
    return x, labels