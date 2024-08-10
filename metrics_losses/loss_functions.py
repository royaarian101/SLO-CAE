# -*- coding: utf-8 -*-
"""
Created on Sat Aug  5 09:44:38 2023

loss functions

@author: Roya Arian, royaarian101@gmail.com
"""

import numpy as np
import tensorflow as tf

def huber_fn(y_true, y_pred):
    error = y_true - y_pred
    is_small_error = tf.abs(error) < 1
    squared_loss = tf.square(error) / 2
    linear_loss  = tf.abs(error) - 0.5
    return tf.where(is_small_error, squared_loss, linear_loss)



# RMSE function
def rmse(predictions, targets):
    return np.sqrt(((predictions - targets) ** 2).mean())

