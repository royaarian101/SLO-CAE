# -*- coding: utf-8 -*-
"""
Created on Mon Jan 10 10:55:05 2022

MLP SLO classification when the features are extracted using proposed CAE 

@author:Roya Arian, royaarian101@gmail.com

"""

#Importing the necessary packages and libaries
import numpy as np
import pickle
from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import keras
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
import tensorflow as tf
from keras.utils import to_categorical
from sklearn import metrics
import models
import metrics_losses
import utils


#####################################################################
## Initializing Parameters
#####################################################################
#### model parameters
learning_rate = 1e-3
epoch = 200
batch_size = 32

number_class = 2  # please choose the number of classes according to your project
nfold = 5   # please choose number of folds in k fold cross validation algorithm

# initial some parameters
d = utils.Initialize(number_class, nfold, classifier="MLP")

############################################################
# main coode
############################################################

# loading data
# load the pickle file of the features of the test and train data extracted by CAE_bottel neck in each fold for classification purpose 
x_train_latent = pickle.load(open("x_train_latent.pkl", 'rb'))
y_train_latent = pickle.load(open("y_train_latent.pkl", 'rb'))

x_valid_latent = pickle.load(open("x_test_latent.pkl", 'rb'))
y_valid_latent = pickle.load(open("y_test_latent.pkl", 'rb'))

####################################################################
# Applying classification for each fold
####################################################################
n = 0

for i in range(5):
    n = n+1
    # print(train_index, val_index)  # you can watch train and validation index using this comment 
    print('%dth fold' % n)
    
    x_train = x_train_latent[n]
    x_valid = x_valid_latent[n]

    y_train = y_train_latent[n]
    y_valid = y_valid_latent[n]  

    input_img = len(x_train)
    
    model = models.mlp_func(input_img, n_class=number_class)
    METRICS = [
    keras.metrics.BinaryAccuracy(name='accuracy'),
    keras.metrics.AUC(name='auc'),
    ]
    
    my_optimizer = tf.keras.optimizers.Adam(lr=learning_rate)
    model.compile(optimizer=my_optimizer, loss='binary_crossentropy', metrics=METRICS)
    callbacks = [EarlyStopping(patience=50, verbose=1, restore_best_weights=True),
        ReduceLROnPlateau(factor=0.1, patience=10, min_lr=1e-5),
        ModelCheckpoint(f'mlp{n}.h5', verbose=1, save_best_only=True, save_weights_only=True, monitor='val_loss', mode='min')
    ]
       
    ### one Hot Encoding
    label_ohc_train = to_categorical(y_train)

    label_ohc_valid = to_categorical(y_valid)

    model.fit(x_train, label_ohc_train, batch_size=batch_size, epochs=epoch, callbacks=callbacks,\
                    validation_data=(x_valid, label_ohc_valid))
    
    
    model.load_weights(f'mlp{n}.h5')

    pred_class = model.predict(x_valid)
    y_pred     = [np.argmax(y, axis=None, out=None) for y in pred_class]
    pred_proba = pred_class[:,1]
    
    #######################################################################
    # Calculating metrics
    #######################################################################
    # retrieve the accuracy and print it
    d['acc_mlp'] = metrics.accuracy_score(y_valid, y_pred)
    
    d['sp_mlp'], d['se_mlp'], d['pr_mlp'], d['f1_mlp'], d['auc_mlp'], d['pr_auc_mlp'], class_acc_mlp, cm_mlp =\
        metrics_losses.metrics_calculation(y_valid, y_pred, pred_proba)

    #################### acc for each class ##################    
    d['class_acc_mlp']  = np.add(d['class_acc_mlp'], class_acc_mlp)

    ###################### Total confusion_matrix for poly kernel ############
    d['confusion_matrix'] = np.add(d['confusion_matrix'],cm_mlp)

    print(f'acc = {metrics.accuracy_score(y_valid, y_pred)}')
########################################
#     ploting confusion matrix
########################################
disp = ConfusionMatrixDisplay(confusion_matrix=d['confusion_matrix']/nfold, display_labels=d['target_names'])
disp.plot()

########################################
#     Metrics printing
########################################
utils.printing(d, classifier="MLP")

