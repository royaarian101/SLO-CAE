# -*- coding: utf-8 -*-
"""
Created on Wed Aug  2 11:19:12 2023

main code: Feature extraction using Convolutional AutoEncoder (CAE) and SVM classification with different Kernels

The Hyper parameters of the proposed CAE were optimized to get the highest possible classification accuracies by the SVM 

@author: Roya Arian, royaarian101@gmail.com

"""

import pickle
import keras
import numpy as np
import matplotlib.pyplot as plt
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras import optimizers
from sklearn import svm
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.model_selection import StratifiedKFold
import preprocess
import models
import metrics_losses
import plotting_curves
import utils

#####################################################################
## Initializing Parameters
#####################################################################
#### model parameters
epochs        = 100
batch_size    = 16
nfilters      = 2
learning_rate = 1e-5
dropout       = 0.1


number_class = 2  # please choose the number of classes according to your project
nfold = 5   # please choose number of folds in k fold cross validation algorithm

# initial some parameters
d = utils.Initialize(number_class, nfold, classifier="SVM")


############################################################
# main coode
############################################################

###### loading data
# load the data and the coresponding lables pickle file  
images_train = pickle.load(open("subjects_slo_data.pkl", 'rb'))  
labels_train = pickle.load(open("labels_slo_data.pkl", 'rb'))

####################################################################
# Applying kfold
####################################################################

kf_nfold = StratifiedKFold(n_splits=nfold, random_state=42, shuffle=True)

n = 0
for train_index, val_index in kf_nfold.split(images_train,list(labels_train.values())):
    n = n+1
    # print(train_index, val_index)  # you can watch train and validation index using this comment
    print(f'---------------------------------------------------------------------\
          \n \t\t\t  fold_{n} \n---------------------------------------------------------------------'\
          ,end = '\n\n\n' )
    x_train = {i: images_train[i] for i in train_index}
    x_valid = {i: images_train[i] for i in val_index}

    y_train = {i: labels_train[i] for i in train_index}
    y_valid = {i: labels_train[i] for i in val_index}


    ################## preparing
    x_train, y_train = preprocess.preparing(x_train,y_train)
    x_valid, y_valid = preprocess.preparing(x_valid,y_valid)

    x_train /= 255
    x_valid /= 255

    ################# Augmentation
    x_train, y_train = preprocess.Augmentation(x_train,y_train)

    channel = np.shape(x_train)[3]
    #################################
    ###### Applying CAE  ###########
    #################################
    input_img = keras.Input(shape=(np.shape(x_train)[1], np.shape(x_train)[2], channel))

# use tensorflow in colab befor 
    autoencoder, encoder = models.CAE_feature_extractor(input_img, channel=channel, n_filters=nfilters)


    my_optimizer = optimizers.Adam(lr=learning_rate)
    autoencoder.compile(optimizer=my_optimizer, loss=metrics_losses.loss_functions.huber_fn)

    callbacks = [EarlyStopping(patience=10, verbose=1),
        ReduceLROnPlateau(factor=0.1, patience=10, min_lr=1e-9, verbose=1),
        ModelCheckpoint(f'CAuto_{n}.h5', verbose=1, save_best_only=True, save_weights_only=True)
    ]

    # autoencoder.summary()

    results = autoencoder.fit(x_train, x_train,
                    epochs=epochs,
                    batch_size=batch_size,
                    shuffle=True,
                    validation_data=(x_valid, x_valid),
                    callbacks=callbacks)


    plt.figure(figsize=(5, 5))
    plt.title(f"Learning curve {n}th fold")
    plt.plot(results.history["loss"], label="loss")
    plt.plot(results.history["val_loss"], label="val_loss")
    plt.plot( np.argmin(results.history["val_loss"]), np.min(results.history["val_loss"]), marker="x", color="r", label="best model")
    plt.xlabel("Epochs")
    plt.ylabel("log_loss")
    plt.legend()


    # load the best model
    autoencoder.load_weights(f'CAuto_{n}.h5')

    encoded_img = autoencoder.predict(x_valid)

    # mean square error calculation of pred and original test images
    error = metrics_losses.loss_functions.rmse(encoded_img, x_valid)
    print(f'The rmse is : {error}')

    # feature selection
    test_latent  = encoder.predict(x_valid)
    train_latent = encoder.predict(x_train)

    # flatten matrix
    latent_train = np.zeros((len(train_latent), np.size(train_latent[0])))
    for train in range (len(train_latent)):
        latent_train[train,:] = (train_latent[train])[:,:,:].flatten()

    latent_test = np.zeros((len(test_latent), np.size(test_latent[0])))
    for test in range (len(test_latent)):
        latent_test[test,:] = (test_latent[test])[:,:,:].flatten()


    # saving the features extracted by CAE_bottel neck in each fold for classification purpose 
    d['x_test_latent'][n] = latent_test
    d['y_test_latent'][n] = y_valid
    d['x_train_latent'][n] = latent_train
    d['y_train_latent'][n] = y_train
    ################################
    #### Applying SVM for classification ############
    ################################

    print('linear SVM')
    linear = svm.SVC(kernel='linear', C=1, decision_function_shape='ovo', probability=True).fit(latent_train, y_train)
    print('rbf SVM')
    rbf = svm.SVC(kernel='rbf', gamma=1e-4, C=10, decision_function_shape='ovo', probability=True).fit(latent_train, y_train)
    print('Poly SVM')
    poly = svm.SVC(kernel='poly', degree=3, C=10, decision_function_shape='ovo', probability=True).fit(latent_train, y_train)
    print('sigmoid SVM')
    sig = svm.SVC(kernel='sigmoid', C=1, decision_function_shape='ovo', probability=True).fit(latent_train, y_train)

    linear_pred_val = linear.predict(latent_test)
    poly_pred_val   = poly.predict(latent_test)
    rbf_pred_val    = rbf.predict(latent_test)
    sig_pred_val    = sig.predict(latent_test)


    linear_proba = linear.predict_proba(latent_test)[:,1]
    poly_proba   = poly.predict_proba(latent_test)[:,1]
    rbf_proba    = rbf.predict_proba(latent_test)[:,1]
    sig_proba    = sig.predict_proba(latent_test)[:,1]

    #######################################################################
    # Calculating metrics
    #######################################################################
    # retrieve the accuracy and print it for all 4 kernel functions
    d['acc_lin'].append(linear.score(latent_test, y_valid))
    d['acc_poly'].append(poly.score(latent_test, y_valid))
    d['acc_rbf'].append(rbf.score(latent_test, y_valid))
    d['acc_sig'].append(sig.score(latent_test, y_valid))

    d['sp_lin'][n-1], d['se_lin'][n-1], d['pr_lin'][n-1], d['f1_lin'][n-1], d['auc_lin'][n-1], d['pr_auc_lin'][n-1], class_acc_l, cm_l =\
        metrics_losses.metrics_calculation(y_valid, linear_pred_val, linear_proba)
    d['sp_poly'][n-1],  d['se_poly'][n-1], d['pr_poly'][n-1], d['f1_poly'][n-1], d['auc_poly'][n-1], d['pr_auc_poly'][n-1], class_acc_p, cm_p =\
        metrics_losses.metrics_calculation(y_valid, poly_pred_val, poly_proba)
    d['sp_rbf'][n-1], d['se_rbf'][n-1], d['pr_rbf'][n-1], d['f1_rbf'][n-1], d['auc_rbf'][n-1], d['pr_auc_rbf'][n-1], class_acc_r, cm_r =\
        metrics_losses.metrics_calculation(y_valid, rbf_pred_val, rbf_proba)
    d['sp_sig'][n-1], d['se_sig'][n-1], d['pr_sig'][n-1], d['f1_sig'][n-1], d['auc_sig'][n-1], d['pr_auc_sig'][n-1], class_acc_s, cm_s =\
        metrics_losses.metrics_calculation(y_valid, sig_pred_val, sig_proba)


    #################### acc for each class ##################
    d['class_acc_lin']  = np.add(d['class_acc_lin'],class_acc_l)
    d['class_acc_rbf']  = np.add(d['class_acc_rbf'],class_acc_r)
    d['class_acc_poly'] = np.add(d['class_acc_poly'],class_acc_p)
    d['class_acc_sig']  = np.add(d['class_acc_sig'],class_acc_s)

    print(f'linear  accuracy of {n}th fold : {linear.score(latent_test, y_valid)}')
    print(f'Poly    accuracy of {n}th fold : {poly.score(latent_test, y_valid)}')
    print(f'RBF     accuracy of {n}th fold : {rbf.score(latent_test, y_valid)}')
    print(f'sigmoid accuracy of {n}th fold : {sig.score(latent_test, y_valid)}')

    ###################### Ploting ROC curve for each fold and the mean ############
    d['y_test'] = np.append(d['y_test'], y_valid, axis = 0)

    ### for poly kernel
    d['tprs_p'], d['aucs_p'], d['y_pred_p'] = plotting_curves.fold_curves\
        (d['ax'], d['ax1'], poly, latent_test, y_valid, n, d['mean_fpr'], poly_proba, d['tprs_p'], d['aucs_p'], d['y_pred_p'])

    #### for linear kernel
    d['tprs_l'], d['aucs_l'], d['y_pred_l'] = plotting_curves.fold_curves\
        (d['ax2'], d['ax3'], linear, latent_test, y_valid, n, d['mean_fpr'], linear_proba, d['tprs_l'], d['aucs_l'], d['y_pred_l'])

    #### for rbf kernel
    d['tprs_r'], d['aucs_r'], d['y_pred_r'] = plotting_curves.fold_curves\
        (d['ax4'], d['ax5'], rbf, latent_test, y_valid, n, d['mean_fpr'], rbf_proba, d['tprs_r'], d['aucs_r'], d['y_pred_r'])



###################### Continuing Ploting ROC and P_R curve for each fold and the mean ############
### for poly kernel
plotting_curves.curve_plotting(d['ax'], d['ax1'],  d['mean_fpr'],  d['aucs_p'],  d['tprs_p'],  d['y_test'],  d['y_pred_p'], 'SVM-CAE', 'Polynomial' )

### for linear kernel
plotting_curves.curve_plotting(d['ax2'],d['ax3'],  d['mean_fpr'],  d['aucs_l'],  d['tprs_l'],  d['y_test'],  d['y_pred_l'], 'SVM-CAE', 'Linear' )

### for rbf kernel
plotting_curves.curve_plotting(d['ax4'], d['ax5'],  d['mean_fpr'], d['aucs_r'],  d['tprs_r'],  d['y_test'],  d['y_pred_r'], 'SVM-CAE', 'RBF' )

# show the plot
plt.show()

########################################
#     ploting confusion matrix
########################################
disp = ConfusionMatrixDisplay(confusion_matrix=d['confusion_matrix']/nfold, display_labels=d['target_names'])
disp.plot()

########################################
#     Metrics printing
########################################
utils.printing(d, classifier="SVM")

########################################
# Saving the extracted features for claasification purpose
########################################
pickle.dump(d['x_test_latent'], open("x_test_latent.pkl", 'wb'))
pickle.dump(d['y_test_latent'], open("y_test_latent.pkl", 'wb'))

pickle.dump(d['x_train_latent'], open("x_train_latent.pkl", 'wb'))
pickle.dump(d['y_train_latent'], open("y_train_latent.pkl", 'wb'))


