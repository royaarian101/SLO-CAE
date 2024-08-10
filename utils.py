# -*- coding: utf-8 -*-
"""
Created on Sat Aug  5 09:31:37 2023


Utilized functions

@author:Roya Arian, royaarian101@gmail.com
"""


import numpy as np
import matplotlib.pyplot as plt

def Initialize(number_class, nfold, classifier):
    
    """
    This function Initializes some evaluation parameters for SVM classifier with different kernels
    as well as MLP classifier
    
    acc: Accuracy
    sp: Spesificity
    se: Sensitivity 
    pr: precision
    f1: f1-score
    auc: ROC AUC
    pr_auc: Precision-Recall AUC
    class_acc: acc of each class individually 
    tprs, aucs, mean_fpr : Necessary params to plot the ROC and PR curves
    
    x_test_latent: to store the features of the test data extracted by CAE_bottel neck in each fold for classification purpose 
    y_test_latent: to store lables of the test data in each fold for classification purpose 
    x_train_latent: to store the features of the train data extracted by CAE_bottel neck in each fold for classification purpose 
    y_train_latent: to store lables of train data in each fold for classification purpose 
    """
    
    d = dict()
    d['nfold'] = nfold
    d['target_names'] = ['Normal' , 'MS']  # classes based on lables (Normal = 0, MS = 1)

    if classifier =="SVM":
        d['fig'], d['ax']   = plt.subplots(figsize=(5, 5))
        d['fig1'], d['ax1'] = plt.subplots(figsize=(5, 5))
        d['fig2'], d['ax2'] = plt.subplots(figsize=(5, 5))
        d['fig3'], d['ax3'] = plt.subplots(figsize=(5, 5))
        d['fig4'], d['ax4'] = plt.subplots(figsize=(5, 5))
        d['fig5'], d['ax5'] = plt.subplots(figsize=(5, 5))
        
    
    
    ### Initializing
    
    d['acc_lin']  = []
    d['acc_rbf']  = []
    d['acc_poly'] = []
    d['acc_sig']  = []
    d['acc_mlp']  = []

    d['sp_lin']  = np.zeros((nfold))
    d['sp_rbf']  = np.zeros((nfold))
    d['sp_poly'] = np.zeros((nfold))
    d['sp_sig']  = np.zeros((nfold))
    d['sp_mlp']  = np.zeros((nfold))

    d['se_lin']  = np.zeros((nfold))
    d['se_rbf']  = np.zeros((nfold))
    d['se_poly'] = np.zeros((nfold))
    d['se_sig']  = np.zeros((nfold))
    d['se_mlp']  = np.zeros((nfold))

    d['pr_lin']  = np.zeros((nfold))
    d['pr_rbf']  = np.zeros((nfold))
    d['pr_poly'] = np.zeros((nfold))
    d['pr_sig']  = np.zeros((nfold))
    d['pr_mlp']  = np.zeros((nfold))

    d['f1_lin']  = np.zeros((nfold))
    d['f1_rbf']  = np.zeros((nfold))
    d['f1_poly'] = np.zeros((nfold))
    d['f1_sig']  = np.zeros((nfold))
    d['f1_mlp']  = np.zeros((nfold))

    d['auc_lin']  = np.zeros((nfold))
    d['auc_rbf']  = np.zeros((nfold))
    d['auc_poly'] = np.zeros((nfold))
    d['auc_sig']  = np.zeros((nfold))
    d['auc_mlp']  = np.zeros((nfold))

    d['pr_auc_lin']  = np.zeros((nfold))
    d['pr_auc_rbf']  = np.zeros((nfold))
    d['pr_auc_poly'] = np.zeros((nfold))
    d['pr_auc_sig']  = np.zeros((nfold))
    d['pr_auc_mlp']  = np.zeros((nfold))

    d['class_acc_lin']  = np.zeros((number_class))
    d['class_acc_rbf']  = np.zeros((number_class))
    d['class_acc_poly'] = np.zeros((number_class))
    d['class_acc_sig']  = np.zeros((number_class))
    d['class_acc_mlp']  = np.zeros((number_class))
    
    
    d['confusion_matrix'] = np.zeros((number_class, number_class))
    
    # r:rbf, l:linear, p:poly, m:mlp
    d['x_test']  = {}
    d['y_testt'] = {}
    d['y_test'] = []
    
    
    d['x_test_latent']  = {}
    d['y_test_latent']  = {}
    d['x_train_latent'] = {}
    d['y_train_latent'] = {}
    
    d['tprs_l'] = []
    d['aucs_l'] = []
    d['y_pred_l']=[]
    
    d['tprs_p'] = []
    d['aucs_p'] = []
    d['y_pred_p']=[]
    
    d['tprs_r'] = []
    d['aucs_r'] = []
    d['y_pred_r']=[]
    
    d['tprs_m'] = []
    d['aucs_m'] = []
    d['y_pred_m']=[]

    d['mean_fpr']  = np.linspace(0, 1, 100)
        
    return d




def printing(d,classifier):
    
    """
    This function prints some evaluation parameters for SVM classifier with different kernels
    as well as MLP classifier
    
    acc: Accuracy
    sp: Spesificity
    se: Sensitivity 
    pr: precision
    f1: f1-score
    auc: ROC AUC
    pr_auc: Precision-Recall AUC
    class_acc: acc of each class individually 
    """
    
    if classifier == 'SVM':
    
        print('acc_lin  = %f' % np.mean(d['acc_lin']))
        print('acc_rbf  = %f' % np.mean(d['acc_rbf']))
        print('acc_poly = %f' % np.mean(d['acc_poly']))
        print('acc_sig  = %f' % np.mean(d['acc_sig']), end='\n\n')
    
        print('sp_lin  = %f' % np.mean(d['sp_lin']))
        print('sp_rbf  = %f' % np.mean(d['sp_rbf']))
        print('sp_poly = %f' % np.mean(d['sp_poly']))
        print('sp_sig  = %f' % np.mean(d['sp_sig']), end='\n\n')
    
        print('se_lin  = %f' % np.mean(d['se_lin']))
        print('se_rbf  = %f' % np.mean(d['se_rbf']))
        print('se_poly = %f' % np.mean(d['se_poly']))
        print('se_sig  = %f' % np.mean(d['se_sig']), end='\n\n')
    
        print('pr_lin  = %f' % np.mean(d['pr_lin']))
        print('pr_rbf  = %f' % np.mean(d['pr_rbf']))
        print('pr_poly = %f' % np.mean(d['pr_poly']))
        print('pr_sig  = %f' % np.mean(d['pr_sig']), end='\n\n')
    
        print('f1_lin  = %f' % np.mean(d['f1_lin']))
        print('f1_rbf  = %f' % np.mean(d['f1_rbf']))
        print('f1_poly = %f' % np.mean(d['f1_poly']))
        print('f1_sig  = %f' % np.mean(d['f1_sig']), end='\n\n')
    
        print('ROC_auc_lin  = %f' % np.mean(d['auc_lin']))
        print('ROC_auc_rbf  = %f' % np.mean(d['auc_rbf']))
        print('ROC_auc_poly = %f' % np.mean(d['auc_poly']))
        print('ROC_auc_sig  = %f' % np.mean(d['auc_sig']), end='\n\n')
    
        print('P_R_auc_lin  = %f' % np.mean(d['pr_auc_lin']))
        print('P_R_auc_rbf  = %f' % np.mean(d['pr_auc_rbf']))
        print('P_R_auc_poly = %f' % np.mean(d['pr_auc_poly']))
        print('P_R_auc_sig  = %f' % np.mean(d['pr_auc_sig']), end='\n\n')
    
        print('acc of class using linear kernel %s' % d['target_names'][0], '= %f' % (d['class_acc_lin'][0]/d['nfold']))
        print('acc of class using linear kernel %s' % d['target_names'][1], '= %f' % (d['class_acc_lin'][1]/d['nfold']), end='\n\n')
        print('acc of class using RBF kernel %s' % d['target_names'][0], '= %f' % (d['class_acc_rbf'][0]/d['nfold']))
        print('acc of class using RBF kernel %s' % d['target_names'][1], '= %f' % (d['class_acc_rbf'][1]/d['nfold']), end='\n\n')
        print('acc of class using Poly kernel %s' % d['target_names'][0], '= %f' % (d['class_acc_poly'][0]/d['nfold']))
        print('acc of class using Poly kernel %s' % d['target_names'][1], '= %f' % (d['class_acc_poly'][1]/d['nfold']), end='\n\n')
        print('acc of class using Sigmoid kernel %s' % d['target_names'][0], '= %f' % (d['class_acc_sig'][0]/d['nfold']))
        print('acc of class using Sigmoid kernel %s' % d['target_names'][1], '= %f' % (d['class_acc_sig'][1]/d['nfold']), end='\n\n')

    elif classifier == 'MLP':

        print('acc_mlp  = %f' % np.mean(d['acc_mlp']))   
        print('sp_mlp  = %f' % np.mean(d['sp_mlp']))
        print('se_mlp  = %f' % np.mean(d['se_mlp']))
        print('pr_mlp  = %f' % np.mean(d['pr_mlp']))
        print('f1_mlp  = %f' % np.mean(d['f1_mlp']))  
        print('ROC_auc_mlp  = %f' % np.mean(d['auc_mlp']))    
        print('P_R_auc_mlp  = %f' % np.mean(d['pr_auc_mlp']), end='\n\n')
        
        print('acc of class using MLP classifier %s' % d['target_names'][0], '= %f' % (d['class_acc_mlp'][0]/d['nfold']))
        print('acc of class using MLP classifie %s' % d['target_names'][1], '= %f' % (d['class_acc_mlp'][1]/d['nfold']), end='\n\n')
