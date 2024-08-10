# -*- coding: utf-8 -*-
"""
Created on Mon Nov 14 13:15:47 2022
curves ploting for each fold

@author: Roya Arian
"""

from sklearn.metrics import precision_recall_curve, auc, RocCurveDisplay
import numpy as np

   

def fold_curves(ax, axx, model, x_valid, y_valid, fold_number, mean_fpr, pred_proba, tprs=[], aucs=[], y_pred=[]):
    """
    A function for ploting ROC and PR curves of each fold 
    
    """
    ############ ROC Curve 
    
    viz = RocCurveDisplay.from_estimator(
        model,
        x_valid,
        y_valid,
        name="ROC Curve fold {}".format(fold_number),
        alpha=0.3,
        lw=1,
        ax=ax,
    )
    interp_tpr = np.interp(mean_fpr, viz.fpr, viz.tpr)
    interp_tpr[0] = 0.0
    tprs.append(interp_tpr)
    aucs.append(viz.roc_auc)
    
    
    ############ P_R Curve 
    precision, recall, _ = precision_recall_curve(y_valid, pred_proba)
    y_pred = np.append(y_pred, pred_proba, axis = 0)
    # plot the model precision-recall curve
    axx.plot(recall, precision, lw=1, alpha=0.3, label=r"P_R_curve fold %d (AUC =  %0.2f)" % (fold_number, auc(recall, precision)))

    return tprs, aucs, y_pred