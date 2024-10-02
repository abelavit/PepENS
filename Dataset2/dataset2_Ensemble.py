# -*- coding: utf-8 -*-
"""
Created on Mon Aug  8 10:18:50 2024

@author: abelac
"""

import numpy as np 
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import precision_score
import warnings
import pickle
import copy
from sklearn.metrics import confusion_matrix
from matplotlib import pyplot
warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning)


file = open("dataset2_B0_inde_test_prob.dat",'rb')
B0_inde_test_prob = pickle.load(file)
file = open("dataset2_CatBoost1_inde_test_prob.dat",'rb')
CatBoost_inde_test_prob = pickle.load(file)
file = open("dataset2_CatBoost2_inde_test_prob.dat",'rb')
CatBoost_inde_test_prob2 = pickle.load(file)
file = open("dataset2_LogisticRegression_inde_test_prob.dat",'rb')
LR_inde_test_prob = pickle.load(file)
file = open("dataset2_inde_test_true_prob.dat",'rb')
y_independent = pickle.load(file)

CatBoost_pred_probability = CatBoost_inde_test_prob[:,1] # for AUC calculation
CatBoost_pred_probability2 = CatBoost_inde_test_prob2[:,1] # for AUC calculation
LR_pred_probability = LR_inde_test_prob[:,1] # for AUC calculation
CatBoost = CatBoost_pred_probability.reshape(CatBoost_pred_probability.size,1)
CatBoost2 = CatBoost_pred_probability2.reshape(CatBoost_pred_probability2.size,1)
LR = LR_pred_probability.reshape(LR_pred_probability.size,1)
auc_B0 = roc_auc_score(y_independent, B0_inde_test_prob)
auc_CatBoost = roc_auc_score(y_independent, CatBoost)
auc_CatBoost2 = roc_auc_score(y_independent, CatBoost2)
auc_LR = roc_auc_score(y_independent, LR)

print('')
print('Individual Model AUCs:')
print('-----------------------------------')
print('EfficientNetB0: ' + str(round(auc_B0,4)))
print('CatBoost 1: ' + str(round(auc_CatBoost,4)))
print('CatBoost 2: ' + str(round(auc_CatBoost2,4)))
print('LR: ' + str(round(auc_LR,4)))
print('-----------------------------------')

print('')

final_pred = (B0_inde_test_prob + CatBoost + CatBoost2 + LR) / 4
auc_overall = round(roc_auc_score(y_independent, final_pred),4)

def round_based_on_thres(probs_to_round, set_thres):
    for i in range(len(probs_to_round)):
        if probs_to_round[i] <= set_thres:
            probs_to_round[i] = 0
        else:
            probs_to_round[i] = 1
    return probs_to_round

# calculate the metrics
set_thres = 0.88
copy_Probs_inde = copy.copy(final_pred)
round_based_on_thres(copy_Probs_inde, set_thres)
fpr, tpr, thresholds = roc_curve(y_independent, final_pred)
inde_pre = round(precision_score(y_independent, copy_Probs_inde),4)
inde_mcc = round(matthews_corrcoef(y_independent, copy_Probs_inde),4)
cm = confusion_matrix(y_independent, copy_Probs_inde) # for acc, sen, and spe calculation
total_preds=sum(sum(cm))
TN = cm[0,0]
FP = cm[0,1]
FN = cm[1,0]
TP = cm[1,1]
inde_sen = round(TP/(TP+FN),4)
inde_spe = round(TN/(TN+FP),4)

# display the metrics
print('Ensemble Result:')
print('-----------------------------------')
print(f'Independent Sen: {inde_sen}')
print(f'Independent Spe: {inde_spe}')
print(f'Independent Pre: {inde_pre}')
print(f'Independent AUC: {auc_overall}')
print(f'Independent MCC: {inde_mcc}')

# plot ROC curve
legend = 'AUC = ' + str(auc_overall)
pyplot.figure(figsize=(12,8))
pyplot.plot([0,1], [0,1], linestyle='--')
pyplot.plot(fpr, tpr, marker='.', label=legend)
pyplot.xlabel('False Positive Rate')
pyplot.ylabel('True Positive Rate')
pyplot.legend()
pyplot.show()
