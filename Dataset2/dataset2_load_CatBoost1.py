# -*- coding: utf-8 -*-
"""
Created on Mon Aug  8 10:18:50 2024

@author: abelac
"""

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import roc_auc_score
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import precision_score
import warnings
import math 
import pickle
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler
warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning)


# function to extract samples
def peptide_feat(window_size, Protein_seq, Feat, j): # funtion to extract peptide length and feature based on window size
    
    if (j - math.ceil(window_size/2)) < -1: # not enough amino acid at N terminus to form peptide
        peptide1 = Protein_seq[j:math.floor(window_size/2)+j+1] # +1 since the stop value for slicing is exclusive
        peptide2 = Protein_seq[j+1:math.floor(window_size/2)+j+1] # other peptide half but excluding the central amino acid
        peptide = peptide2[::-1] + peptide1
        
        feat1 = Feat[j:math.floor(window_size/2)+j+1] # +1 since the stop value for slicing is exclusive
        feat2 = Feat[j+1:math.floor(window_size/2)+j+1] # other peptide half but excluding the central amino acid
        final_feat = np.concatenate((feat2[::-1], feat1))
        mirrored = 'Yes'
        
    elif ((len(Protein_seq) - (j+1)) < (math.floor(window_size/2))): # not enough amino acid at C terminus to form peptide
        peptide1 = Protein_seq[j-math.floor(window_size/2):j+1]
        peptide2 = Protein_seq[j-math.floor(window_size/2):j]
        peptide = peptide1 + peptide2[::-1]
        
        feat1 = Feat[j-math.floor(window_size/2):j+1]
        feat2 = Feat[j-math.floor(window_size/2):j]
        final_feat = np.concatenate((feat1, feat2[::-1]))
        mirrored = 'Yes'
        
    else:
        peptide = Protein_seq[j-math.floor(window_size/2):math.floor(window_size/2)+j+1]
        final_feat = Feat[j-math.floor(window_size/2):math.floor(window_size/2)+j+1]
        mirrored = 'No'
        
    return peptide, final_feat, mirrored




## Prepare data
Dataset_test_tsv = pd.read_table("Dataset2_test.tsv")
Dataset_train_tsv = pd.read_table("Dataset2_train.tsv")

file = open("T5_Features.dat",'rb')
Proteins = pickle.load(file)
file = open("HSE_Features.dat",'rb')
Proteins2 = pickle.load(file)
file = open("PSSM_Features.dat",'rb')
Proteins3 = pickle.load(file)

column_headers = list(Proteins.columns.values)
DatasetTestProteins = pd.DataFrame(columns = column_headers)
DatasetTestProteins2 = pd.DataFrame(columns = column_headers)
DatasetTestProteins3 = pd.DataFrame(columns = column_headers)

matching_index = 0
for i in range(len(Dataset_test_tsv)):
    for j in range(len(Proteins)):
        if (Dataset_test_tsv['seq'][i].upper() == Proteins['Prot_seq'][j].upper()):           
            DatasetTestProteins.loc[matching_index] = Proteins.loc[j]
            matching_index += 1
            break
matching_index = 0
for i in range(len(Dataset_test_tsv)):
    for j in range(len(Proteins2)):
        if (Dataset_test_tsv['seq'][i].upper() == Proteins2['Prot_seq'][j].upper()):
            DatasetTestProteins2.loc[matching_index] = Proteins2.loc[j]
            matching_index += 1
            break
matching_index = 0
for i in range(len(Dataset_test_tsv)):
    for j in range(len(Proteins3)):
        if (Dataset_test_tsv['seq'][i].upper() == Proteins3['Prot_seq'][j].upper()):
            DatasetTestProteins3.loc[matching_index] = Proteins3.loc[j]
            matching_index += 1
            break   
            
DatasetTrainProteins = pd.DataFrame(columns = column_headers)
DatasetTrainProteins2 = pd.DataFrame(columns = column_headers)
DatasetTrainProteins3 = pd.DataFrame(columns = column_headers)

matching_index = 0
for i in range(len(Dataset_train_tsv)):
    for j in range(len(Proteins)):
        if (Dataset_train_tsv['seq'][i].upper() == Proteins['Prot_seq'][j].upper()):       
            DatasetTrainProteins.loc[matching_index] = Proteins.loc[j]
            matching_index += 1
            break

matching_index = 0
for i in range(len(Dataset_train_tsv)):
    for j in range(len(Proteins2)):
        if (Dataset_train_tsv['seq'][i].upper() == Proteins2['Prot_seq'][j].upper()):
            DatasetTrainProteins2.loc[matching_index] = Proteins2.loc[j]
            matching_index += 1
            break    
matching_index = 0
for i in range(len(Dataset_train_tsv)):
    for j in range(len(Proteins3)):
        if (Dataset_train_tsv['seq'][i].upper() == Proteins3['Prot_seq'][j].upper()):
            DatasetTrainProteins3.loc[matching_index] = Proteins3.loc[j]
            matching_index += 1
            break 

# generate samples for Test protein sequences
column_names = ['Code','Protein_len','Seq_num','Amino_Acid','Position','Label','Peptide','Mirrored','Feature','Prot_positives']
Test_Samples = pd.DataFrame(columns = column_names)
#Test_Negatives = pd.DataFrame(columns = column_names)

Pos_index = 0
Neg_index = 0
window_size = 3 # -1 to +1
seq_num = 0

# extract feature and peptide for all sites 
for i in range(len(DatasetTestProteins)):
    Protein_seq = DatasetTestProteins['Prot_seq'][i]
    Feat = DatasetTestProteins['Feat'][i] # transpose the feature matrix
    Feat2 = DatasetTestProteins2['Feat'][i]
    Feat3 = DatasetTestProteins3['Feat'][i]
    positive_counts = DatasetTestProteins['Prot_label'][i].count('1')
    
    seq_num += 1
    for j in range(len(Protein_seq)): # go through the protein seq
        
        A_sample = pd.DataFrame(columns = column_names) # create new dataframe using same column names. This dataframe will just have 1 entry.
        A_sample.loc[0,'Code'] = DatasetTestProteins['Prot_name'][i] # store the protein name
        A_sample.loc[0,'Protein_len'] = DatasetTestProteins['Prot_len'][i] # store the protein length
        A_sample.loc[0,'Label'] = DatasetTestProteins['Prot_label'][i][j]
        A_sample.loc[0,'Prot_positives'] = positive_counts
        A_sample.loc[0,'Amino_Acid'] = Protein_seq[j] # store the amino acid 
        A_sample.loc[0,'Position'] = j # store the position of the amino acid


        peptide, T5_feat, mirrored = peptide_feat(window_size, Protein_seq, Feat, j) # call the function to extract peptide and feature based on window size
        peptide, HSE_feat, mirrored = peptide_feat(window_size, Protein_seq, Feat2, j)
        peptide, PSSM_feat, mirrored = peptide_feat(window_size, Protein_seq, Feat3, j)
        
        A_sample.loc[0,'Peptide'] = peptide
        Feat_vec = np.concatenate((T5_feat.flatten(),HSE_feat.flatten(),PSSM_feat.flatten()))
        A_sample.loc[0,'Feature'] = np.float16(Feat_vec)
        A_sample.loc[0,'Seq_num'] = seq_num
        A_sample.loc[0,'Mirrored'] = mirrored        
        
        Test_Samples = pd.concat([Test_Samples, A_sample], ignore_index=True, axis=0)
            
                  
    print('Test Protein ' + str(i+1) + ' out of ' + str(len(DatasetTestProteins)))
print('Number of Proteins in Test: ' + str(len(DatasetTestProteins)))
print('Number of samples in Test: ' + str(len(Test_Samples)))

# generate samples for Train protein sequences
Train_Positives = pd.DataFrame(columns = column_names)
Train_Negatives_All = pd.DataFrame(columns = column_names)

Pos_index = 0
Neg_index = 0
seq_num = 0

# extract feature and peptide for all sites 
for i in range(len(DatasetTrainProteins)):
    Protein_seq = DatasetTrainProteins['Prot_seq'][i]
    Feat = DatasetTrainProteins['Feat'][i] # transpose the feature matrix
    Feat2 = DatasetTrainProteins2['Feat'][i]
    Feat3 = DatasetTrainProteins3['Feat'][i]
    positive_counts = DatasetTrainProteins['Prot_label'][i].count('1')
    
    seq_num += 1
    for j in range(len(Protein_seq)): # go through the protein seq
            
        A_sample = pd.DataFrame(columns = column_names) # create new dataframe using same column names. This dataframe will just have 1 entry.
        A_sample.loc[0,'Code'] = DatasetTrainProteins['Prot_name'][i] # store the protein name
        A_sample.loc[0,'Protein_len'] = DatasetTrainProteins['Prot_len'][i] # store the protein length
        A_sample.loc[0,'Label'] = DatasetTrainProteins['Prot_label'][i][j]
        A_sample.loc[0,'Prot_positives'] = positive_counts
        A_sample.loc[0,'Amino_Acid'] = Protein_seq[j] # store the amino acid 
        A_sample.loc[0,'Position'] = j # store the position of the amino acid

        peptide, T5_feat, mirrored = peptide_feat(window_size, Protein_seq, Feat, j) # call the function to extract peptide and feature based on window size
        peptide, HSE_feat, mirrored = peptide_feat(window_size, Protein_seq, Feat2, j)
        peptide, PSSM_feat, mirrored = peptide_feat(window_size, Protein_seq, Feat3, j)
        
        A_sample.loc[0,'Peptide'] = peptide
        Feat_vec = np.concatenate((T5_feat.flatten(),HSE_feat.flatten(),PSSM_feat.flatten()))
        A_sample.loc[0,'Feature'] = np.float16(Feat_vec)
        A_sample.loc[0,'Seq_num'] = seq_num
        A_sample.loc[0,'Mirrored'] = mirrored
                        
        if A_sample.loc[0,'Label'] == '1':
            Train_Positives = pd.concat([Train_Positives, A_sample], ignore_index=True, axis=0)
               
        else: 
            Train_Negatives_All = pd.concat([Train_Negatives_All, A_sample], ignore_index=True, axis=0)
      
            
    print('Train Protein ' + str(i+1) + ' out of ' + str(len(DatasetTrainProteins)))
print('Number of Proteins in Train: ' + str(len(DatasetTrainProteins)))
print('Feature vector size: ' + str(Test_Samples['Feature'][0].shape))
print('Num of Train Positives: ' + str(len(Train_Positives)))
print('Num of Train Negatives (All): ' + str(len(Train_Negatives_All)))

## randomly pick negative samples to balance it with positve samples (1.5x positive samples)
Negative_Samples = Train_Negatives_All.sample(n=round(len(Train_Positives)*1.5), random_state=12)

## combine positive and negative sets to make the final dataset
Train_set = pd.concat([Train_Positives, Negative_Samples], ignore_index=True, axis=0)

## collect the features and labels of train set
# collect the features and labels of train set
np.set_printoptions(suppress=True)
X = [0]*len(Train_set)
for i in range(len(Train_set)):
    feat = Train_set['Feature'][i]
    X[i] = feat
X_all = np.asarray(X)
X_all = np.float16(X_all)
y_all = Train_set['Label'].astype(int)


## calculate performance on independent by loading saved model
# collect the features and labels for independent set
X_independent = [0]*len(Test_Samples)
for i in range(len(Test_Samples)):
    feat = Test_Samples['Feature'][i]
    X_independent[i] = feat
X_independent_arr = np.asarray(X_independent)
X_independent_arr = np.float16(X_independent_arr)
y_independent = Test_Samples['Label'].astype(int)

scaler = StandardScaler()
scaler.fit(X_all) # fit on training set only
X_all = scaler.transform(X_all) # apply transform to the training set

X_independent_test = scaler.transform(X_independent_arr) # apply standardization (transform) to the test set

#model details: CatBoostClassifier(random_state=0,depth=7, iterations=500, learning_rate=0.1, eval_metric='AUC', loss_function='Logloss')
loaded_model = CatBoostClassifier()

loaded_model.load_model('dataset2_Catboost_model1.cbm')

# do predictions of the trained model on test data
predictions = loaded_model.predict(X_independent_test)
pred_probs = loaded_model.predict_proba(X_independent_test) # for AUC calculation
pred_probability = pred_probs[:,1] # for AUC calculation

pre_independent = precision_score(y_independent, predictions)
mcc_independent = matthews_corrcoef(y_independent, predictions)
auc_independent = roc_auc_score(y_independent, pred_probability)
F1_score_independent = f1_score(y_independent, predictions)
cm = confusion_matrix(y_independent, predictions) # for acc, sen, and spe calculation
total_preds=sum(sum(cm))
TN = cm[0,0]
FP = cm[0,1]
FN = cm[1,0]
TP = cm[1,1]
acc_independent = (TN+TP)/(total_preds)
sen_independent = TP/(TP+FN)
spe_independent = TN/(TN+FP)

print('-----------------------------------')
print('Independent test set Results')
print('-----------------------------------')
print('SEN: ' + str(round(sen_independent,4)))
print('-----------------------------------')
print('SPE: ' + str(round(spe_independent,4)))
print('-----------------------------------')
print('PRE: ' + str(round(pre_independent,4)))
#print('-----------------------------------')   
#print('ACC: ' + str(round(acc_independent,4)))
print('-----------------------------------') 
print('MCC: ' + str(round(mcc_independent,4)))
print('-----------------------------------')
print('AUC: ' + str(round(auc_independent,4)))
#print('-----------------------------------') 
#print('F1 score: ' + str(round(F1_score_independent,4)))
print('-----------------------------------')

