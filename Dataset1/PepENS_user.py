# -*- coding: utf-8 -*-
"""
Created on Fri Aug 22 10:18:50 2025

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
warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning)
from matplotlib import pyplot
import tensorflow as tf
import tensorflow.keras.layers as tfl
from tensorflow.keras.applications import EfficientNetB0
from sklearn.manifold import TSNE
import joblib
import warnings; 
warnings.simplefilter('ignore')
from sklearn.metrics import roc_curve
import copy
from tqdm import tqdm


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
file = open("T5_Features_user.dat",'rb')
Feature1 = pickle.load(file)
file = open("HSE_Features_user.dat",'rb')
Feature2 = pickle.load(file)
file = open("PSSM_Features_user.dat",'rb')
Feature3 = pickle.load(file)


# generate samples for traditional ML models
column_names = ['Code','Protein_len','Seq_num','Amino_Acid','Position','Peptide','Feature']
Samples_classical = pd.DataFrame(columns = column_names)

Pos_index = 0
Neg_index = 0
window_size = 3 # -1 to +1
seq_num = 0

# extract feature and peptide for all sites 
for i in tqdm(range(len(Feature1)), desc="Processing proteins for LR, CatBoost1, and CatBoost2", unit="protein"):
    Protein_seq = Feature1['Prot_seq'][i]
    Feat = Feature1['Feat'][i] # transpose the feature matrix
    Feat2 = Feature2['Feat'][i]
    Feat3 = Feature3['Feat'][i]
    
    seq_num += 1
    for j in range(len(Protein_seq)): # go through the protein seq
        
        A_sample = pd.DataFrame(columns = column_names) # create new dataframe using same column names. This dataframe will just have 1 entry.
        A_sample.loc[0,'Code'] = Feature1['Prot_name'][i] # store the protein name
        A_sample.loc[0,'Protein_len'] = Feature1['Prot_len'][i] # store the protein length
        A_sample.loc[0,'Amino_Acid'] = Protein_seq[j] # store the amino acid 
        A_sample.loc[0,'Position'] = j # store the position of the amino acid
        
        peptide, T5_feat, mirrored = peptide_feat(window_size, Protein_seq, Feat, j) # call the function to extract peptide and feature based on window size
        peptide, HSE_feat, mirrored = peptide_feat(window_size, Protein_seq, Feat2, j)
        peptide, PSSM_feat, mirrored = peptide_feat(window_size, Protein_seq, Feat3, j)
        
        A_sample.loc[0,'Peptide'] = peptide
        Feat_vec = np.concatenate((T5_feat.flatten(),HSE_feat.flatten(),PSSM_feat.flatten()))
        A_sample.loc[0,'Feature'] = np.float16(Feat_vec)
        A_sample.loc[0,'Seq_num'] = seq_num       
        
        Samples_classical = pd.concat([Samples_classical, A_sample], ignore_index=True, axis=0)

print('')
print('Number of Proteins: ' + str(len(Feature1)))
print('Number of samples: ' + str(len(Samples_classical)))
print('')

# collect the features 
X = [0]*len(Samples_classical)
for i in range(len(Samples_classical)):
    feat = Samples_classical['Feature'][i]
    X[i] = feat
X_arr = np.asarray(X)
X_arr = np.float16(X_arr)

# Load the saved scaler
with open("Traditional_ML_scaler.pkl", "rb") as f:
    scaler = pickle.load(f)
X_transformed = scaler.transform(X_arr) # apply standardization (transform)


################### Logisitc Regression ###################

file = open("dataset1_LogisticRegression_model.pkl",'rb') #{'C': 0.001, 'max_iter': 600, 'penalty': 'l2', 'solver': 'liblinear'}
model = pickle.load(file)


pred_probs = model.predict_proba(X_transformed)
LR_prob = pred_probs[:,1] 


################### CatBoost1 ###################

#model details: CatBoostClassifier(random_state=0,depth=6, iterations=1200, learning_rate=0.1, eval_metric='AUC', loss_function='Logloss')
loaded_model = CatBoostClassifier()

loaded_model.load_model('dataset1_Catboost_model1.cbm')

pred_probs = loaded_model.predict_proba(X_transformed) 
CatBoost1_prob = pred_probs[:,1]


################### CatBoost2 ###################

loaded_model = CatBoostClassifier()

loaded_model.load_model('dataset1_Catboost_model2.cbm')

pred_probs = loaded_model.predict_proba(X_transformed) 
CatBoost2_prob = pred_probs[:,1] 


################### EfficientNetB0 ###################

finetune_from_layer = 1

# generate samples for EfficientNetB0 model
Samples_Deep = pd.DataFrame(columns = column_names)

Pos_index = 0
Neg_index = 0
window_size = 1 # -0 to +0
seq_num = 0

# extract feature and peptide for all sites 
for i in tqdm(range(len(Feature1)), desc="Processing proteins for EfficientNetB0", unit="protein"):
    Protein_seq = Feature1['Prot_seq'][i]
    Feat = Feature1['Feat'][i] # transpose the feature matrix
    Feat2 = Feature2['Feat'][i]
    Feat3 = Feature3['Feat'][i]
    
    seq_num += 1
    for j in range(len(Protein_seq)): # go through the protein seq
        
        A_sample = pd.DataFrame(columns = column_names) # create new dataframe using same column names. This dataframe will just have 1 entry.
        A_sample.loc[0,'Code'] = Feature1['Prot_name'][i] # store the protein name
        A_sample.loc[0,'Protein_len'] = Feature1['Prot_len'][i] # store the protein length
        A_sample.loc[0,'Amino_Acid'] = Protein_seq[j] # store the amino acid 
        A_sample.loc[0,'Position'] = j # store the position of the amino acid
        
        peptide, T5_feat, mirrored = peptide_feat(window_size, Protein_seq, Feat, j) # call the function to extract peptide and feature based on window size
        peptide, HSE_feat, mirrored = peptide_feat(window_size, Protein_seq, Feat2, j)
        peptide, PSSM_feat, mirrored = peptide_feat(window_size, Protein_seq, Feat3, j)
        
        A_sample.loc[0,'Peptide'] = peptide
        Feat_vec = np.concatenate((T5_feat.flatten(),HSE_feat.flatten(),PSSM_feat.flatten()))
        A_sample.loc[0,'Feature'] = np.float16(Feat_vec)
        A_sample.loc[0,'Seq_num'] = seq_num       
        
        Samples_Deep = pd.concat([Samples_Deep, A_sample], ignore_index=True, axis=0)

# collect the features 
X = [0]*len(Samples_Deep)
for i in range(len(Samples_Deep)):
    feat = Samples_Deep['Feature'][i]
    X[i] = feat
X_arr = np.asarray(X)
X_arr = np.float16(X_arr)

ln = joblib.load("Norm2Scaler.pkl")
X_norm = ln.transform(X_arr)

# Create t-SNE object
distance_metric = 'cosine'
reducer = TSNE(
    n_components=2,
    metric=distance_metric,
    perplexity=20,
    random_state=0,
    init='random',
    learning_rate='auto',
    n_jobs=-1
)

it = joblib.load("image_transformer.pkl")
X_img = it.transform(X_norm)
X_img = np.float16(X_img)

preprocess_input = tf.keras.applications.efficientnet.preprocess_input 

pixel_size = (130,130)
IMG_SHAPE = pixel_size + (3,)

base_model = EfficientNetB0(input_shape=IMG_SHAPE, include_top=False, weights='imagenet')

# unfreeze the base model by making it trainable
base_model.trainable = True

fine_tune_at = finetune_from_layer

# Freeze all the layers before the `fine_tune_at` layer
for layer in base_model.layers[:fine_tune_at]:
  layer.trainable = False

# create the input layer
inputs = tf.keras.Input(shape=IMG_SHAPE)

# data preprocessing using the same weights the model was trained on
x = preprocess_input(inputs)

# set training to False to avoid keeping track of statistics in the batch norm layer
x = base_model(x)

x = tfl.Flatten(name="flatten")(x)

# include dropout with probability of 0.2 to avoid overfitting
x = tfl.Dropout(0.2)(x)
        
# use a prediction layer with one neuron (as a binary classifier only needs one)
outputs = tfl.Dense(1,activation='sigmoid')(x) 

cnn_model = tf.keras.Model(inputs, outputs)

cnn_model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=2.1917342685441004e-06, momentum=0.9749707652819419, weight_decay=4.8312608215452734e-08),
                      loss='binary_crossentropy',
                      metrics=['AUC'])


cnn_model.load_weights('dataset1_B0_model_weights.h5')

B0_prob = cnn_model.predict(X_img, verbose=2)


################### PepENS ###################

CatBoost1 = CatBoost1_prob.reshape(CatBoost1_prob.size,1)
CatBoost2 = CatBoost2_prob.reshape(CatBoost2_prob.size,1)
LR = LR_prob.reshape(LR_prob.size,1)

final_pred = (B0_prob + CatBoost1 + CatBoost2 + LR) / 4

def round_based_on_thres(probs_to_round, set_thres):
    for i in range(len(probs_to_round)):
        if probs_to_round[i] <= set_thres:
            probs_to_round[i] = 0
        else:
            probs_to_round[i] = 1
    return probs_to_round

set_thres = 0.875
Predicted_labels_all = copy.copy(final_pred)
round_based_on_thres(Predicted_labels_all, set_thres)

# Split predictions by protein length
splits = np.split(Predicted_labels_all, np.cumsum(Feature1["Prot_len"])[:-1])

Labels = pd.DataFrame({
    "Predicted_label": [arr.flatten().astype(int).tolist() for arr in splits]
})
# Convert each list into a string
Labels["Predicted_label"] = Labels["Predicted_label"].apply(lambda x: "".join(map(str, x)))

Labels["Prot_name"] = Feature1["Prot_name"]
Labels["Prot_seq"] = Feature1["Prot_seq"]

# Output 
print('')
print("Protein sequence -> Predicted label")
for seq, label in zip(Labels["Prot_seq"], Labels["Predicted_label"]):
    print(f"{seq} -> {label}")


