# -*- coding: utf-8 -*-
"""
Created on Mon Aug 16 10:18:50 2024

@author: abelac
"""

import numpy as np
import pandas as pd
import warnings
import pickle
import math
from matplotlib import pyplot
warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning)
import tensorflow as tf
import tensorflow.keras.layers as tfl
from tensorflow.keras.applications import EfficientNetB0
from pyDeepInsight import ImageTransformer
from pyDeepInsight.utils import Norm2Scaler
from sklearn.preprocessing import LabelEncoder
from sklearn.manifold import TSNE

import warnings; 
warnings.simplefilter('ignore')

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




# Prepare data
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

Pos_index = 0
Neg_index = 0
window_size = 1 # -0 to +0
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

finetune_from_layer = 1
print(f"Finetuning from layer {finetune_from_layer}")
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

column_names = ['Code','Protein_len','Seq_num','Amino_Acid','Position','Label','Peptide','Mirrored','Feature','Prot_positives']
# randomly pick negative samples to balance it with positve samples (1.5x positive samples)
Negative_Samples = Train_Negatives_All.sample(n=round(len(Train_Positives)*1.5), random_state=42)

# combine positive and negative sets to make the final dataset
Train_set = pd.concat([Train_Positives, Negative_Samples], ignore_index=True, axis=0)


# collect the features and labels of train set
np.set_printoptions(suppress=True)
X_val = [0]*len(Train_set)
for i in range(len(Train_set)):
    feat = Train_set['Feature'][i]
    X_val[i] = feat
X_train_orig = np.asarray(X_val)
X_train_orig = np.float16(X_train_orig)
y_val = Train_set['Label'].to_numpy(dtype=float)
Y_train_orig = y_val.reshape(y_val.size,1)

# Generate a random order of elements with np.random.permutation and simply index into the arrays Feature and label 
idx = np.random.RandomState(seed=42).permutation(len(X_train_orig))
X_train,Y_train = X_train_orig[idx], Y_train_orig[idx]

# load test data
X_independent = [0]*len(Test_Samples)
for i in range(len(Test_Samples)):
    feat = Test_Samples['Feature'][i]
    X_independent[i] = feat
X_test = np.asarray(X_independent)
X_test = np.float16(X_test)
y_independent = Test_Samples['Label'].to_numpy(dtype=float)
Y_test = y_independent.reshape(y_independent.size,1)
#X_test = scaler.transform(X_test) # apply standardization (transform) to the test set

# Normalize data using LogScaler and encode classes
ln = Norm2Scaler()
X_train_norm = ln.fit_transform(X_train)
X_test_norm = ln.transform(X_test)
le = LabelEncoder()
y_train_enc = le.fit_transform(Y_train)
y_test_enc = le.transform(Y_test)
num_classes = np.unique(y_train_enc).size

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
'''
reducer = umap.UMAP(
    n_components=2,
    #min_dist=0.8,
    metric='cosine',
    n_jobs=-1
)
'''
'''
# Create KernelPCA object
reducer = KernelPCA(
    n_components=2,
    random_state=0,
    kernel='poly',
    n_jobs=-1
)
'''

# Initialize image transformer
pixel_size = (130,130)
it = ImageTransformer(
    feature_extractor=reducer, 
    discretization='assignment',
    pixels=pixel_size)

# Train image transformer on training data and transform training and testing sets. Values should be between 0 and 1.
it.fit(X_train_norm, y=Y_train, plot=True)
X_train_img = it.transform(X_train_norm)
X_train_img = np.float16(X_train_img)
X_test_img = it.transform(X_test_norm)
X_test_img = np.float16(X_test_img)

# plot a sample image
pyplot.imshow(X_train_img[0])

preprocess_input = tf.keras.applications.efficientnet.preprocess_input # reuse the pretrained normalization values MobileNetV2 was trained on

IMG_SHAPE = pixel_size + (3,)
base_model = EfficientNetB0(input_shape=IMG_SHAPE, include_top=False, weights='imagenet')


#base_model.summary()

# unfreeze the base model by making it trainable
base_model.trainable = True

# Let's take a look to see how many layers are in the base model
print("Number of layers in the base model: ", len(base_model.layers))

# Fine-tune from this layer onwards
fine_tune_at = finetune_from_layer

# Freeze all the layers before the `fine_tune_at` layer
for layer in base_model.layers[:fine_tune_at]:
  layer.trainable = False

# create the input layer
inputs = tf.keras.Input(shape=IMG_SHAPE)

# data preprocessing using the same weights the model was trained on
x = preprocess_input(inputs)

# set training to False to avoid keeping track of statistics in the batch norm layer
#x = base_model(x, training=False)
x = base_model(x)
#x = base_model

# add the new Binary classification layers
# use global avg pooling to summarize the info in each channel
#x = tfl.GlobalAveragePooling2D()(x) 

x = tfl.Flatten(name="flatten")(x)
#x = tfl.Dense(256, activation="relu")(x)

# include dropout with probability of 0.2 to avoid overfitting
x = tfl.Dropout(0.2)(x)
        
# use a prediction layer with one neuron (as a binary classifier only needs one)
outputs = tfl.Dense(1,activation='sigmoid')(x) 

cnn_model = tf.keras.Model(inputs, outputs)

cnn_model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=2.700747489333154e-10, momentum=0.8498772006669485, weight_decay=0.2396203978419422),
                      loss='binary_crossentropy',
                      metrics=['AUC'])


cnn_model.load_weights('dataset2_B0_model_weights.h5')

eval_result = cnn_model.evaluate(X_test_img, Y_test, verbose=2)
print(f"test loss: {round(eval_result[0],4)}, test auc: {round(eval_result[1],4)}")

Inde_test_prob = cnn_model.predict(X_test_img, verbose=2)
#pickle.dump(Inde_test_prob,open("B0_inde_test_prob.dat","wb"))



