# PepENS
PepENS is an innovative model that seamlessly integrates sequence and structure-based information with ensemble learning. It represents a pioneering, consensus-based method by combining embeddings from ProtT5-XL-UniRef50 with Position Specific Scoring Matrices and Half-Sphere Exposure features to train an ensemble model consisting of EfficientNetB0 via image output from DeepInsight technology, CatBoost, and Logistic Regression. To the best of our knowledge, this is the first time representations learnt using attention mechanism in transformers are transformed into images to utilize the potential of Convolutional Neural Networks in protein-peptide prediction. The spatial relationships formed between the features by DeepInsight to produce images and the feature extraction from those images by the Convolutional Neural Network play a key role in the performance of the ensemble model. Extensive evaluations demonstrate that PepENS establishes a new benchmark, surpassing existing methods. Superior predictive performance of PepENS redefines computational approaches and holds the potential to accelerate drug discovery, deepen our understanding of disease mechanisms, and inspire new computational strategies in bioinformatics.

![Architecture](https://github.com/user-attachments/assets/281fe6fb-2834-49c0-8018-f8d4e9099c1f)

# Download and Use
The codes for Datasets 1 and 2 are found in the respective folders of this repository.  execute the codes, 
## 1. Load the trained PepCNN model
   The result obtained in our work can be replicated by executing dataset1_PepCNN.py script for Dataset1, and dataset2_PepCNN.py script for Dataset2. For instance, to obtain the result of PepCNN on dataset1, run the dataset1_PepCNN.py script after downloading the following files by going to this [link](https://figshare.com/projects/Load_protein-peptide_binding_PepCNN_model/176094) (caution: data size is around 1.3GB for each dataset): 
   - model weights: dataset1_best_model_weights.h5
   - training set negative samples: dataset1_Train_Negatives_All.dat
   - training set positive samples: dataset1_Train_Positives.dat
   - testing set: dataset1_Test_Samples.dat
## 2. Train the CNN model
To train the network from scratch, it can be done by executing dataset1_PepCNN_train.py script for Dataset1, and dataset2_PepCNN_train.py script for Dataset2. For instance, to train the network on dataset1, run the dataset1_PepCNN_train.py script after downloading the following files by going to this [link](https://figshare.com/projects/Train_the_CNN_model/176151) (caution: data size is 1.22GB for both datasets): 
   - testing protein sequences: Dataset1_test.tsv
   - protein sequences excluding testing sequences: Dataset1_train.tsv
   - pre-trained transformer embeddings: T5_Features.dat
   - PSSM features: PSSM_Features.dat
   - HSE features: HSE_Features.dat

Package versions:
Python 3.10.12,
Pandas 1.5.3,
Pickle 4.0,
Numpy 1.25.2,
scikit-learn 1.2.2,
Matplotlib 3.7.2,
Tensorflow 2.12.0
