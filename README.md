# PepENS
PepENS is an innovative model that seamlessly integrates sequence and structure-based information with ensemble learning. It represents a pioneering, consensus-based method by combining embeddings from ProtT5-XL-UniRef50 with Position Specific Scoring Matrices and Half-Sphere Exposure features to train an ensemble model consisting of EfficientNetB0 via image output from DeepInsight technology, CatBoost, and Logistic Regression. To the best of our knowledge, this is the first time representations learnt using attention mechanism in transformers are transformed into images to utilize the potential of Convolutional Neural Networks in protein-peptide prediction. The spatial relationships formed between the features by DeepInsight to produce images and the feature extraction from those images by the Convolutional Neural Network play a key role in the performance of the ensemble model. Extensive evaluations demonstrate that PepENS establishes a new benchmark, surpassing existing methods. Superior predictive performance of PepENS redefines computational approaches and holds the potential to accelerate drug discovery, deepen our understanding of disease mechanisms, and inspire new computational strategies in bioinformatics.

![Architecture](https://github.com/user-attachments/assets/281fe6fb-2834-49c0-8018-f8d4e9099c1f)

# Download and Use
The codes for Datasets 1 and 2 are found in the respective folders of this repository.      
## 1. Load the PepENS model
The results obtained in our work can be replicated by executing dataset1_Ensemble.py script for Dataset1, and dataset2_Ensemble.py script for Dataset2. The respective scripts will load the probability files of the individual models and output the final result. 
## 2. Load individual models
To obtain the probabilities of the individual models, firstly download the three features from this [link](https://figshare.com/account/home#/projects/176151) (caution: data size is 1.22GB) and then run the scripts containing the word 'load' in the name (for instance, run the dataset1_load_CatBoost1.py script to obtain the probabilities of the CatBoost1 model of dataset 1). In case of the EfficientNetB0 model, the model weights would be needed which can be downloaded from [here](https://figshare.com/articles/software/EfficientNetB0_model_weights/27126339). 

## Packages
Packages for running EfficientNetB0:  
python 3.10.13  
numpy 1.23.5  
pandas 1.5.3  
pickle 4.0  
matplotlib 3.8.0  
tensorflow 2.14.0  
pyDeepInsight 0.1.1  
scikit-learn 1.3.0

Packages for running other scripts:  
python 3.10.12  
pandas 1.5.3  
pickle 4.0  
numpy 1.25.2  
scikit-learn 1.2.2  
catboost 1.2.5  
matplotlib 3.7.2

