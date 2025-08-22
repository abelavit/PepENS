# PepENS
PepENS is an innovative model that seamlessly integrates sequence and structure-based information with ensemble learning. It represents a pioneering, consensus-based method by combining embeddings from ProtT5-XL-UniRef50 with Position Specific Scoring Matrices and Half-Sphere Exposure features to train an ensemble model consisting of EfficientNetB0 via image output from DeepInsight technology, CatBoost, and Logistic Regression. To the best of our knowledge, this is the first time representations learnt using attention mechanism in transformers are transformed into images to utilize the potential of Convolutional Neural Networks in protein-peptide prediction. The spatial relationships formed between the features by DeepInsight to produce images and the feature extraction from those images by the Convolutional Neural Network play a key role in the performance of the ensemble model. Extensive evaluations demonstrate that PepENS establishes a new benchmark, surpassing existing methods. Superior predictive performance of PepENS redefines computational approaches and holds the potential to accelerate drug discovery, deepen our understanding of disease mechanisms, and inspire new computational strategies in bioinformatics.

<img width="732" height="1117" alt="Architecture" src="https://github.com/user-attachments/assets/48a6c123-fd07-4688-8033-818d0ec4206b" />

# Download and Use
The codes for Datasets 1 and 2 are found in the respective folders of this repository.      
## 1. Load the PepENS model
The results obtained in our work can be replicated by executing dataset1_Ensemble.py script for Dataset1, and dataset2_Ensemble.py script for Dataset2. The respective scripts will load the probability files of the individual models and output the final result. 
## 2. Load individual models
To obtain the probabilities of the individual models, firstly download the three features (HSE_Features.dat, T5_Features.dat, and PSSM_Features.dat) from this [link](https://figshare.com/projects/Train_the_CNN_model/176151) (caution: data size is 1.22GB) and then run the scripts containing the word 'load' in the name (for instance, run the dataset1_load_CatBoost1.py script to obtain the probabilities of the CatBoost1 model of dataset 1). For the EfficientNetB0 model, the model weights can be downloaded from [here](https://figshare.com/articles/software/EfficientNetB0_model_weights/27126339). 
## 3. Predicting peptide binding sites on other protein(s) 
The users can use the PepENS tool to predict the peptide binding sites in their protein(s). The script 'PepENS_user.py' (Dataset1 directory) can be run to achieve this. The users would need to extract the PSSM, Transformer embedding, and the HSE features prior to using the tool. Example protein files (HSE_Features_user.dat, T5_Features_user.dat, and PSSM_Features_user.dat) are provided on which the script can be run and this serves as a guide.      
## Packages 
python 3.8.20  
numpy 1.24.3  
pandas 1.5.0   
matplotlib 3.7.5   
tensorflow 2.13.0  
pyDeepInsight 0.2.0  
scikit-learn 1.3.2  
catboost 1.2.5  


