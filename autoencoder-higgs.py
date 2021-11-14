# -*- coding: utf-8 -*-
"""
Spyder Editor

@author: Alkım
"""

#Import all necessary libraries and tools
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1' #CPU is used only

import pickle
import tensorflow as tf
print(tf.__version__)
from keras.models import load_model, Model, Sequential
from keras.layers import Dense, Input, concatenate
from keras.callbacks import ModelCheckpoint, TensorBoard
import numpy as np
import pandas as pd
from keras import regularizers 
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import (confusion_matrix, precision_recall_curve, auc,
                             roc_curve, recall_score, classification_report, f1_score,
                             precision_recall_fscore_support, average_precision_score, plot_roc_curve)
import matplotlib.pyplot as plt
from scipy import interp
from matplotlib.pyplot import figure
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.utils import shuffle
from sklearn.model_selection import KFold

#Load combined data values to the variables
infile_combined = open("C:/Users/Alkım/Desktop/dfs/combinedmhc90_d1.pkl",'rb') #Combined data
model_test = pickle.load(infile_combined, encoding="latin1")
#Load background data values to the variables
infile_background = open("C:/Users/Alkım/Desktop/dfs/bkgdmhc90_d1.pkl",'rb') #Background data
model_train = pickle.load(infile_background, encoding="latin1")

#Define features
feature_text=['TreeS_lepton_pt','TreeS_MET','TreeS_N_jets',
  'TreeS_N_Bjets','TreeS_R_lb','TreeS_Eta_lb',
	'TreeS_jet1Pt','TreeS_jet1Phi','TreeS_jet1Eta',
  'TreeS_jet1Btag',
	'TreeS_lepton_eta','TreeS_lepton_phi',
	'TreeS_METEta', 'TreeS_METPhi','TreeS_Pt_btag']

#Define Signal Label
label_text = ['signal']

#Filter the label values for testing
label_values_test= model_test.filter(label_text)

#Filter feature values for testing
feature_values_test= model_test.filter(feature_text)

#Filter the label values for training
label_values_train= model_train.filter(label_text)

#Filter feature values for training
feature_values_train= model_train.filter(feature_text)

#Filtered data values are assigned to numpy array
features_array_test=np.array(feature_values_test) 
label_array_test=np.array(label_values_test)
features_array_train=np.array(feature_values_train) 
label_array_train=np.array(label_values_train)

scaler = MinMaxScaler()

#Scale the input of train data
scaled_seqs = scaler.fit_transform(features_array_train)

#Scale the input of test data
scaled_seqs1 = scaler.fit_transform(features_array_test)

seed = 42 #Seed chosen as 42
np.random.seed(seed)

#Define the number of neuron for layers
input_size=15
hidden_size1=8
hidden_size2=4

#Empty array definitions to hold values
aucscores = []
aucprscores = []
mse_values=[]
mse_m=[]
y_values=[]
total_mse=[]
thresholdvalues=[]
tprs = []
aucs = []

nb_epoch = 50 #Number of epochs is set as 50
batch_size = 256 #Batch size is set as 256

X=scaled_seqs #scaled train data (Background)
x=scaled_seqs1 #scaled test data (Combined)
y=label_array_test #output for test data (Combined)

#Shuffle the test data
x, y = shuffle(x, y, random_state=42)

n_split=5 #number of split

#Define some necessary variables for plotting
mean_fpr = np.linspace(0, 1, 100)
fig, ax = plt.subplots()

i=0; #define counter
 
for train_index,test_index in KFold(n_split).split(X): #split train data X (cross validation loop)
  i=i+1  #increase counter
  
  #autoencoder neural network structure
  input_a = Input(shape=(input_size,))
  hidden_1 = Dense(hidden_size1, activation='tanh')(input_a)
  hidden_2 = Dense(hidden_size2, activation='relu')(hidden_1)
  hidden_3 = Dense(hidden_size2, activation='tanh')(hidden_2)
  hidden_4 = Dense(hidden_size1, activation='relu')(hidden_3)
  output_a = Dense(15, activation='tanh', name="output")(hidden_4)

  #split data as train and test(will be used only for training)
  X_train,X_test=X[train_index],X[test_index]
  x_train,x_test=x[train_index],x[test_index]
  print(np.intersect1d(X_train[:,0], x_train[:,0]).shape)
  y_train,y_test=y[train_index],y[test_index]
  
  #define autoencoder model
  autoencoder = Model(inputs=input_a, outputs=output_a)
  autoencoder.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])
  #training model with the features data
  autoencoder.fit(X_train, X_train,
                       epochs=nb_epoch,
                       batch_size=batch_size,
                       shuffle=True,
                       validation_data=(x_test, x_test), #validation of data with the test data
                       verbose=1).history
  
  autoencoder.save(str(i)+"-90GeV") #Save trained model
  #autoencoder=load_model(str(i)+"-90GeV") #Load trained model
  
  predictions = autoencoder.predict(x_test)  # prediction with the test data
  mse = np.mean(np.power(x_test - predictions, 2), axis=1) #mean squared error calculation(Reconstruction error)
  
  #necessary array operations
  mse=np.array(mse)
  mse=np.reshape(mse, (np.size(x_test,0), ))
  mse=np.transpose(mse)
  
  #necessary array operations to adjust the appropriate shapes
  y_test=np.transpose(y_test)
  y_test=np.reshape(y_test, (np.size(x_test,0), ))
  y_values.append(y_test)
  
  #print the shapes to check
  print(x_test.shape)
  print(y_test.shape)
  print(mse.shape)
  print(pd.value_counts(y_test, sort = True))
  
  #Define a dataframe to hold mse and y_test values
  error_df = pd.DataFrame({'reconstruction_error': mse, 
                        'true_class': y_test})
  
  #By using the roc_curve function fpr, tpr and thresholds are obtained
  fpr, tpr, thresholds = roc_curve(error_df.true_class, error_df.reconstruction_error)
  #obtaine the aucpr scores
  ap = average_precision_score(y_test, mse)
  #obtaine the auc roc scores
  roc_auc = auc(fpr, tpr)
  #plot roc curve for the each cross validation fold in single plot as grey color
  ax.plot(fpr, tpr, color='grey', label='ROC fold {} (AUC = %0.3f)'.format(i)% roc_auc, alpha=0.2)
  #hold auc scores in an array
  aucscores.append(roc_auc * 100)
  #hold aucpr scores in an array
  aucprscores.append(ap * 100)
  
  #Necessary operations for the plotting
  interp_tpr = interp(mean_fpr, fpr, tpr)
  interp_tpr[0] = 0.0
  tprs.append(interp_tpr)
  aucs.append(roc_auc)
#print the mean value of auc scores and standard deviation of auc scores
print("%.2f%% (+/- %.2f%%)" % (np.mean(aucscores), np.std(aucscores)))
#print the mean value of aucpr scores and standard deviation of aucpr scores
print("%.2f%% (+/- %.2f%%)" % (np.mean(aucprscores), np.std(aucprscores)))
#Center line for the roc curve
ax.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
        label='Chance', alpha=.8)
#Obtaine the mean tpr value for all cross validations
mean_tpr = np.mean(tprs, axis=0)
mean_tpr[-1] = 1.0
#Obtaine the mean and standard deviation auc value for all cross validations
mean_auc = auc(mean_fpr, mean_tpr)
std_auc = np.std(aucs)
#plot the mean auc roc curve
ax.plot(mean_fpr, mean_tpr, color='b',
        label=r'Mean ROC (AUC = %0.3f $\pm$ %0.2f)' % (mean_auc, std_auc),
        lw=2, alpha=.8)
#Obtaine the standard deviation of tprs
std_tpr = np.std(tprs, axis=0)
#Obtaine the upper and lower limit of tprs for plotting
tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
#fill the gap between cross validation roc curves
ax.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,
                label=r'$\pm$ 1 std. dev.')
#Define the title and labels of the graph
ax.set(xlim=[-0.05, 1.05], ylim=[-0.05, 1.05],
       title="Receiver Operating Characteristic for 90GeV Invariant Mass Data", ylabel='True Positive Rate', xlabel='False Positive Rate')
ax.legend(loc="lower right")
plt.show()