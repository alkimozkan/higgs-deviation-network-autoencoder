# -*- coding: utf-8 -*-
"""
Created on Sat May  2 22:39:23 2020

@author: Alkım
"""
#Import necessary libraries and another tools
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import numpy as np
np.random.seed(42)
MAX_INT = np.iinfo(np.int32).max
data_format = 0
# %tensorflow_version 1.14
tf.set_random_seed(42)
print(tf.__version__)
sess = tf.Session()

from keras import regularizers
from keras import backend as K
from keras.models import Model, load_model
from keras.layers import Input, Dense, concatenate
from keras.optimizers import RMSprop
from keras.callbacks import ModelCheckpoint, TensorBoard
from sklearn.metrics import (confusion_matrix, precision_recall_curve, auc,
                             roc_curve, recall_score, classification_report, f1_score,
                             precision_recall_fscore_support, average_precision_score, roc_auc_score)
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.utils import shuffle
from scipy import interp
import sys
import time
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure 
from sklearn import preprocessing
import pickle

infile_combined = open("C:/Users/Alkım/Desktop/dfs/combinedmhc90_d1.pkl",'rb') #Combined data
model_test = pickle.load(infile_combined, encoding="latin1")

#Define features
feature_text=['TreeS_lepton_pt','TreeS_MET','TreeS_N_jets',
  'TreeS_N_Bjets','TreeS_R_lb','TreeS_Eta_lb',
	'TreeS_jet1Pt','TreeS_jet1Phi','TreeS_jet1Eta',
  'TreeS_jet1Btag',
	'TreeS_lepton_eta','TreeS_lepton_phi',
	'TreeS_METEta', 'TreeS_METPhi','TreeS_Pt_btag']


#Label for signal
label_text = ['signal']

#Filter the label values 
label_values_test= model_test.filter(label_text)

#Filter the feature values 
feature_values_test= model_test.filter(feature_text)

#Filtered data values are assigned to numpy array
features_array_test=np.array(feature_values_test) 
label_array_test=np.array(label_values_test)

def aucPerformance(mse, labels):
    roc_auc = roc_auc_score(labels, mse)
    ap = average_precision_score(labels, mse)
    print("AUC-ROC: %.4f, AUC-PR: %.4f" % (roc_auc, ap))
    return roc_auc, ap;

def dev_network_s(input_shape):
    
    x_input = Input(shape=input_shape)
    intermediate = Dense(1000, activation='relu',
                kernel_regularizer=regularizers.l2(0.01), name = 'hl1')(x_input)
    intermediate = Dense(250, activation='relu',
                kernel_regularizer=regularizers.l2(0.01), name = 'hl2')(intermediate)
    intermediate = Dense(15, activation='relu',
                kernel_regularizer=regularizers.l2(0.01), name = 'hl3')(intermediate)
    intermediate = Dense(1, activation='linear', name = 'score')(intermediate)
    return Model(x_input, intermediate)

def deviation_loss(y_true, y_pred):
    '''
    z-score-based deviation loss
    '''    
    confidence_margin = 500.     
    ref = K.variable(np.random.normal(loc = 0., scale= 1.0, size = 500000) , dtype='float32')
    dev = (y_pred - K.mean(ref)) / K.std(ref)
    inlier_loss = K.abs(dev) 
    outlier_loss = K.abs(K.maximum(confidence_margin - dev, 0.))
    return K.mean((1 - y_true) * inlier_loss + y_true * outlier_loss)

def deviation_network(input_shape, network_depth):
    '''
    only a network with depth 4 is allowed
    '''
    if network_depth == 4:
        model = dev_network_s(input_shape)
    else:
        sys.exit("The network depth is not set properly")
    rms = RMSprop(clipnorm=1.)
    #set model loss and optimizer parameters
    model.compile(loss=deviation_loss, optimizer=rms)
    return model


def batch_generator_sup(x, outlier_indices, inlier_indices, batch_size, nb_batch, rng):
    """batch generator
    """
    rng = np.random.RandomState(rng.randint(MAX_INT, size = 1))
    counter = 0
    while 1:                
        if data_format == 0:
            ref, training_labels = input_batch_generation_sup(x, outlier_indices, inlier_indices, batch_size, rng)
        counter += 1
        yield(ref, training_labels)
        if (counter > nb_batch):
            counter = 0

def input_batch_generation_sup(x_train, outlier_indices, inlier_indices, batch_size, rng):
        
    dim = x_train.shape[1]
    ref = np.empty((batch_size, dim))    
    training_labels = []
    n_inliers = len(inlier_indices)
    n_outliers = len(outlier_indices)
    for i in range(batch_size):    
        if(i % 2 == 0):
            sid = rng.choice(n_inliers, 1)
            ref[i] = x_train[inlier_indices[sid]]
            training_labels += [0]
        else:
            sid = rng.choice(n_outliers, 1)
            ref[i] = x_train[outlier_indices[sid]]
            training_labels += [1]
    return np.array(ref), np.array(training_labels)

def inject_noise(seed, n_out, random_seed):   
    '''
    add anomalies to training data
    randomly swape 8% features of anomalies
    '''  
    rng = np.random.RandomState(random_seed) 
    n_sample, dim = seed.shape
    swap_ratio = 0.08
    n_swap_feat = int(swap_ratio * dim)
    noise = np.empty((n_out, dim))
    for i in np.arange(n_out):
        outlier_idx = rng.choice(n_sample, 2, replace = False)
        o1 = seed[outlier_idx[0]]
        o2 = seed[outlier_idx[1]]
        swap_feats = rng.choice(dim, n_swap_feat, replace = False)
        noise[i] = o1.copy()
        noise[i, swap_feats] = o2[swap_feats]
    return noise

data_set='HiggsBoson'
names = data_set.split(',')
names = ['HiggsBoson']

network_depth = 4
random_seed = 42 #random seed chosen as 42

#Empty array definitions to hold values
aucscores = []
aucprscores = []
score_values=[]
scores_m=[]
y_values=[]
total_mse=[]
thresholdvalues=[]

#Loop for the data
for nm in names:
    runs = 5 #Runs chosen as 5 according to cross validation fold
    
    #create numpy arrays with zero to hold value later
    rauc = np.zeros(runs)
    ap = np.zeros(runs)
    filename = nm.strip()
   
    #Rename the variables
    x= np.array(features_array_test)
    labels= np.array(label_array_test)
    
    #Shuffle the data for train and test
    x, labels = shuffle(x, labels, random_state=42)
    
    outlier_indices = np.where(labels == 1)[0] #detect the anomaly indices
    outliers = x[outlier_indices] #detect the features that are anomalies
    n_outliers_org = outliers.shape[0] #detect the number of anomalies
    
    train_time = 0 #set train time as zero
    test_time = 0  #set test time as zero
    
    n_split=5 #number of split
    i=0; # set counter as zero
    tprs = []
    aucs = []
    
    #Define some necessary variables for plotting
    mean_fpr = np.linspace(0, 1, 100)
    fig, ax = plt.subplots()
    
    for train_index,test_index in KFold(n_split).split(x): #split data based on x data Samples (cross validation loop)
        i=i+1 #increase counter by one
        
        x_train,x_test=x[train_index],x[test_index] #split data as train and test
        y_train,y_test=labels[train_index],labels[test_index] #split data as train and test
        print(np.count_nonzero(y_test)) #print the number of anomaly to check for test data
        
        #assign train and test data into the numpy array
        y_train = np.array(y_train)
        y_test = np.array(y_test)
        print(filename + ': round ' + str(i)) #print the number of iteration
        outlier_indices = np.where(y_train == 1)[0] #detect the anomaly indices
        inlier_indices = np.where(y_train == 0)[0] #detect the normal indices
        n_outliers = len(outlier_indices) #count the total anomaly number
        
        #Print the training size and the number of anomalies        
        print("Original training size: %d, No. outliers: %d" % (x_train.shape[0], n_outliers))
        
        #determine the number of noise by using the normal number
        n_noise  = len(np.where(y_train == 0)[0]) * 0.02 / (1. - 0.02)
        n_noise = int(n_noise) #convert number of noise to integer
        
        #Create a variable to generate and hold random numbers
        rng = np.random.RandomState(random_seed)
        
        #Decrease the number of anomaly to the desired value
        if n_outliers > 50000:
            mn = n_outliers - 50000
            remove_idx = rng.choice(outlier_indices, mn, replace=False)
            x_train = np.delete(x_train, remove_idx, axis=0)
            y_train = np.delete(y_train, remove_idx, axis=0)
         
        #Add noises(anomalies) to replicate anomaly contaminated dataset
        noises = inject_noise(outliers, n_noise, random_seed)
        x_train = np.append(x_train, noises, axis = 0)
        y_train = np.append(y_train, np.zeros((noises.shape[0], 1)))
        
        outlier_indices = np.where(y_train == 1)[0] #detect the anomaly indices again
        inlier_indices = np.where(y_train == 0)[0] #detect the normal indices again
        print(y_train.shape[0], outlier_indices.shape[0], inlier_indices.shape[0])
        input_shape = x_train.shape[1:]
        n_samples_trn = x_train.shape[0]
        n_outliers = len(outlier_indices) #count the number of anomaly
        
        #Print the train size and number of anomaly
        print("Training data size: %d, No. outliers: %d" % (x_train.shape[0], n_outliers))
        
        start_time = time.time() #define a variable to calculate elapsed time
        epochs = 50 #set number of epochs as 50
        batch_size = 512 #set batch_size as 512
        nb_batch = 5000 #set nb_batch as 5000
        
        #define the model
        model = deviation_network(input_shape, network_depth)
        print(model.summary())
        
        #Training the data with necessary parameters
        model.fit_generator(batch_generator_sup(x_train, outlier_indices, inlier_indices, batch_size, nb_batch, rng),
        steps_per_epoch = nb_batch,
        epochs = epochs)
        
        model.save(str(i)+"-90DevNetwork") #Save trained model
        #model = load_model(str(i)+"-90DevNetwork", custom_objects={'deviation_loss': deviation_loss}) #load trained model
        
        scores = model.predict(x_test) #Predictions with test data
        
        #Calculate train and test time
        train_time += time.time() - start_time
        start_time = time.time()
        test_time += time.time() - start_time
        
        rauc[i-1], ap[i-1] = aucPerformance(scores, y_test) #aucperformance
        
        #necessary array operations to adjust the appropriate shapes
        scores=np.reshape(scores, (np.size(x_test,0), ))
        scores=np.transpose(scores)
        scores_mean=np.mean(scores)
        scores_m.append(scores_mean)
        score_values.append(scores)
        
        #necessary array operations to adjust the appropriate shapes
        y_test=np.transpose(y_test)
        y_test=np.reshape(y_test, (np.size(x_test,0), ))
        y_values.append(y_test)
        
        #Define a dataframe to hold scores and y_test values
        error_df = pd.DataFrame({'AUC_Scores': scores, 'true_class': y_test})
        
        #By using the roc_curve function; fpr, tpr and thresholds are obtained
        fpr, tpr, thresholds = roc_curve(error_df.true_class, error_df.AUC_Scores)
        
        #obtaine aucpr scores
        ap1 = average_precision_score(y_test, scores)
        
        #obtaine auc roc scores
        roc_auc = auc(fpr, tpr)
        
        #plot roc curve for the each cross validation fold in single plot as grey color
        ax.plot(fpr, tpr, color='grey', label='ROC fold {} (AUC = %0.3f)'.format(i)% roc_auc, alpha=0.2)
        
        #hold aucpr scores in an array
        aucprscores.append(ap1 * 100)
        #hold auc scores in an array
        aucscores.append(roc_auc * 100)
        
        #Necessary operations for the plotting
        interp_tpr = interp(mean_fpr, fpr, tpr)
        interp_tpr[0] = 0.0
        tprs.append(interp_tpr)
        aucs.append(roc_auc)
        
        interp_tpr = interp(mean_fpr, fpr, tpr)
        error_df.describe()

mean_auc = np.mean(rauc) #Calculate mean of the auc scores
std_auc = np.std(rauc)   #Calculate standard deviation of the auc scores
mean_aucpr = np.mean(ap) #Calculate mean of the auc-pr scores  
std_aucpr = np.std(ap)   #Calculate stardard deviation of the auc-pr scores
train_time = train_time/runs #Calculate average train time
test_time = test_time/runs   #Calculate average test time

#Print mean auc, mean aucpr and train-test times
print("average AUC-ROC: %.4f, average AUC-PR: %.4f" % (mean_auc, mean_aucpr))
print("average runtime: %.4f seconds" % (train_time + test_time))
#Center line for the roc curve
ax.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
        label='Chance', alpha=.8)
#Obtaine the mean tpr value for all cross validations
mean_tpr = np.mean(tprs, axis=0)
mean_tpr[-1] = 1.0
#mean_auc = auc(mean_fpr, mean_tpr)

#Plot the mean auc roc curve
ax.plot(mean_fpr, mean_tpr, color='b',
        label=r'Mean ROC (AUC = %0.3f $\pm$ %0.2f)' % (mean_auc, std_auc),
        lw=2, alpha=.8)
std_tpr = np.std(tprs, axis=0)

#Obtaine the standard deviation of tprs
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