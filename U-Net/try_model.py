import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colors
import matplotlib.colors as clr
import pandas as pd
import os
#import tensorrt
import tensorflow as tf
from tensorflow import keras

from tensorflow.keras import layers
from tensorflow.keras import losses
from tensorflow.keras import models
from tensorflow.keras import metrics
from tensorflow.keras import optimizers

#From my functions
from functions_architectures import generate_index
from functions_architectures import unet_model
from functions_architectures import spatial_folders
from functions_architectures import temporal_folders
from functions_architectures import diceloss
from functions_architectures import unet_model_attention


#Specifications for model input parameters
window_size = 1024 #256
n_regions = 28
n_feature_variables = 6
n_target_variables = 1
n_years = 11
n_epochs = 2

Input_shapes = (window_size,window_size,n_feature_variables)

directory_model_path = '/bettik/moncadaf/data/outputs/machine_learning_calving_project/model_architecture/'

weights = []

# for i in [5]:

#     print('I am loading the model: ', i)
    
#     model = unet_model(window_size, n_feature_variables)

#     model.load_weights(directory_model_path + 'weights_' + str(i) + '_temporal_prova_cv_conv7_relu.weights.h5')

#     weights.append(model.get_weights())



# averaged_weights = []

# #This is necessary to make the weights of the models compatible between each other
# for weight_tensors in zip(*weights):
#     # Convert each weight tensor to NumPy array and calculate mean
#     averaged_tensor = np.mean([np.array(w) for w in weight_tensors], axis=0)
#     averaged_weights.append(averaged_tensor)

# #Set the model with the averaged weights
# model_try = unet_model(window_size, n_feature_variables)

# model_try.set_weights(averaged_weights)

model = unet_model_attention(window_size, n_feature_variables)
model.compile(optimizer='adam', loss=diceloss, metrics=['accuracy', metrics.Precision()])


print('I have loaded and initialised the model')

features_path = '/bettik/moncadaf/data/outputs/machine_learning_calving_project/cnn_dataset/normalised_dataset/features.npy' 
features = np.load(features_path, allow_pickle=True)

targets_path = '/bettik/moncadaf/data/outputs/machine_learning_calving_project/cnn_dataset/normalised_dataset/target.npy' 
targets = np.load(targets_path, allow_pickle=True)

print('Data loaded')

n_features = len(features)
n_targets = len(targets)

#This function generates the indexes for the cross_validation
index_tot = generate_index(n_features, n_targets)
print('Indexes generated')
cv_features, cv_targets, X_test, y_test = temporal_folders(features, targets, index_tot)
print('Data splitted')

i =0

X_val_test  = np.array(cv_features[i])
y_val_test = np.array(cv_targets[i])

X_val_train = np.array(np.concatenate([cv_features[j] for j in range(len(cv_features)) if j != i]))
y_val_train = np.array(np.concatenate([cv_targets[j] for j in range(len(cv_targets)) if j != i]))

print('Next step is training the model')

history = model.fit(X_val_train, y_val_train, batch_size = 20, epochs = n_epochs, verbose = 1, validation_data = (X_val_test, y_val_test))


#  

# #calculate the loss
# for image in range(0, len(y_pred)):
#     loss = diceloss(y_test[image,:,:,0], y_pred[image,:,:,0])
#     print('Loss: ', loss)

#save the predictions
# cnn_dataset_path = '/bettik/moncadaf/data/outputs/machine_learning_calving_project/cnn_dataset/normalised_dataset/'
# np.save(cnn_dataset_path + 'y_pred_avg_temporal_prova_cv_conv7_relu_outputs_sigmoid.weights.npy', y_pred)
# np.save(cnn_dataset_path + 'y_test_avg_temporal_prova_cv_conv7_relu_outputs_sigmoid.weights.npy', y_test)

print('Is working!')

