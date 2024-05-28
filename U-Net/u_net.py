import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colors
import matplotlib.colors as clr
import pandas as pd
import os
import sys
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

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))


if len(sys.argv) < 2:
    print("No arguments provided!")
    sys.exit(1)

    # sys.argv[0] is the script name, so the first argument is sys.argv[1]
model_number = int(sys.argv[1])
print(f"Argument received: {model_number}")

#Load Dataset

features_path = '/bettik/moncadaf/data/outputs/machine_learning_calving_project/cnn_dataset/normalised_dataset/features.npy' 
features = np.load(features_path, allow_pickle=True)

targets_path = '/bettik/moncadaf/data/outputs/machine_learning_calving_project/cnn_dataset/normalised_dataset/target.npy' 
targets = np.load(targets_path, allow_pickle=True)

n_features = len(features)
n_targets = len(targets)

#This function generates the indexes for the cross_validation
index_tot = generate_index(n_features, n_targets)


#IMPORTANT: ONE OF THE TWO FOLLOWING LINE NEED TO BE COMMENTED!!!!!!!!!!!!!!!!
cross_strategy = 'Attention_batch_30_epoch_30_NUMBER_' + str(model_number)
n_epochs = 30
batch_size = 30

#This function generates the temporal folders
cv_features, cv_targets, X_test, y_test = temporal_folders(features, targets, index_tot)

#This function generates the spatial folders
#cv_features, cv_targets, X_test, y_test = spatial_folders(features, targets, index_tot)

print('Data loaded')


#Specifications for model input parameters
window_size = 1024
n_regions = 28
n_feature_variables = 6
n_target_variables = 1
n_years = 11


#Here i perform the cross-validation
# for i in range(len(cv_features)):

#     print('Training model ', i+1)

#     X_val_test  = np.array(cv_features[i])
#     y_val_test = np.array(cv_targets[i])

#     X_val_train = np.array(np.concatenate([cv_features[j] for j in range(len(cv_features)) if j != i]))
#     y_val_train = np.array(np.concatenate([cv_targets[j] for j in range(len(cv_targets)) if j != i]))

#     model = unet_model(window_size,n_feature_variables)
#     model.compile(optimizer='adam',
#                   loss=diceloss,
#                   metrics=['accuracy', metrics.Precision()])

#     print('Model ', i+1,  ' compiled, ready to train, ', cross_strategy)

#     weight = model.get_weights()

#     history = model.fit(X_val_train, y_val_train, batch_size = 20, epochs = n_epochs, validation_data = (X_val_test, y_val_test), verbose = 1)


#     model_dir = '/bettik/moncadaf/data/outputs/machine_learning_calving_project/model_architecture/'

#     #Save the model
#     model.save(model_dir + 'model_' + str(i+1) + cross_strategy +'.h5')

#     #Save the history
#     history_df = pd.DataFrame(history.history)
#     history_df.to_csv(model_dir + 'history_' + str(i+1) + cross_strategy + '.csv')

#     #save th weights
#     model.save_weights(model_dir + 'weights_' + str(i+1) + cross_strategy +'.weights.h5')

#     print('model_' + str(i+1) + cross_strategy +'.h5 saved')

model = unet_model_attention(window_size,n_feature_variables)
model.compile(optimizer='adam', loss=diceloss, metrics=['accuracy', metrics.Precision()])

i = model_number -1

X_val_test  = np.array(cv_features[i])
y_val_test = np.array(cv_targets[i])

print('the lenght of feature is ', len(cv_features))

X_val_train = np.array(np.concatenate([cv_features[j] for j in range(len(cv_features)) if j != i]))
y_val_train = np.array(np.concatenate([cv_targets[j] for j in range(len(cv_targets)) if j != i]))

history = model.fit(X_val_train, y_val_train, batch_size = batch_size, epochs = n_epochs, verbose = 1, validation_data = (X_val_test, y_val_test))

model_dir = '/bettik/moncadaf/data/outputs/machine_learning_calving_project/model_architecture/'

#Save the model
model.save(model_dir + 'model_' + cross_strategy +'.h5')

#Save the history
history_df = pd.DataFrame(history.history)
history_df.to_csv(model_dir + 'history_' + cross_strategy + '.csv')

#save th weights
model.save_weights(model_dir + 'weights_' + cross_strategy +'.weights.h5')

print('model_' + cross_strategy +'.h5 saved')


