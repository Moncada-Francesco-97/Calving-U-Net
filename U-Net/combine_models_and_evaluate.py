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


#Specifications for model input parameters
window_size = 1024 #256
n_regions = 28
n_feature_variables = 6
n_target_variables = 1
n_years = 11

Input_shapes = (window_size,window_size,n_feature_variables)
optimizer = 'ADAM'
loss = 'binary_crossentropy'


def unet_model(input_shape=(window_size, window_size, n_feature_variables)):
    inputs = keras.Input(shape=input_shape)

    # Encoder 
    # First argument is the number of output channels(filters), the second is the kernel size.
    conv1 = layers.Conv2D(64, 3, activation='relu', padding='same')(inputs)
    conv1 = layers.Conv2D(64, 3, activation='relu', padding='same')(conv1)
    pool1 = layers.MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = layers.Conv2D(128, 3, activation='relu', padding='same')(pool1)
    conv2 = layers.Conv2D(128, 3, activation='relu', padding='same')(conv2)
    pool2 = layers.MaxPooling2D(pool_size=(2, 2))(conv2)

    # Bottleneck 
    conv3 = layers.Conv2D(256, 3, activation='relu', padding='same')(pool2)
    conv3 = layers.Conv2D(256, 3, activation='relu', padding='same')(conv3)

    # Decoder
    up4 = layers.UpSampling2D(size=(2, 2))(conv3)
    concat4 = layers.Concatenate(axis=-1)([conv2, up4])
    conv4 = layers.Conv2D(128, 3, activation='relu', padding='same')(concat4)
    conv4 = layers.Conv2D(128, 3, activation='relu', padding='same')(conv4)

    up5 = layers.UpSampling2D(size=(2, 2))(conv4)
    concat5 = layers.Concatenate(axis=-1)([conv1, up5])
    conv5 = layers.Conv2D(64, 3, activation='relu', padding='same')(concat5)
    conv5 = layers.Conv2D(64, 3, activation='relu', padding='same')(conv5)

    # Output layer
    outputs = layers.Conv2D(1, 1, activation='sigmoid')(conv5)

    model = keras.Model(inputs=inputs, outputs=outputs)
    return model

directory_model_path = '/bettik/moncadaf/data/outputs/machine_learning_calving_project/model_architecture/'

# Load the models
model_1 = tf.keras.models.load_model(directory_model_path + 'model_1.h5')
model_2 = tf.keras.models.load_model(directory_model_path + 'model_2.h5')
model_3 = tf.keras.models.load_model(directory_model_path + 'model_3.h5')
model_4 = tf.keras.models.load_model(directory_model_path + 'model_4.h5')
model_5 = tf.keras.models.load_model(directory_model_path + 'model_5.h5')

models = [model_1, model_2, model_3, model_4, model_5]
print('I have loaded the models')

#Get the weights of the models
weight_1 = model_1.get_weights()
weight_2 = model_2.get_weights()
weight_3 = model_3.get_weights()
weight_4 = model_4.get_weights()
weight_5 = model_5.get_weights()

weights = [weight_1, weight_2, weight_3, weight_4, weight_5]
averaged_weights = []

#This is necessary to make the weights of the models compatible between each other
for weight_tensors in zip(*weights):
    # Convert each weight tensor to NumPy array and calculate mean
    averaged_tensor = np.mean([np.array(w) for w in weight_tensors], axis=0)
    averaged_weights.append(averaged_tensor)
    
model_averaged = unet_model()
model_averaged.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])
model_averaged.set_weights(averaged_weights)

print('I have combined the models')

features_path = '/bettik/moncadaf/data/outputs/machine_learning_calving_project/cnn_dataset/normalised_dataset/features.npy' 
features = np.load(features_path, allow_pickle=True)

targets_path = '//bettik/moncadaf/data/outputs/machine_learning_calving_project/cnn_dataset/normalised_dataset/target.npy' 
targets = np.load(targets_path, allow_pickle=True)

print('Data loaded')

#Index which has the same lenght of the target array, is 1 for the first 11 elements, 2 for the following 11 elements, and so on
index = np.zeros(len(features))
for i in range(len(features)):
    index[i] = int(i/11)+1

#Index which has the same lenght of the target array, which goes from 2005 to 2015, adn then repeat till the end
years = np.zeros(len(targets))
for i in range(len(targets)):
    years[i] = 2005 + i%11

index_tot = np.concatenate((index.reshape(-1,1).astype(int), years.reshape(-1,1).astype(int)), axis=1)


X_test = []
y_test = []


#Initialise the blocks. The ice shelves are distributed geographically
for sample in range(len(features)):    
    region, year = index_tot[sample]

    if region in [1,8,9,12,17,21]:
        y_test.append(targets[sample])
        X_test.append(features[sample])
print('Ready to test the model')


from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix


#Apllly the model to the test data and evaluate it
X_test = np.array(X_test)
y_test = np.array(y_test)

y_pred = model_averaged.predict(X_test)

#Check if there are nans in the predictions
print(np.isnan(y_pred).any())

print(np.shape(y_pred))
print(np.shape(y_test))
print(type(y_pred))
print(type(y_test))

loss, accuracy = model_averaged.evaluate(X_test, y_test, verbose=0)

print('Loss: ', loss)
print('Accuracy: ', accuracy)

#save the y_test and y_prediction
cnn_dataset_path = '/bettik/moncadaf/data/outputs/machine_learning_calving_project/cnn_dataset/normalised_dataset/'

np.save(cnn_dataset_path + 'y_test_avg.npy', y_test)
np.save(cnn_dataset_path + 'y_pred_avg.npy', y_pred)

print('I have saved the predictions and the test data')
# #flatten the arrays
# y_pred = y_pred.flatten()
# y_test = y_test.flatten()

# precision = precision_score(y_test, y_pred)
# recall = recall_score(y_test, y_pred)
# f1 = f1_score(y_test, y_pred)


# #print('Precision: ', precision)
# print('Recall: ', recall)
# print('F1: ', f1)

# cm = confusion_matrix(y_test, y_pred)

# sav_file = '/bettik/moncadaf/data/outputs/machine_learning_calving_project/model_architecture/'

# #save the matrix
# np.save(sav_file + 'confusion_matrix.npy', cm)


