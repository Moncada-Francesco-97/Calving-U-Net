# Import necessary Python packages 
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
import tensorflow as tf
#import keras

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

#Load Dataset
#resized_features.npy'

features_path = '/bettik/moncadaf/data/outputs/machine_learning_calving_project/cnn_dataset/normalised_dataset/features.npy' 
features = np.load(features_path, allow_pickle=True)

targets_path = '//bettik/moncadaf/data/outputs/machine_learning_calving_project/cnn_dataset/normalised_dataset/target.npy' 
targets = np.load(targets_path, allow_pickle=True)

print('Data loaded')

#Easy split of the data
# from sklearn.model_selection import train_test_split #3 mins

# print('I have imported sklearn')
# X_train, X_test, y_train, y_test = train_test_split(features, targets, test_size=0.2, random_state=42)
# X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

#Block split of the data

#I want to create an index which has the same lenght of the target array, but is 1 for the first 11 elements, 2 for the following 11 elements, and so on
index = np.zeros(len(features))
for i in range(len(features)):
    index[i] = int(i/11)+1

#I want to create an index which has the same lenght of the target array, which goes from 2005 to 2015, adn then repeat till the end
years = np.zeros(len(targets))
for i in range(len(targets)):
    years[i] = 2005 + i%11

index_tot = np.concatenate((index.reshape(-1,1).astype(int), years.reshape(-1,1).astype(int)), axis=1)

X_train = []
y_train = []
X_test = []
y_test = []

for sample in range(len(features)):    
    region, year = index_tot[sample]

    if region in [1,8,9,12,17,21]:
        y_test.append(targets[sample])
        X_test.append(features[sample])
    else:
        y_train.append(targets[sample])
        X_train.append(features[sample])
        
X_train = np.array(X_train)
y_train = np.array(y_train)
X_test = np.array(X_test)
y_test = np.array(y_test)

print('Data split, the shape of the training data is:', X_train.shape, 'and the shape of the test data is:', X_test.shape)
print('The shape of the training target is:', y_train.shape, 'and the shape of the test target is:', y_test.shape)



saving_dir = '/bettik/moncadaf/data/outputs/machine_learning_calving_project/cnn_dataset/normalised_dataset/'

# np.save(saving_dir + 'X_train.npy', X_train)
# np.save(saving_dir + 'X_test.npy', X_test)
# np.save(saving_dir + 'y_train.npy', y_train)
# np.save(saving_dir + 'y_test.npy', y_test)


# print('The X_train shape is:', X_train.shape)
# print('The X_val shape is:', X_val.shape)
# print('The X_test shape is:', X_test.shape)



window_size = 1024 #256
n_regions = 28
n_feature_variables = 6
n_target_variables = 1
n_years = 11

# Metrics
METRICS = [
      metrics.TruePositives(name='tp'),
      metrics.FalsePositives(name='fp'),
      metrics.TrueNegatives(name='tn'),
      metrics.FalseNegatives(name='fn'),
      metrics.BinaryAccuracy(name='accuracy'),
      metrics.Precision(name='precision'),
      metrics.Recall(name='recall'),
      metrics.AUC(name='auc'),
      metrics.AUC(name='prc', curve='PR')]

Input_shapes = (window_size,window_size,n_feature_variables)
optimizer = 'ADAM'
loss = 'binary_crossentropy'


#Easy implementation of U-Net model

def unet_model(input_shape=(window_size, window_size, n_feature_variables)):
    inputs = keras.Input(shape=input_shape)

    # Encoder First argument is the number of output channels(filters), the second is the kernel size.
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

# Create the U-Net model
print('Before declaring the model')
model = unet_model()

model.summary()

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

print('Model compiled, ready to train')

#history = model.fit(X_train, y_train, batch_size = 10, epochs = 10, validation_data = (X_val, y_val), verbose = 1)
history = model.fit(X_train, y_train, batch_size = 10, epochs = 10, validation_split = 0.2, verbose = 1)

# Save the model
model.save('/bettik/moncadaf/data/outputs/machine_learning_calving_project/model_architecture/model_v_001.h5')
model.save('/bettik/moncadaf/data/outputs/machine_learning_calving_project/model_architecture/model_v_001.keras')
print('Model saved')

print('Ready to predict')
prediction = model.predict(X_test, verbose = 1)

# Evaluate the model
loss, accuracy = model.evaluate(X_test, y_test, verbose=1)

print('Test loss:', loss)
print('Test accuracy:', accuracy)

#save the OUTPUTS
np.save(saving_dir + 'predictions_v_001.npy', prediction)
