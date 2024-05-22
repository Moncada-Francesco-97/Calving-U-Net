# unet_model.py

from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import tensorflow as tf




# Function to create a U-Net model
# Input: window_size: int, n_feature_variables: int
# Output: model: keras.Model

def unet_model(window_size, n_feature_variables):

    input_shape=(window_size, window_size, n_feature_variables)
    inputs = keras.Input(shape=input_shape)

    # Encoder 
    conv0 = layers.Conv2D(32, 3, activation='relu', padding='same')(inputs)
    conv0 = layers.Conv2D(32, 3, activation='relu', padding='same')(conv0)
    pool0 = layers.MaxPooling2D(pool_size=(2, 2))(conv0)

    # First argument is the number of output channels(filters), the second is the kernel size.
    conv1 = layers.Conv2D(64, 3, activation='relu', padding='same')(pool0)
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

    up6 = layers.UpSampling2D(size=(2, 2))(conv5)
    concat6 = layers.Concatenate(axis=-1)([conv0, up6])
    conv6 = layers.Conv2D(32, 3, activation='relu', padding='same')(concat6)
    conv6 = layers.Conv2D(32, 3, activation='relu', padding='same')(conv6)

    # Additional layer with ReLU activation
    conv7 = layers.Conv2D(32, 3, activation='relu', padding='same')(conv6)

    # Output layer with sigmoid activation
    outputs = layers.Conv2D(1, 1, activation='sigmoid')(conv7)

    model = keras.Model(inputs=inputs, outputs=outputs)
    return model



#Create Folders for spatial cross validation
#Features and targets are lists of numpy arrays
#Index_tot is a numpy array with the region and year of each sample

def spatial_folders(features, targets, index_tot):
    X_test = []
    y_test = []

    cv_1_features, cv_1_targets = [], []
    cv_2_features, cv_2_targets = [], []
    cv_3_features, cv_3_targets = [], []
    cv_4_features, cv_4_targets = [], []
    cv_5_features, cv_5_targets = [], []

    for sample in range(len(features)):    
        region, year = index_tot[sample]

        if region in [1,8,9,12,17,21]:
            y_test.append(targets[sample])
            X_test.append(features[sample])
        if region in [3,4,15,25]:
            cv_1_features.append(features[sample])
            cv_1_targets.append(targets[sample])
        if region in [5,18,20,23]:
            cv_2_features.append(features[sample])
            cv_2_targets.append(targets[sample])
        if region in [2,6,13,22]:
            cv_3_features.append(features[sample])
            cv_3_targets.append(targets[sample])
        if region in [7,10,14, 24, 26]:
            cv_4_features.append(features[sample])
            cv_4_targets.append(targets[sample])
        if region in [11,16,19,27,28]:
            cv_5_features.append(features[sample])
            cv_5_targets.append(targets[sample])

    cv_features = [cv_1_features, cv_2_features, cv_3_features, cv_4_features, cv_5_features]
    cv_targets = [cv_1_targets, cv_2_targets, cv_3_targets, cv_4_targets, cv_5_targets]

    return cv_features, cv_targets, X_test, y_test



# Function to split the data for temporal cross validation
# Input: features: list of numpy arrays, targets: list of numpy arrays, index_tot: numpy array
# Output: cv_features: list of lists of numpy arrays, cv_targets: list of lists of numpy arrays, X_test: list of numpy arrays, y_test: list of numpy arrays

def temporal_folders(features, targets, index_tot):
    X_test = []
    y_test = []

    cv_1_features, cv_1_targets = [], []
    cv_2_features, cv_2_targets = [], []
    cv_3_features, cv_3_targets = [], []
    cv_4_features, cv_4_targets = [], []
    cv_5_features, cv_5_targets = [], []

    for sample in range(len(targets)):
        region, year = index_tot[sample]

        if year in [2008, 2012]:
            y_test.append(targets[sample])
            X_test.append(features[sample])
        if year in [2005, 2011]:
            cv_1_features.append(features[sample])
            cv_1_targets.append(targets[sample])
        if year in [2006, 2013]:
            cv_2_features.append(features[sample])
            cv_2_targets.append(targets[sample])
        if year in [2007, 2014]:
            cv_3_features.append(features[sample])
            cv_3_targets.append(targets[sample])
        if year in [2009, 2015]:
            cv_4_features.append(features[sample])
            cv_4_targets.append(targets[sample])
        if year in [2010]:
            cv_5_features.append(features[sample])
            cv_5_targets.append(targets[sample])

    cv_features = [cv_1_features, cv_2_features, cv_3_features, cv_4_features, cv_5_features]
    cv_targets = [cv_1_targets, cv_2_targets, cv_3_targets, cv_4_targets, cv_5_targets]

    return cv_features, cv_targets, X_test, y_test





# Function to generate the index for the spatial cross validation
# Input: number_features: int, number_targets: int
# Output: index_tot: numpy array, fulled with arrays of region and year [[region, year], [region, year], ...]

def generate_index(number_features, number_targets):
    index = np.zeros(number_features)
    for i in range(number_features):
        index[i] = int(i / 11) + 1

    years = np.zeros(number_targets)
    for i in range(number_targets):
        years[i] = 2005 + i % 11

    index_tot = np.concatenate((index.reshape(-1, 1).astype(int), years.reshape(-1, 1).astype(int)), axis=1)
    return index_tot

#DICE LOSS CALCULATES THE DICE COEFFICIENT BETWEEN TWO TENSORS
#Input is the true mask and the predicted mask
#Output is the dice coefficient (a float)
def diceloss(y_true, y_pred):
    smooth = 1e-6
    y_true, y_pred = tf.cast(
        y_true, dtype=tf.float32), tf.cast(y_pred, tf.float32)
    nominator = 2 * tf.reduce_sum((tf.multiply(y_pred, y_true)))
    denominator = tf.reduce_sum(y_pred) + tf.reduce_sum(y_true) + smooth
    result = 1 - tf.divide(nominator, denominator)
    return result

from tensorflow.keras import layers, Model, Input

def attention_block(F_g, F_l, F_int, bn=False):
    g = layers.Conv2D(F_int, kernel_size=(1, 1), strides=(1, 1), padding='valid')(F_g)
    x = layers.Conv2D(F_int, kernel_size=(1, 1), strides=(1, 1), padding='valid')(F_l)
    psi = layers.Add()([g, x])
    psi = layers.Activation('relu')(psi)

    psi = layers.Conv2D(1, kernel_size=(1, 1), strides=(1, 1), padding='valid')(psi)
    psi = layers.Activation('sigmoid')(psi)

    return layers.Multiply()([F_l, psi])

def unet_model_attention(window_size, n_feature_variables):
    input_shape = (window_size, window_size, n_feature_variables)
    inputs = Input(shape=input_shape)

    # Encoder 
    conv0 = layers.Conv2D(32, 3, activation='relu', padding='same')(inputs)
    conv0 = layers.Conv2D(32, 3, activation='relu', padding='same')(conv0)
    pool0 = layers.MaxPooling2D(pool_size=(2, 2))(conv0)

    conv1 = layers.Conv2D(64, 3, activation='relu', padding='same')(pool0)
    conv1 = layers.Conv2D(64, 3, activation='relu', padding='same')(conv1)
    pool1 = layers.MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = layers.Conv2D(128, 3, activation='relu', padding='same')(pool1)
    conv2 = layers.Conv2D(128, 3, activation='relu', padding='same')(conv2)
    pool2 = layers.MaxPooling2D(pool_size=(2, 2))(conv2)

    # Bottleneck 
    conv3 = layers.Conv2D(256, 3, activation='relu', padding='same')(pool2)
    conv3 = layers.Conv2D(256, 3, activation='relu', padding='same')(conv3)

    # Decoder with attention
    up4 = layers.UpSampling2D(size=(2, 2))(conv3)
    att4 = attention_block(up4, conv2, 128)
    concat4 = layers.Concatenate(axis=-1)([att4, up4])
    conv4 = layers.Conv2D(128, 3, activation='relu', padding='same')(concat4)
    conv4 = layers.Conv2D(128, 3, activation='relu', padding='same')(conv4)

    up5 = layers.UpSampling2D(size=(2, 2))(conv4)
    att5 = attention_block(up5, conv1, 64)
    concat5 = layers.Concatenate(axis=-1)([att5, up5])
    conv5 = layers.Conv2D(64, 3, activation='relu', padding='same')(concat5)
    conv5 = layers.Conv2D(64, 3, activation='relu', padding='same')(conv5)

    up6 = layers.UpSampling2D(size=(2, 2))(conv5)
    att6 = attention_block(up6, conv0, 32)
    concat6 = layers.Concatenate(axis=-1)([att6, up6])
    conv6 = layers.Conv2D(32, 3, activation='relu', padding='same')(concat6)
    conv6 = layers.Conv2D(32, 3, activation='relu', padding='same')(conv6)

    # Additional layer with ReLU activation
    conv7 = layers.Conv2D(32, 3, activation='relu', padding='same')(conv6)

    # Output layer with sigmoid activation
    outputs = layers.Conv2D(1, 1, activation='sigmoid')(conv7)

    model = Model(inputs=inputs, outputs=outputs)
    return model





