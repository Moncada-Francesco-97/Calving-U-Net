import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
import os
import importlib
import cv2

#In this script we resize the images to a new shape, in this case 256x256

def resising_images(image, new_shape):
    new_image = cv2.resize(image, new_shape, interpolation = cv2.INTER_NEAREST)
    return new_image

#Loading Original Dataset
features_path = '/bettik/moncadaf/data/outputs/machine_learning_calving_project/cnn_dataset/normalised_dataset/features.npy' #Change in cluster
features = np.load(features_path, allow_pickle=True)

targets_path = '/bettik/moncadaf/data/outputs/machine_learning_calving_project/cnn_dataset/normalised_dataset/target.npy' #Change in cluster
targets = np.load(targets_path, allow_pickle=True)

#RESISING FEATURES

# Initialize resized_features array with the correct shape
resized_features = np.zeros((len(features), 256, 256, features.shape[3]))

# Iterate over each sample in the features array and resize each channel of the image
for i in range(len(features)):
    for j in range(features.shape[3]):
        resized_features[i, :, :, j] = resising_images(features[i, :, :, j], (256,256))


#RESISING TARGETS

# Initialize resized_targets array with the correct shape
resized_targets = np.zeros((len(targets), 256, 256, targets.shape[3]))

for i in range(len(targets)):
    for j in range(targets.shape[3]):
        resized_targets[i, :, :, j] = resising_images(targets[i, :, :, j], (256,256))

np.save('/bettik/moncadaf/data/outputs/machine_learning_calving_project/cnn_dataset/normalised_dataset/resized_features.npy', resized_features)
np.save('/bettik/moncadaf/data/outputs/machine_learning_calving_project/cnn_dataset/normalised_dataset/resized_targets.npy', resized_targets)