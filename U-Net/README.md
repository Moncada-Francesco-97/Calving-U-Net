# U-Net architecture

This directory contains the code used to create, train, and test a Convolutional Neural Network (CNN) using the U-Net architecture with attention blocks. The operations are performed on the prepared dataset. The code is structured to train five different models, each trained on a different subset of the training set derived from a cross-validation procedure.

### Training Strategy

The training dataset was divided according to the block strategy recommended by [Roberts (2016)](https://www.researchgate.net/publication/311523792_Cross-validation_strategies_for_data_with_temporal_spatial_hierarchical_or_phylogenetic_structure) for cross-validation. We opted for a temporal division, setting aside two years for each ice shelf as a testing dataset. Additionally, the [functions_architecture.py](functions_architecture.py) file includes a function to divide the dataset to preserve spatial variability.

### File Descriptions

- **[functions_architecture.py](functions_architecture.py)**:
  - Contains functions to properly load the dataset.
  - Implements a personalized loss function based on the Dice loss function.
  - Defines the architecture of the U-Net model with attention blocks.

- **[u_net.py](u_net.py)**:
  - Performs cross-validation, training, and model initialization.
  - Allows setting of all model hyperparameters.

- **[conda_environment.txt](conda_environment.txt)**:
  - Contains the conda environment specifications used to develop the code.
