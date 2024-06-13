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

- **[try_model.py](try_model.py)**:
  - This file was created to test if the architecture was working. It train the model on a subset of the training test.

- **[conda_environment.txt](conda_environment.txt)**:
  - Contains the conda environment specifications used to develop the code.
 
- **[make_job.py](make_job.py)**:
  - Script to launch the job on the cluster

### Perspectives

At the current state, the model is not able to predict large calving events. For furure developement, two possible implementations are possible:

- **Edge-aware Attention**: Develop a specialized attention module within the U-Net that highlights edge features in the feature maps. This module should learn to focus on regions with edges and reduce the emphasis on less relevant background areas. For reference check out [Edge-aware U-Nets](https://www.sciencedirect.com/science/article/pii/S1746809421010697).

- **Edge Detection Loss**: Add an edge detection loss component in addition to the segmentation loss to explicitly train the model to detect edges. This loss function should penalize differences between the predicted and ground truth edge maps, encouraging the model to improve edge localization accuracy. For reference check out [Edge Detection Loss](https://pdf.sciencedirectassets.com/777839/1-s2.0-S2666827021X00059/1-s2.0-S26668270210010[…]uY29t&ua=080c5f5f065702040102&rr=88f8deae2eac0b3a&cc=nl).
