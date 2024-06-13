# Calving U-Net

This repository provides the code used for creating an Antarctic dataset containing various physical variables across 36 regions of interest, mainly along the Antarctic coasts as reported in the figure below. The dataset, is the result of a rasterisation, standardization and interpolation process of multiple remote sensing datasets, and the analysed variables are:

- **Basal Melting**
- **Ice Thickness**
- **Sea Ice Concentration**
- **Velocity Components**
- **Ice Shelves Shape**

The dataset has a spatial resolution of 500m and a yearly temporal resolution from 2005 to 2016.

<a name="regions"></a>
<img width="811" alt="regions" src="https://github.com/Moncada-Francesco-97/machine_learning_calving_project/assets/110817494/88f24c62-6094-4ae4-ac6c-341588c154df">

## Data Sources
The produced dataset is the result of a standardization and interpolation process applied to different remote sensing Antarctic datasets representing the current state of the art. The rasterred dataset are the followings:
- **Ice Thickness and Basal Melting**: [F.Paolo et al. (2023)](https://tc.copernicus.org/articles/17/3409/2023/)
- **Sea Ice Concentration**: [NOAA/NSIDC Climate Data Record of Passive Microwave Sea Ice Concentration, Version 4](https://nsidc.org/data/g02202/versions/4)
- **Ice Velocity**: 
  - [Continent-Wide, Interferometric SAR Phase, Mapping of Antarctic Ice Velocity](https://agupubs.onlinelibrary.wiley.com/doi/10.1029/2019GL083826)
  - [MEaSUREs InSAR-Based Antarctica Ice Velocity Map, Version 2](https://nsidc.org/data/nsidc-0484/versions/2)
- **Ice Shelves Geometry**: 
  - [BedMachine version 3](https://nsidc.org/data/nsidc-0756/versions/3)
  - [Greene et. al (2022)](https://zenodo.org/records/5903643)

## Applications

The dataset was used to train a Convolutional Neural Network (CNN) model to predict the yearly advance or retreat of the calving front, using the previous year's variables as input. In particular, a U-Net architecture with attention modules was implemented.

## Code

### [Cnn dataset preparation](./cnn_dataset_preparation/)

In this directory is stored the code to make the dataset. The [README](cnn_dataset_preparation/README.md) file included the directory describes all the code.


### [U-Net](./U-Net/)

In this directory is stored the code used to create the u-net model architecture, build and test the model. The [README](U-Net/README.md) file included the directory describes all the code.


## Downloads

#Add instruction about how to download

