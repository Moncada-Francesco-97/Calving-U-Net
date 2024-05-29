# Overview (README IS WORK IN PROGRESS 29.05.24!)

<a name="regions"></a>
<img width="811" alt="regions" src="https://github.com/Moncada-Francesco-97/machine_learning_calving_project/assets/110817494/88f24c62-6094-4ae4-ac6c-341588c154df">



This repository provides the code used for the creation of an Antarctic dataset containings the physical variables of **basal melting**, **ice thickness**, **sea ice concentration**, **velocity components** and **ice shelves shape** in 36 different regions of interest located mainly on the Antarctic coasts. The produced dataset has a spatial resolution of 500m and a yearly temporal resolution, ranging from 2005 till 2016, and is the result of a standardization and interpolation process applied to different remote sensing Antarctic datasets representing the current state of the art. Specifically, ice thickness and basal melting were derived from the work of [F.Paolo et al. (2023)](https://tc.copernicus.org/articles/17/3409/2023/), sea ice concentration from [NOAA/NSIDC Climate Data Record of Passive Microwave Sea Ice Concentration, Version 4](https://nsidc.org/data/g02202/versions/4), ice velocity from both [Continent-Wide, Interferometric SAR Phase, Mapping of Antarctic Ice Velocity](https://agupubs.onlinelibrary.wiley.com/doi/10.1029/2019GL083826) and [MEaSUREs InSAR-Based Antarctica Ice Velocity Map, Version 2](https://nsidc.org/data/nsidc-0484/versions/2). The 2-D Ice shelfs' geometry was derived by combining [BedMachine version 3](https://nsidc.org/data/nsidc-0756/versions/3) dataset and the annual coasline evolution dataset released by [Greene et. al (2022)](https://zenodo.org/records/5903643). The two latter datasets allowed to attribute whether each pixel was filled by floating ice, grounded ice, land or sea during the different years, and store this information in the dataset.

The dataset was firstly used to train a CNN model to predict the yearly calving front advance or retreat, providing as input to the model the variables of the previous year. The architecture used is the u-net, with the implementation of attention modules.




## How to use.

#Add image of the regions and esplication about how to download

## Code

### [Cnn dataset preparation](./cnn_dataset_preparation/)

In this directory is stored the code to make the dataset. The [README](cnn_dataset_preparation/README.md) file included the directory describes all the code.


### [U-Net](./U-Net/)

In this directory is stored the code used to create the u-net model architecture, build and test the model. The [README](U-Net/README.md) file included the directory describes all the code.

