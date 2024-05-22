# Overview (README IS WORK IN PROGRESS 16.05.24!)

<a name="regions"></a>
![regions](https://github.com/Moncada-Francesco-97/machine_learning_calving_project/assets/110817494/f288f36c-7ab9-4a0d-abcf-47537b489228)

This repository provides the code used for the creation of the dataset xxxxxxx: a 500m spatial resolution dataset of 5 different physical variables in the 28 sub-regions of the Antarctic continent reported in the figure [above](#regions), with a yearly resolutin which ranges from 2005 till 2016. The variables are **basal melting**, **ice thickness**, **sea ice cocentration**, **velocity components** and **ice shelves shape**. The dataset is the result of a standardisation and interpolation process applied to different dataset which represent the state of the art of remote sensing datasets: ice thickness and basal melting were derived from the work of [F.Paolo](https://tc.copernicus.org/articles/17/3409/2023/), sea ice concentration from [NOAA/NSIDC Climate Data Record of Passive Microwave Sea Ice Concentration, Version 4](https://nsidc.org/data/g02202/versions/4), ice velocity is derived from both [Continent-Wide, Interferometric SAR Phase, Mapping of Antarctic Ice Velocity](https://agupubs.onlinelibrary.wiley.com/doi/10.1029/2019GL083826) and [MEaSUREs InSAR-Based Antarctica Ice Velocity Map, Version 2](https://nsidc.org/data/nsidc-0484/versions/2), while the ice shelves shape is fro the work of XXXXXX.


## How to use.

#Add image of the regions and esplication about how to download

## Code

### [Cnn dataset preparation](./cnn_dataset_preparation/)

In this directory is stored the code to make the dataset. The [README](cnn_dataset_preparation/README.md) file included the directory describes all the code.


### [U-Net](./U-Net/)

In this directory is stored the code used to create the u-net model architecture, build and test the model. The [README](U-Net/README.md) file included the directory describes all the code.

