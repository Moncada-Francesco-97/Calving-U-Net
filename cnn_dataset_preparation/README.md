# CNN Dataset Preparation

In this directory is stored the code to create the dataset. The .ipynb files were used to test locally the code, while the python ones are used directly on the cluster

[creating_mask.py](creating_masks.py): this file creates the regional masks for floating ice, grounded ice. It requires the shape file with the locations of the regions, and the tif file containing the ice shelf geometry.

[create_dataset.py](create_dataset.py): This file launches the jobs which creates the dataset.

[bath_region.py](bath_region.py): this file is called by the create-dataset.py file. It loads the functions which create the basal melting, ice velocity, ice thickness and ice velocity dataset.

[Preprocessing_data.py](Preprocessing_data.py): This file reshape and normalize the dataset to make it ready for the machine learning pipelines. Just positive variables (ice thickness and sea ice concentration) are mapped to [0,1], the negative ones (ice velocity and basal melting) to [-1,1]. The output shape is [#samples,1024,1024,#feature variables] for the features, and [#samples,1024,1024,1] for the targets.

[basal_melting.py](basal_melting.py), [sea_ice_concentration.py](sea_ice_concentration.py), [velocity.py](velocity.py), [thickness.py](thickness.py) : those files contain the code to respectively create the basal melting, sea ice concentration, velocity and thickness dataset, for each region. Each of thos files require the shape file with the locations of the regions, and the tif files containing the yearly variable data for the  whole Antarctic data.

[functions.py](functions.py): contains the functions used in the other files to read the shapefile and load the masks.
