# CNN Dataset Preparation

![dataset_iter](https://github.com/Moncada-Francesco-97/machine_learning_calving_project/assets/110817494/9036a482-f402-4be7-b83d-6b3234ef24bd)


In this directory is stored the code to create the dataset.

### File Descriptions

- **[creating_mask.py](creating_masks.py)**:
  - Creates regional masks for floating ice, sea and grounded ice.
  - Requires the shapefile with the locations of the regions and the TIFF file containing the ice shelf geometry.

- **[create_dataset.py](create_dataset.py)**:
  - Launches the jobs to create the dataset.

- **[bath_region.py](bath_region.py)**:
  - Called by [create_dataset.py](create_dataset.py).
  - Loads functions to create the basal melting, ice velocity, ice thickness, and ice velocity datasets.

- **[Preprocessing_data.py](Preprocessing_data.py)**:
  - Reshapes and normalizes the dataset to prepare it for machine learning pipelines.
  - Maps positive variables (ice thickness and sea ice concentration) to [0,1], and negative ones (ice velocity and basal melting) to [-1,1].
  - The output shape is `[#samples, 1024, 1024, #feature variables]` for the features and `[#samples, 1024, 1024, 1]` for the targets.

- **[basal_melting.py](basal_melting.py), [sea_ice_concentration.py](sea_ice_concentration.py), [velocity.py](velocity.py), [thickness.py](thickness.py)**:
  - Each of those file collects and interpolates its respectively variable. More infos about how datas are merged and about the tecnique adopted for interpolation are reported as comments in each file
  - Requires the shapefile with the locations of the regions and the TIFF files containing the yearly variable data for the entire Antarctic data.
 
  
- **[mask_dataset.py](mask_dataset.py)**:
  - It creates the masks which contains the regional information of the region, basically if a pixel is part of the floating ice, grounded ice, land or sea. Those masks are the same of the one created by [creating_mask.py](creating_mask.py), but saved in a way that is easier to use the data to create the final netcdf files.
 

- **[functions.py](functions.py)**:
  - Contains the functions used in the other files to read the shapefile and load the masks.
 
- **[netcdf_creation.py](netcdf_creation.py)**:
  - It gather the files containing the physical variables and creates the final netcdf file

- **[conda_environment.txt](conda_environment.txt)**:
  - Contains the conda environment specifications used to develop the code.

