import pandas as pd
import fiona
from shapely.geometry import shape
import numpy as np


#Read Shapefiles
def read_shapefile(shape_file, region_id):
    """
    Read a shapefile and extract boundary information.
    
    Args:
    - shape_file (str): Path to the shapefile.
    - region_id (int): ID of the region to extract.
    
    Returns:
    - pd.DataFrame: DataFrame containing boundary information.
    """
    ids = []
    boundaries = []

    # Open and extract boundaries
    with fiona.open(shape_file, "r") as shapefile:
        for feature in shapefile:
            ids.append(int(feature['id'])) #id is registered as a string in the geometry file
            polygon = shape(feature['geometry'])
            bounds = polygon.bounds
            boundaries.append(bounds)

    # Create a DataFrame with the information retrieved from the previous block
    df = pd.DataFrame({'boundaries': boundaries}, index=ids)
    df = df.loc[df.index == region_id]
    
    return df



def load_masks(cnn_dataset_directory, df, common_years):

    """
    Load masks of ice, land, sea, grounded ice, and borders from the specified directory.
    
    Args:
    - cnn_dataset_directory (str): Path to the directory containing the masks.
    - df (pd.DataFrame): DataFrame with index representing region IDs and columns representing years.
    - common_years (list): List of common years across the dataset.
    
    Returns:
    - pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame: DataFrames for ice mask, land mask,
      sea mask, grounded ice mask, and borders mask.
    """

    ice_mask = np.load(cnn_dataset_directory + 'ice_mask.npy', allow_pickle=True)
    land_mask = np.load(cnn_dataset_directory + 'land_mask.npy', allow_pickle=True)
    sea_mask = np.load(cnn_dataset_directory + 'sea_mask.npy', allow_pickle=True)
    grounded_ice_mask = np.load(cnn_dataset_directory + 'grounded_ice_mask.npy', allow_pickle=True)
    boarders_mask = np.load(cnn_dataset_directory + 'boarders.npy', allow_pickle=True)


    regions = np.arange(1, 37)


    ice_mask = pd.DataFrame(ice_mask, index=regions, columns=common_years)
    land_mask = pd.DataFrame(land_mask, index=regions, columns=common_years)
    sea_mask = pd.DataFrame(sea_mask, index=regions, columns=common_years)
    grounded_ice_mask = pd.DataFrame(grounded_ice_mask, index=regions, columns=common_years)
    boarders_mask = pd.DataFrame(boarders_mask, index=regions, columns=common_years)

    region_id = df.index

    ice_mask = ice_mask.loc[region_id]
    land_mask = land_mask.loc[region_id]
    sea_mask = sea_mask.loc[region_id]
    grounded_ice_mask = grounded_ice_mask.loc[region_id]
    boarders_mask = boarders_mask.loc[region_id]
    
    return ice_mask, land_mask, sea_mask, grounded_ice_mask, boarders_mask




import netCDF4 as nc

#Data dictionary is a dictionary that contains the variables that we want to save in the NetCDF file

# Define a function to create NetCDF files
# Coordinates  are the coordinates of the region (xmin, ymin, xmax, ymax)
# Data_dict is a dictionary with the variables that we want to save in the NetCDF file
# Filename is the name of the file where we want to save the NetCDF file

def create_netcdf_file(coordinates, data_dict, filename):

    xmin, ymin, xmax, ymax = coordinates
    # Create a new NetCDF file

    with nc.Dataset(filename, 'w', format = 'NETCDF4') as ds:
        # Define dimensions

        print('The shape of the first variable is:',data_dict['bm'][0][0,0].shape)

        ds.createDimension('x', data_dict['bm'][0][0,0].shape[1])
        ds.createDimension('y', data_dict['bm'][0][0,0].shape[0])
        ds.createDimension('time', len(data_dict['bm'][0][0]))

        # Create variables
        x = ds.createVariable('x', 'f4', ('x',))
        x.units = 'meters'
        x.long_name = 'x coordinate of projection'
        x.standard_name = 'projection_x_coordinate'
        x.axis = 'X'
        if xmin<xmax:
            x[:] = np.linspace(xmin, xmax, data_dict['bm'][0][0,0].shape[1])
        else:
            x[:] = np.linspace(xmax, xmin, data_dict['bm'][0][0,0].shape[1])[::-1]
            
        
        print('The first and last coordinates of x are respectively:', x[0], x[-1])

        y = ds.createVariable('y', 'f4', ('y',))
        y.units = 'meters'
        y.long_name = 'y coordinate of projection'
        y.standard_name = 'projection_y_coordinate'
        y.axis = 'Y'
        if ymin<ymax:
            y[:] = np.linspace(ymin, ymax, data_dict['bm'][0][0, 0].shape[0])[::-1]
        else:
            y[:] = np.linspace(ymax, ymin, data_dict['bm'][0][0, 0].shape[0])
        
        print('The first and last coordinates of y are respectively:', y[0], y[-1])

        time = ds.createVariable('time', 'f4', ('time',))
        time.units = 'years from 2005 till 2016'
        time.long_name = 'time'
        time.calendar = 'gregorian'
        time.axis = 'T'
        time[:] = np.arange(2005,2017,1)

        # Add projection information
        ds.projection = 'EPSG:3031'  

        variables = {}
        for var_name, (var_data,var_meta) in data_dict.items():

            var_units = var_meta['units']
            var_description = var_meta['description']

            print('Variable:', var_name, 'Shape:', var_data[0,:].shape)

            variables[var_name] = ds.createVariable(var_name, 'f4', ('time', 'y', 'x',))
            variables[var_name].units = var_units  # Update with appropriate units
            variables[var_name].long_name = var_description  # Update with appropriate description

            variables[var_name][:] = np.stack(var_data[0],axis=0)

                   # Add metadata as global attributes
        ds.description = (f"This NetCDF file contains the physical variables basal melting, ice thickness, "
                          f"sea ice concentration, and ice velocity components in the Antarctic region limited by the polar stereographic coordinates:\n"
                          f"xmin = {xmin}, ymin = {ymin}, xmax = {xmax}, ymax = {ymax}.\n"
                          f"The caraterisation of each pixel is also stored in this dataset: the value stored is 0 if is sea is present, 1 if grounded ice, 2 if is land, 3 if is floating ice. \n")
        ds.history = 'Created on 2024-05-15 by Francesco Moncada. You can find the Git-Hub repository with the code used to generate it here https://github.com/Moncada-Francesco-97/machine_learning_calving_project.'  # Update with creation information


