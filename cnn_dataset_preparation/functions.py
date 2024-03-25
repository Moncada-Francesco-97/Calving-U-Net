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




#Load Masks
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

    ice_mask = pd.DataFrame(ice_mask, index=df.index, columns=common_years)
    land_mask = pd.DataFrame(land_mask, index=df.index, columns=common_years)
    sea_mask = pd.DataFrame(sea_mask, index=df.index, columns=common_years)
    grounded_ice_mask = pd.DataFrame(grounded_ice_mask, index=df.index, columns=common_years)
    boarders_mask = pd.DataFrame(boarders_mask, index=df.index, columns=common_years)
    
    return ice_mask, land_mask, sea_mask, grounded_ice_mask, boarders_mask



