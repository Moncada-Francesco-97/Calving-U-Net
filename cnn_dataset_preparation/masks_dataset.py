import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
#import xarray as xr
import imageio 

import fiona
from shapely.geometry import shape
import rasterio
import rasterio.transform
import rasterio.mask
from fiona import Feature, Geometry
from shapely.geometry import mapping, shape
import os
import sys
import functions
from functions import read_shapefile

#This function creates a single dataset with the masks of the ice shelves for a given region. Region id is tipically a single number,
#which identifies the region.


def masks_dataset(region_id):

    common_years = np.arange(2005,2017)

    # read shapefile
    shape_file = '/bettik/moncadaf/data/shapefiles_antarctica/squares.shp.gpkg'
    df = read_shapefile(shape_file, region_id)

    root = '/bettik/millanr/DATA_SERVER/ANTARCTICA/OCEANICE/COASTLINE/JPL_iceshelves_geometry/FILES_FOR_FRANCESCO/JPL_iceshelves_geometryJPL_antarctic_coastline_'
    end = '_filled.tif'

    masks_region = pd.DataFrame(index = df.index, columns = common_years)

    for id in df.index:
        for year in common_years:
            tif_path = root + str(year) + end
            with rasterio.open(tif_path, crs = 'EPSG:3031') as src:

                xmin, ymin, xmax, ymax = df['boundaries'].loc[id]
                window = rasterio.windows.from_bounds(xmin, ymin, xmax, ymax, src.transform)
                image = src.read(1, window=window)

                masks_region.loc[id, year] = image

    #save the masks as numpy arrays
    saving_directory = '/bettik/moncadaf/data/outputs/machine_learning_calving_project/cnn_dataset/'

    np.save(saving_directory + 'masks_region_' + str(region_id) + '.npy', masks_region)

    print('Masks for region ' + str(region_id) + ' saved in ' + saving_directory)

