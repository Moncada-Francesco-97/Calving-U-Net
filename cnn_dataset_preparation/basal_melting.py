import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
#from sklearn.metrics import r2_score
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.animation as animation # animation
import imageio 
import fiona
import rasterio
import rasterio.transform
import rasterio.mask
from fiona import Feature, Geometry
from shapely.geometry import mapping, shape
import os
import functions
import importlib
import sys
from scipy.interpolate import griddata
from scipy.interpolate import RegularGridInterpolator


def basal_melting(region_id):

    common_years = np.arange(2005, 2017, 1)
    # read shapefile
    shape_file = '/bettik/moncadaf/data/shapefiles_antarctica/squares.shp.gpkg'

    from functions import read_shapefile
    df = read_shapefile(shape_file, region_id)

    # Load the masks
    importlib.reload(functions)  # Reload the module
    from functions import load_masks
    mask_directory = '/bettik/moncadaf/data/masks/'

    ice_mask, land_mask, sea_mask, grounded_ice_mask, boarders_mask = load_masks(mask_directory, df, common_years)


    root = '/bettik/moncadaf/data/basal_melting/melt_' 
    end = '_warp_ps.tif'

    file_name_bm = []

    for year in common_years:
        file_name_bm.append(root + str(year) + end)

    # I need to open bed machine
    bm = pd.DataFrame(index=df.index, columns=common_years)
    percentile_value = 30

    for id in df.index:

        xmin, ymin, xmax, ymax = df['boundaries'].loc[id]

        for year in common_years:

            file_name = root + str(year) + end

            with rasterio.open(file_name, crs = 'EPSG:3031') as src:

                window = rasterio.windows.from_bounds(xmin, ymin, xmax, ymax, src.transform) #this is bm
                image = src.read(1, window=window) #this is bm in the window

                bm_tmp = np.zeros_like(image, dtype = float)
                bm_tmp = image

                percentile = np.percentile(bm_tmp[~np.isnan(bm_tmp)], percentile_value) #calculating percentile
                print(percentile)

                #filling the boarders with the percentile value
                bm_tmp[land_mask.loc[id, year]] = 0 #filling the land with nan
                bm_tmp[sea_mask.loc[id, year]] = 0
                bm_tmp[grounded_ice_mask.loc[id, year]] = 0
                bm_tmp[boarders_mask.loc[id, year]] = percentile

                bm.loc[id, year] = bm_tmp


    interpolated_values = pd.DataFrame(index = df.index, columns = common_years)

    for id in df.index:
        for year in common_years:

            bm_nan = np.where(np.isnan(bm.loc[id,year]))
            bm_not_nan = np.where(~np.isnan(bm.loc[id,year]))

            bm_values = bm.loc[id,year][bm_not_nan]

            interpolated_values.loc[id,year] = griddata(
                bm_not_nan, bm_values, bm_nan, method='linear')
            
    #Substitute the values where i interpolated
    bm_interpolated = bm.copy() #store the values where i interpolated

    for id in df.index:
        for year in common_years:

            nan_indices = np.isnan(bm_interpolated.loc[id,year])
            bm_interpolated.loc[id,year][nan_indices] = interpolated_values.loc[id,year]
            bm_interpolated.loc[id,year] = np.where(ice_mask.loc[id,year] == True, bm_interpolated.loc[id,year], 0)

    #Save the interpolated values
    saving_directory = '/bettik/moncadaf/data/outputs/machine_learning_calving_project/cnn_dataset/'


    np.save(saving_directory + 'bm_region_' + str(region_id) + '.npy', bm_interpolated)


