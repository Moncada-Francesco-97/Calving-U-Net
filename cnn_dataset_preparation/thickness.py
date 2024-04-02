import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
#from sklearn.metrics import r2_score
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.animation as animation # animation
import imageio 
import sys
import fiona
import rasterio
import rasterio.transform
import rasterio.mask
from fiona import Feature, Geometry
from shapely.geometry import mapping, shape
import os
import functions
import fiona
from shapely.geometry import shape
import importlib
from functions import read_shapefile
import numpy as np
from scipy.interpolate import RegularGridInterpolator
from scipy.interpolate import griddata

common_years = np.arange(2005,2017)

def thickness(region_id):
    # read shapefile
    #print(f'Processing region {region_id}, we are into thickness.py')
    shape_file = '/bettik/moncadaf/data/shapefiles_antarctica/squares.shp.gpkg'
    df = read_shapefile(shape_file, region_id)


    #Load the masks
    importlib.reload(functions)  # Reload the module
    from functions import load_masks

    cnn_dataset_directory = '/bettik/moncadaf/data/masks/'
    ice_mask, land_mask, sea_mask, grounded_ice_mask, boarders_mask = load_masks(cnn_dataset_directory, df, common_years)

    #print('Masks loaded')


    #need to create the list of tif files

    root = '/bettik/moncadaf/data/ice_thickness/thickness_'
    end = '_warp_ps.tif'

    thickness_paolo = pd.DataFrame(index = df.index, columns = common_years)
    percentile_value = 30

    for year in common_years:
        for id in df.index:
            with rasterio.open(root + str(year) + end, crs = 'EPSG:3031') as src:

                xmin, ymin, xmax, ymax = df['boundaries'].loc[id]
                window = rasterio.windows.from_bounds(xmin, ymin, xmax, ymax, src.transform)
                image = src.read(1, window=window)

                thickness_paolo_tmp = np.zeros_like(image, dtype=float)
                thickness_paolo_tmp = image

                #calculate the percentile
                percentile = np.percentile(thickness_paolo_tmp[~np.isnan(thickness_paolo_tmp)], percentile_value)

                #add the percentile on the boarders
                thickness_paolo_tmp[boarders_mask.loc[id, year]] = percentile

                #Where there is sea we set the thickness to 0
                thickness_paolo_tmp[sea_mask.loc[id, year]] = 0

                thickness_paolo.loc[id, year] = thickness_paolo_tmp

    
    #print('Paolo thickness loaded')

    bed_machine_file = '/bettik/moncadaf/data/shapefiles_antarctica/Bed_Machine_thickness.tif' #Change in cluseter

    # I need to open bed machine
    thickness = pd.DataFrame(index=df.index, columns=common_years)

    for id in df.index:
        for year in common_years:
            with rasterio.open(bed_machine_file, crs='EPSG:3031') as src:

                xmin, ymin, xmax, ymax = df['boundaries'].loc[id]
                window = rasterio.windows.from_bounds(xmin, ymin, xmax, ymax, src.transform)
                image = src.read(1, window=window) #this is bed machine

                print('The shape of the bed_machine for the year {} is {}'.format(year, image.shape))

                # Here I am removing pixels where there is ice (according to Greene mask)
                image = np.where(ice_mask.loc[id, year] == True, np.nan, image)

                #Here i am putting paolo thicness where paolo is not nan, and leaving bed machine where paolo is nan
                image = np.where(~np.isnan(thickness_paolo.loc[id, year]), thickness_paolo.loc[id, year], image)

                thickness.loc[id, year] = image


    #Interpolation 

    print('Right before thicknes interpolation')

    interpolated_thickness_cnn_values_2 = pd.DataFrame(index=df.index, columns=common_years)

    for id in df.index:
        for year in common_years:

            thickness_tmp = thickness.loc[id, year]

            #get coordinates of nan values
            nan_coords = np.where(np.isnan(thickness_tmp))

            #get not a nan coordinates
            not_nan_coords = np.where(~np.isnan(thickness_tmp))

            #get the values of not nan
            values = thickness_tmp[not_nan_coords]

            #perform interpolation
            interpolated_thickness_cnn_values_2.loc[id,year] = griddata(not_nan_coords, values, nan_coords, method='linear')

    thickness_tif_interpolated_cnn_2 = thickness.copy()

    for id in df.index:
        for year in common_years:
            thickness_tif_interpolated_cnn_2.loc[id, year][np.where(np.isnan(thickness_tif_interpolated_cnn_2.loc[id,year]))] = interpolated_thickness_cnn_values_2.loc[id,year]


    #Save the interpolated values
    saving_directory = '/bettik/moncadaf/data/outputs/machine_learning_calving_project/cnn_dataset/'


    np.save(saving_directory + f'thickness_region_{region_id}.npy', thickness_tif_interpolated_cnn_2)
    print(f'Saved thickness region {region_id}')

