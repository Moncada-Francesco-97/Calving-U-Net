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

common_years = np.arange(2005,2017)

#Get information from the shapefile

import fiona
from shapely.geometry import shape

shape_file = '/UPDATE/squares.shp.gpkg'

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

# Sort the dataset according to the index
df = df.sort_index()
pd.set_option('display.max_rows', None)

#Load the masks
cnn_dataset_directory = '/UPDATE/'

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



#need to create the list of tif files

root = '/UPDATE/thickness_'
end = '_warp_ps.tif'

thickness_paolo = pd.DataFrame(index = ids, columns = common_years)
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


bed_machine_file = '/UPDATE/Bed_Machine_thickness.tif' #Change in cluseter

# I need to open bed machine
thickness = pd.DataFrame(index=ids, columns=common_years)

for id in df.index:
    for year in common_years:
        with rasterio.open(bed_machine_file, crs='EPSG:3031') as src:

            xmin, ymin, xmax, ymax = df['boundaries'].loc[id]
            window = rasterio.windows.from_bounds(xmin, ymin, xmax, ymax, src.transform)
            image = src.read(1, window=window) #this is bed machine

            # Here I am removing pixels where there is ice (according to Greene mask)
            image = np.where(ice_mask.loc[id, year] == True, np.nan, image)

            #Here i am putting paolo thicness where paolo is not nan, and leaving bed machine where paolo is nan
            image = np.where(~np.isnan(thickness_paolo.loc[id, year]), thickness_paolo.loc[id, year], image)

            thickness.loc[id, year] = image


#Interpolation 

import numpy as np
from scipy.interpolate import RegularGridInterpolator
from scipy.interpolate import griddata

interpolated_thickness_cnn_values_2 = pd.DataFrame(index=ids, columns=common_years)

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

#Save the interpolated thickness as npy file
directory = '/UPDATE/' #Change 
np.save(directory + 'thickness.npy', thickness_tif_interpolated_cnn_2)
