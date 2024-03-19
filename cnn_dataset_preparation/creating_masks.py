import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.animation as animation # animation
import imageio 

import fiona
from shapely.geometry import shape
import rasterio
import rasterio.transform
import rasterio.mask
from fiona import Feature, Geometry
from shapely.geometry import mapping, shape
import os

common_years = np.arange(2005,2017)

#Shape file with all the regions
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

#save the dataframe
root = '/UPDATE/JPL_iceshelves_geometryJPL_antarctic_coastline_'
end = '_filled.tif'


ice_mask = pd.DataFrame(index = ids, columns = common_years)
land_mask = pd.DataFrame(index = ids, columns = common_years)
sea_mask = pd.DataFrame(index = ids, columns = common_years)
grounded_ice_mask = pd.DataFrame(index = ids, columns = common_years)

for id in df.index:
    for year in common_years:
        tif_path = root + str(year) + end
        with rasterio.open(tif_path, crs = 'EPSG:3031') as src:

            xmin, ymin, xmax, ymax = df['boundaries'].loc[id]
            window = rasterio.windows.from_bounds(xmin, ymin, xmax, ymax, src.transform)
            image = src.read(1, window=window)

            ice_mask_tmp = np.zeros_like(image, dtype=bool)
            land_mask_tmp = np.zeros_like(image, dtype=bool)
            sea_mask_tmp = np.zeros_like(image, dtype=bool)
            grounded_ice_mask_tmp = np.zeros_like(image, dtype=bool)

            ice_mask_tmp[image == 3] = True
            land_mask_tmp[image == 2] = True
            sea_mask_tmp[image == 0] = True
            grounded_ice_mask_tmp[image == 1] = True

            ice_mask.loc[id, year] = ice_mask_tmp
            land_mask.loc[id, year] = land_mask_tmp
            sea_mask.loc[id, year] = sea_mask_tmp
            grounded_ice_mask.loc[id, year] = grounded_ice_mask_tmp

#save the masks as numpy arrays
cnn_dataset_directory = '/UPDATE/'
np.save(cnn_dataset_directory + 'ice_mask.npy', ice_mask)
np.save(cnn_dataset_directory + 'land_mask.npy', land_mask)
np.save(cnn_dataset_directory + 'sea_mask.npy', sea_mask)
np.save(cnn_dataset_directory + 'grounded_ice_mask.npy', grounded_ice_mask)


from scipy.ndimage import binary_dilation

#now we create the boarders, by expanding the sea ice mask by 1 pixel
boarders = pd.DataFrame(index = ids, columns = common_years)

sea_mask_expanded = sea_mask.copy()

for id in df.index:
    for year in common_years:
        sea_mask_expanded.loc[id, year] = binary_dilation(sea_mask.loc[id, year], iterations=1)
        boarders.loc[id, year] = sea_mask_expanded.loc[id, year]*1 - sea_mask.loc[id, year]*1

        #change boarders to boolean
        boarders.loc[id, year] = boarders.loc[id, year].astype(bool)

#save the boarders as numpy arrays
np.save(cnn_dataset_directory + 'boarders.npy', boarders)