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
from functions import read_shapefile
import importlib

common_years = np.arange(2005, 2017, 1)


def sea_ice_concentration(region_id):

    common_years = np.arange(2005, 2017, 1)

    # read shapefile
    shape_file = '/bettik/moncadaf/data/shapefiles_antarctica/squares.shp.gpkg'
    df = read_shapefile(shape_file, region_id)

    # Load the masks
    importlib.reload(functions)  # Reload the module
    from functions import load_masks
    mask_directory = '/bettik/moncadaf/data/masks/'

    ice_mask, land_mask, sea_mask, grounded_ice_mask, boarders_mask = load_masks(mask_directory, df, common_years)

    #Extract the Sea Ice Concentration

    #in this script i want to create a list of all the files i want to open

    # Define the root directory
    root = '/bettik/moncadaf/data/sea_ice_concentration/'

    # Define the list of files
    list_of_files = []
    common_years = np.arange(2005,2017,1)

    # Loop through the years and months to generate filenames
    for year in range(2005, 2017):
        for month in range(1, 13):
            # Pad single-digit months with a leading zero
            padded_month = f"{month:02d}"
            # Generate the filename and add it to the list
            filename = f"{root}seaice_conc_monthly_sh_{year}{padded_month}.tif"
            list_of_files.append(filename)

    #Averaging all the years

    sic = pd.DataFrame(index = df.index, columns = common_years)

    for id in df.index:

        month = 0

        for year in common_years:

            image_avg = np.zeros((1024,1024))

            for i in range(0,12):

                j = month + i
                file = list_of_files[j]
                #print(file)

                with rasterio.open(file, crs = 'EPSG:3031') as src:

                    xmin, ymin, xmax, ymax = df.loc[id, 'boundaries']
                    window = rasterio.windows.from_bounds(xmin, ymin, xmax, ymax, src.transform) 
                    image = src.read(1, window=window) #this is bm in the window

                    print('Year: ' + str(year) + ' Month: ' + str(i) + ' Image shape: ' + str(image.shape))
                    image_avg = image_avg + image
                    
                    if i == 11:
                        # print('Saved till month ' + str(j))
                        image_avg = image_avg/12
                        sic.loc[id,year] = image_avg
                        month = month + 12


    # Now I want to put the ice and land, and grounded masks on top of the sic

    for id in df.index:
        
            for year in common_years:
        
                sic.loc[id,year] = np.where(sic.loc[id,year]==251, np.nan, sic.loc[id,year])
                sic.loc[id,year] = np.where(sic.loc[id,year]==252, np.nan, sic.loc[id,year])
                sic.loc[id,year] = np.where(sic.loc[id,year]==254, np.nan, sic.loc[id,year])
                sic.loc[id,year] = np.where(sic.loc[id,year]==255, np.nan, sic.loc[id,year])

                sic.loc[id,year] = np.where(~sea_mask.loc[id,year], np.nan, sic.loc[id,year])


    #Now perform the interpolation
    from skimage.restoration import inpaint

    sic_interpolated = pd.DataFrame(index = df.index, columns = common_years)

    for id in df.index:
        for year in common_years:

            #print(id, year)

            sea_ice_tmp = sic.loc[id, year]
            nan_mask = np.isnan(sea_ice_tmp)

        # Check if the array is empty or only contains NaN values
            if sea_ice_tmp.size == 0 or np.all(nan_mask):
                print(f"Skipping inpainting for region {id}, year {year} due to empty or all-NaN array.")
                sic_interpolated.loc[id, year] = sea_ice_tmp  # Keep it as is or fill with a default value
                continue

            sea_ice_tmp = inpaint.inpaint_biharmonic(sea_ice_tmp, nan_mask)
            sea_ice_tmp = np.where(~sea_mask.loc[id,year] , 0, sea_ice_tmp)

            sic_interpolated.loc[id, year] = sea_ice_tmp

    #Save the interpolated values
    saving_directory = '/bettik/moncadaf/data/outputs/machine_learning_calving_project/cnn_dataset/'    

    np.save(saving_directory + f'sic_region_{region_id}.npy', sic_interpolated)
    print(f'Saved region {region_id}')

