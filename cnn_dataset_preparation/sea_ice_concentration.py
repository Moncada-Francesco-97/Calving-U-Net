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
    root = '/Users/francesco/Desktop/Thesis/Data/monthly/seaice_conc_monthly_sh_'

    #################### first block ####################
    start_date = "1978-11"
    end_date = "1987-07"

    date_range = pd.date_range(start=start_date, end=end_date, freq="MS")
    date_list_1 = [date.strftime("%Y%m") for date in date_range]

    end_1 = '_n07_v04r00.tif'
    #################### second block ####################
    start_date = "1987-08"
    end_date = "1991-12"

    date_range = pd.date_range(start=start_date, end=end_date, freq="MS")
    date_list_2 = [date.strftime("%Y%m") for date in date_range]
    end_2 = '_f08_v04r00.tif'
    #################### third block ####################
    start_date = "1992-01"
    end_date = "1995-09"

    date_range = pd.date_range(start=start_date, end=end_date, freq="MS")
    date_list_3 = [date.strftime("%Y%m") for date in date_range]
    end_3 = '_f11_v04r00.tif'
    #################### fourth block ####################
    start_date = "1995-10"
    end_date = "2007-12"

    date_range = pd.date_range(start=start_date, end=end_date, freq="MS")
    date_list_4 = [date.strftime("%Y%m") for date in date_range]
    end_4 = '_f13_v04r00.tif'
    #################### fifth block ####################
    start_date = "2008-01"
    end_date = "2022-12"

    date_range = pd.date_range(start=start_date, end=end_date, freq="MS")
    date_list_5 = [date.strftime("%Y%m") for date in date_range]
    end_5 = '_f17_v04r00.tif'
    #####################################################

    primo = []
    for i in date_list_1:
        tmp = root + str(i) + end_1
        primo.append(tmp)

    secondo = []
    for i in date_list_2:
        tmp = root + str(i) + end_2
        secondo.append(tmp)

    terzo = []
    for i in date_list_3:
        tmp = root + str(i) + end_3
        terzo.append(tmp)

    quarto = []
    for i in date_list_4:
        tmp = root + str(i) + end_4
        quarto.append(tmp)

    quinto = []
    for i in date_list_5:
        tmp = root + str(i) + end_5
        quinto.append(tmp)

    #now i want to merge the lists
    #list_of_files = primo + secondo + terzo + quarto + quinto
    list_of_files = terzo + quarto + quinto
    list_of_files = list_of_files[156:300] #this way i am selecting from 2005 till 2016 (included)

    #Averaging all the years

    sic = pd.DataFrame(index = df.index, columns = common_years)

    for id in df.index:

        month = 0

        for year in common_years:

            image_avg = np.zeros((1024,1024))

            for i in range(0,12):

                j = month + i
                file = list_of_files[j]
                print(file)

                with rasterio.open(file, crs = 'EPSG:3031') as src:

                    xmin, ymin, xmax, ymax = df.loc[id, 'boundaries']
                    window = rasterio.windows.from_bounds(xmin, ymin, xmax, ymax, src.transform) 
                    image = src.read(1, window=window) #this is bm in the window
                    image_avg = image_avg + image
                    
                    if i == 11:
                        print('Saved till month ' + str(j))
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

            print(id, year)

            sea_ice_tmp = sic.loc[id, year]
            nan_mask = np.isnan(sea_ice_tmp)
            sea_ice_tmp = inpaint.inpaint_biharmonic(sea_ice_tmp, nan_mask)
            sea_ice_tmp = np.where(~sea_mask.loc[id,year] , 0, sea_ice_tmp)

            sic_interpolated.loc[id, year] = sea_ice_tmp

    #Save the interpolated values
    saving_directory = '/bettik/moncadaf/data/outputs/machine_learning_calving_project/cnn_dataset/'    

    np.save(saving_directory + f'sic_region_{region_id}.npy', sic_interpolated)
    print(f'Saved region {region_id}')

