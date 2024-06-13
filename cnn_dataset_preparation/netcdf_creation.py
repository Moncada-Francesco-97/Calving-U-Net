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
import fiona
from shapely.geometry import shape
import importlib
import numpy as np
from scipy.interpolate import RegularGridInterpolator
from scipy.interpolate import griddata
import netCDF4 as nc

from functions import create_netcdf_file

region = '/bettik/moncadaf/data/shapefiles_antarctica/squares.shp.gpkg'

coords = []

with fiona.open(region, "r") as shapefile:
    shapes = [feature["geometry"] for feature in shapefile]
    #Extract the boundary of the shapefile

    for i in range(len(shapes)):
        boundary = shape(shapes[i])
        #Extract the coordinates of the boundary
        xmin, ymin, xmax, ymax = boundary.bounds
        coords.append([xmin, ymin, xmax, ymax])

root_dir = '/bettik/moncadaf/data/outputs/machine_learning_calving_project/cnn_dataset/'
netcdf_directory = '/bettik/moncadaf/dataset/netcdf/'

for i in range(1,37):

    print('We are working on region:', i)

    xmint, ymint, xmaxt, ymaxt = coords[i-1]
    print(xmint, ymint, xmaxt, ymaxt)

    bm = np.load(root_dir+'bm_region_'+str(i)+'.npy', allow_pickle=True)
    sic = np.load(root_dir+'sic_region_'+str(i)+'.npy', allow_pickle=True)
    thickness = np.load(root_dir+'thickness_region_'+str(i)+'.npy', allow_pickle=True)
    v_x = np.load(root_dir+'v_x_region_'+str(i)+'.npy', allow_pickle=True)
    v_y = np.load(root_dir+'v_y_region_'+str(i)+'.npy', allow_pickle=True)
    mask = np.load(root_dir+'masks_region_'+str(i)+'.npy', allow_pickle=True)

    data_dict = {
        'bm': (bm, {'units': '[m]/[yr]', 'description': 'Basal Melting'}),
        'sic': (sic, {'units': '[%]', 'description': 'Description of sic variable'}),
        'thickness': (thickness, {'units': '[m]', 'description': 'Ice Thickness'}),
        'v_x': (v_x, {'units': '[m]/[yr]', 'description': 'x component of velocity'}),
        'v_y': (v_y, {'units': '[m]/[yr]', 'description': 'y component of velocity'}),
        'masks': (mask, {'units': '[-]', 'description': 'mask containing informations about each pixel type. The value of each pixel is 0 if sea is present, 1 if is grounded ice, 2 if is land, 3 if is floating ice.'}),
    }

    create_netcdf_file([xmint, ymint, xmaxt, ymaxt], data_dict, netcdf_directory+'netcdf_region_'+str(i)+'.nc')