import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
import imageio 
import fiona
import rasterio
import rasterio.transform
import rasterio.mask
from fiona import Feature, Geometry
from shapely.geometry import mapping, shape
import functions
import importlib
import sys
from scipy.interpolate import griddata
from scipy.interpolate import RegularGridInterpolator
import sys
from functions import read_shapefile
import importlib


def create_oar_file(region_number):
    with open('oar_template.txt', 'r') as template_file:
        oar_template = template_file.read()

    oar_content = oar_template.format(region_number=region_number)

    with open(f'Job_Region{region_number}.oar', 'w') as oar_file:
        oar_file.write(oar_content)

def create_bat_file(region_number):
    with open('bat_region.py', 'r') as bat_template_file:
        bat_template = bat_template_file.read()

    with open(f'bat_Region{region_number}.py', 'w') as bat_file:
        bat_file.write(bat_template)

def main():
    # Iterate over regions from 1 to 28
    for region_number in range(1, 3):
        create_oar_file(region_number)
        create_bat_file(region_number)

if __name__ == "__main__":
    main()
