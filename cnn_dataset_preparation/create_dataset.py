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
import subprocess



# def create_oar_file(region_number):
#     with open('oar_template.txt', 'r') as template_file:
#         oar_template = template_file.read()

#     print(f'OAR Template content:\n{oar_template}')  # Add this line for debugging
#     oar_content = oar_template.format(region_number)

#     with open(f'Job_Region{region_number}.oar', 'w') as oar_file:
#         oar_file.write(oar_content)


def create_bat_file(region_number):
    with open('bat_region.py', 'r') as bat_template_file:
        bat_template = bat_template_file.read()

    with open(f'bat_region.py', 'w') as bat_file:
        bat_file.write(bat_template)

    print(f'bat_region.py was read and written for region {region_number}')



def create_oar_file(region_number):
    f = open(f'Job_Region{region_number}.oar', 'w')
    f.write('#!/bin/bash\n')
    f.write(f'#OAR -n Job_Region_{region_number}\n')
    f.write('#OAR -l /nodes=1/core=16,walltime=4:00:00\n')
    f.write('#OAR -O result_prep.%jobid%.out\n')
    f.write('#OAR -E result_prep.%jobid%.err\n')
    f.write('#OAR -t idempotent\n')
    f.write('#OAR -q default\n')
    f.write('#OAR --project ice_speed\n')
    f.write('\n')
    f.write('source /home/moncadaf/.bashrc\n')
    f.write('eval "$(/home/moncadaf/miniconda3/bin/conda shell.bash hook)"\n')
    f.write('conda activate myenv\n')
    f.write('cd /bettik/moncadaf/machine_learning_calving_project/cnn_dataset_preparation/\n')
    f.write(f'python bat_region.py {region_number}\n')
    f.close()
    
    print(f'Job_Region{region_number}.oar created')
    
    os.system('chmod 777 Job_Region'+str(region_number)+'.oar')
    os.system(f'oarsub -S ./Job_Region{region_number}.oar')

def main():
    # Iterate over regions from 1 to 28
    for region_number in [24]:
        print(f'Creating files for region {region_number}')
        create_oar_file(region_number)

        #subprocess.run(f"python bat_region.py {region_number}", shell =True)



        #create_bat_file(region_number)

if __name__ == "__main__":
    main()
