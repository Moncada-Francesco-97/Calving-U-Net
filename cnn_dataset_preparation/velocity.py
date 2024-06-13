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

####CODE DESCRIPTION#######

#This code is used to load and interpolate the ice velocity data. Compared to the other variables, the ice velocity dataset has been more complicate to interpolate,
#due to the many missing data points. The code takes as reference the ice shelf geometry provided in the work of Green et al.(2022), and is structured in the following way:

###BLOCK 1: Introduce the function, load the masks and the velocity data, both multi year and single year, and smooth and threshold the data.
    #The multi year file is the result of a multi-year average, and has almost a complete coverage of the Antarctic continent.
    #The single year files are the yearly velocity data, which have a lot of missing data points. 


###BLOCK 2: Create the masks for the ice shelves and perform a first interpolation. 

# Since we wanted to exploit at maximum the data points inside ice shelves, we decided to spatially interpolate the missing data twice,
# once in the floating ice and later including also the grounded ice.
# To look directly on the ice shelf floating ice (and not on the overall quantity of floating ice), we managed to divide ice shelf by ice shelf by labeliing the connected regions of floating ice,
# so that each ice shelf has an unique label; this was done thanks to the label library. We then checked the coverage of each ice shelf, and if it was larger than 70%, we kept the data and
# performed spatial interpolation, otherwise we set the data to 0.
# After this process I merge the interpolated data of the floatinng ice shelves with the original data of the grounded ice.
# Also in this case I check  wether the coverage of this new region is larger than 70%, and if so I inerpolate spatially.

###BLOCK 3: Perform a pixel by pixel time interpolation of the data, for region where coverage is less than 70%.
#To maximise the exploitation of available data, and since we have years where data coverage is really poor, we perform a second interpolation in time, pixel by pixel.
# To do so is necessary to reshape the dataset in order to have a for each pixel a time serie from which is possible to interpolate the missing values.
#To make this interpolation we decided to act not directly on the velocity data, but on the ratio between the single year velocity and the multi year velocity, which we used as benchline.
# Having performed this interpolation, we multiply the ratio matrix with the multi year velocity, to rescale it.
# In doing this process, we avoided to extrapolate, and in case the first(s) or last(s) values of the time serie were missing, we set it to the closest in time non missing value.

#BLOCK 4: Perform the final interpolation of the data, and save the results.
# To take into account front advance we created a mask which include the regions where we have 0 values in the multi year velocity dataset (which means that we are in the sea according to this dataset),
# but we have floating ice according to Green dataset (which means that a front advance happened instead).
# We then interpolate the data in those regions, and we save the final dataset.

#N.B: In the code there are many 'tailored' solution for the Amundsen Bay region, where many data were added to take into account the evolution of the ice shelves,
# in particular Pine Island and Thwaites.



def velocity(region_id):

    #BLOCK 1

    #Functions ########################################

    def interpolation_excluding_extreames(A):
        #This function interpolates the nan values in the array A, excluding the first and last values if they are nan,
        #and the second and second to last if they are nan. This is to avoid the case where the first and last values are nan

        A = np.array(A)

        x = np.interp(np.arange(len(A)), 
                np.arange(len(A))[np.isnan(A) == False], 
                A[np.isnan(A) == False])

        if np.isnan(A[0]) == True:
            x[0] = A[0]
            if np.isnan(A[1]) == True:
                x[1] = A[1]
                if np.isnan(A[2]) == True:
                    x[2] = A[2]
                    if np.isnan(A[3]) == True:
                        x[3] = A[3]

        if np.isnan(A[-1]) == True:
            x[-1] = A[-1]
            if np.isnan(A[-2]) == True:
                x[-2] = A[-2]
                if np.isnan(A[-3]) == True:
                    x[-3] = A[-3]
                    if np.isnan(A[-4]) == True:
                        x[-4] = A[-4]
        return x


    def fill_first_values(A):
        #This function fills the first nan values with the first non nan value

        A = np.array(A)

        for i in range(len(A)):

            if np.isnan(A[i]) == False:
                A[:i] = A[i]
                break

        return A
    ####################################################################

    common_years = np.arange(2005,2017)


    #I set a threshold for the interpolation, if the coverage is larger than 70% I use the original data, otherwise I interpolate
    threshold_interpolation = 0.7

    # read shapefile
    from functions import read_shapefile
    shape_file = '/bettik/moncadaf/dataset/squares.shp.gpkg'
    df = read_shapefile(shape_file, region_id)


    #Load the masks ########################################################
    importlib.reload(functions)  # Reload the module
    from functions import load_masks

    # Load the masks
    mask_directory = '/bettik/moncadaf/dataset/produced_data/masks/'
    ice_mask, land_mask, sea_mask, grounded_ice_mask, boarders_mask = load_masks(mask_directory, df, common_years)

    ####################################################################



    #MULTI YEAR VELOCITY (does not change through time) ########################################################
    velocity_multy_years_x_path = '/bettik/moncadaf/dataset/raw_data/velocity/velocity_multi_years_X.tif'
    velocity_multy_years_y_path = '/bettik/moncadaf/dataset/raw_data/velocity/velocity_multi_years_Y.tif'

    #dataset
    velocity_multy_years_x =pd.DataFrame(index = df.index, columns = ['image'])
    velocity_multy_years_y =pd.DataFrame(index = df.index, columns = ['image'])

    #Extract the velocity data from the multi year dataset
    for id in df.index:

            with rasterio.open(velocity_multy_years_x_path, crs = 'EPSG:3031') as src:
                    xmin, ymin, xmax, ymax = df.loc[id, 'boundaries']
                    window = rasterio.windows.from_bounds(xmin, ymin, xmax, ymax, src.transform)
                    velocity_multy_years_x.loc[id, 'image'] = src.read(1, window=window)
                    velocity_multy_years_x.loc[id, 'image'][velocity_multy_years_x.loc[id, 'image']==0] = np.nan #Note that I set to nan all the 0 values. This is useful later

            with rasterio.open(velocity_multy_years_y_path, crs = 'EPSG:3031') as src:
                    xmin, ymin, xmax, ymax = df.loc[id, 'boundaries']
                    window = rasterio.windows.from_bounds(xmin, ymin, xmax, ymax, src.transform)
                    velocity_multy_years_y.loc[id,'image'] = src.read(1, window=window)
                    velocity_multy_years_y.loc[id, 'image'][velocity_multy_years_y.loc[id, 'image']==0] = np.nan

    ####################################################################



    #EXTRACTING SINGLE YEAR VELOCITY (does change through time) ########################################
    list_x = []
    list_y = []

    #Velocity
    velocity_x_tif = pd.DataFrame(index = df.index, columns = common_years)
    velocity_y_tif = pd.DataFrame(index = df.index, columns = common_years)

    velocity_x_tif_nan = pd.DataFrame(index = df.index, columns = common_years)
    velocity_y_tif_nan = pd.DataFrame(index = df.index, columns = common_years)

    #FILE LIST (change in cluster)
    for i in common_years:
        list_x.append('/bettik/moncadaf/dataset/raw_data/velocity/' + f"{i}_{i+1}_VX.tif")
        list_y.append('/bettik/moncadaf/dataset/raw_data/velocity/' + f"{i}_{i+1}_VY.tif")

    #calculate the velocity for each year, for each region
    for name_x, name_y, year in zip(list_x, list_y, common_years): #for each year

        for id in df.index: #for each region

            xmin, ymin, xmax, ymax = df.loc[id, 'boundaries']

            with rasterio.open(name_x, crs = 'EPSG:3031') as src:
                window = rasterio.windows.from_bounds(xmin, ymin, xmax, ymax, src.transform)
                image_x = src.read(1, window=window)

                #Exclude the 0.01 and 99.99 percentile (easiest way to exclude the outliers)
                values = np.nanpercentile(image_x.ravel(), [0.05, 99.95])
                image_x[image_x < values[0]] = np.nan
                image_x[image_x > values[1]] = np.nan

                #set to nan all the 0 values
                image_x[image_x == 0] = np.nan
                velocity_x_tif.loc[id, year] = image_x
                velocity_x_tif_nan.loc[id, year] = np.isnan(image_x)


            with rasterio.open(name_y, crs = 'EPSG:3031') as src:
                window = rasterio.windows.from_bounds(xmin, ymin, xmax, ymax, src.transform)
                image_y = src.read(1, window=window)

                #Exclude the 0.05 and 99.95 percentile
                values = np.nanpercentile(image_y.ravel(), [0.05, 99.95])
                image_y[image_y < values[0]] = np.nan
                image_y[image_y > values[1]] = np.nan

                #set to nan all the 0 values
                image_y[image_y == 0] = np.nan
                velocity_y_tif.loc[id, year] = image_y
                velocity_y_tif_nan.loc[id, year] = np.isnan(image_y)


    #Smoothing and thresholding
    from silx.image.medianfilter import medfilt2d

    velocity_x_tif_smoothed = velocity_x_tif.copy()
    velocity_y_tif_smoothed = velocity_y_tif.copy()

    kernel_size = 4
    threshold = 50 

    for id in df.index:
        for year in common_years:

            #smothering
            vx_tmp_smoothed = medfilt2d(velocity_x_tif_smoothed.loc[id, year], kernel_size)
            vy_tmp_smoothed = medfilt2d(velocity_y_tif_smoothed.loc[id, year], kernel_size)

            #thresholding (excluding those pixels whose difference before and after thresholding has surpassed 50 m/year)
            velocity_x_tif_smoothed.loc[id, year][abs(velocity_x_tif_smoothed.loc[id, year] - vx_tmp_smoothed) > threshold] = np.nan
            velocity_y_tif_smoothed.loc[id, year][abs(velocity_y_tif_smoothed.loc[id, year] - vy_tmp_smoothed) > threshold] = np.nan

    ##########################################################################


    #Special Treatment for Amundsen Bay ######################################## 
 
    if region_id == 24:
        from skimage.restoration import inpaint
        print('Amundsen Bay specific treatment, first treatment: interpolation of the missing values in specific regions, namely Pine Island and Thwaites')
        
        year = 2011
        mask_nan = np.isnan(velocity_x_tif_smoothed.loc[24, year][120:200, 300:450])
        velocity_x_tif_smoothed.loc[24, year][120:200, 300:450] = inpaint.inpaint_biharmonic(velocity_x_tif_smoothed.loc[24, year][120:200, 300:450], mask_nan)
        mask_nan_2 = np.isnan(velocity_y_tif_smoothed.loc[24, year][120:200, 300:450])
        velocity_y_tif_smoothed.loc[24, year][120:200, 300:450] = inpaint.inpaint_biharmonic(velocity_y_tif_smoothed.loc[24, year][120:200, 300:450], mask_nan_2)
    ####################################################################



    #BLOCK 2: 


    #Create the masks for the ice shelves ########################
    from scipy.ndimage import label

    ice_shelves_masks = pd.DataFrame(index = df.index, columns = common_years)
    number_of_ice_shelves = pd.DataFrame(index = df.index, columns = common_years)

    for id in df.index:
        for year in common_years:
            
            ice_shelf_mask = ice_mask.loc[id, year]
            ice_shelf_grounded_ice_mask = grounded_ice_mask.loc[id, year]

            combined_mask = ice_shelf_mask | ice_shelf_grounded_ice_mask

            # Perform connected component labeling on the mask (identify as an ice shelf any connected region of ice)
            labeled_mask, num_labels = label(combined_mask)

            # Create a new array to hold the labeled mask with unique labels for each ice shelf
            labeled_combined = np.zeros_like(labeled_mask)

            # Assign unique labels to each ice shelf
            for label_idx in range(1, num_labels + 1):  # Start from 1 to exclude background label, which is categorised as 0
                ice_shelf = labeled_mask == label_idx
                labeled_combined[ice_shelf] = label_idx

            # Store the labeled mask and the number of ice shelves
            ice_shelves_masks.loc[id,year] = labeled_combined
            number_of_ice_shelves.loc[id,year] = num_labels

    #The mask with all the ice shelves with coverage larger than 70%
    masks_filtered = pd.DataFrame(index = df.index, columns = common_years)

    for id in df.index:
        for year in common_years:

            prova = ice_shelves_masks.loc[id,year]
            prova_id = number_of_ice_shelves.loc[id,year] #contains the number of ice shelves in the region

            prova_filtered = prova.copy()
            ice_shelves_numbers = np.arange(1,prova_id+1,1)

            #From here we are working on the single ice shelf
            j = 0

            #In this cycle I check wether for each ice shelf the coverage is larger than 70%.
            #If is larger--> I keep the values f velocity stored in velcoity_x_tif
            #If is smaller--> I set the values of the ice shelf to 0

            for i in range(1,prova_id+1): #Iterating over the ice shelves of the region

                ice_shelf_mask = prova == ice_shelves_numbers[j] #Get the mask of ice shelf number i
                ice_shelf_mask = ice_shelf_mask.astype(bool) #convert ice_shelf_mask to boolean

                #Look in the velocity_x_tif dataset at velocity values of the pixels of ice shelf number i 
                data_in_ice_shelf_roi = np.where(ice_shelf_mask, velocity_x_tif.loc[id, year], np.nan)

                # Count the number of non-NaN pixels in the masked data_in_ice_shelf_roi
                num_non_nan_pixels = np.sum(~np.isnan(data_in_ice_shelf_roi))

                # Count the number of True pixels in the ice_shelf_mask
                num_true_pixels = np.sum(ice_shelf_mask)

                # Calculate the ratio of non-NaN pixels to True pixels
                ratio_non_nan_to_true = num_non_nan_pixels / num_true_pixels

                if ratio_non_nan_to_true < 0.7:

                    prova_filtered = np.where(prova_filtered == ice_shelves_numbers[j], 0, prova_filtered)

                j = j+1

            masks_filtered.loc[id,year] = prova_filtered

    #Convert all the mask to the same boolean format. mask_filtered is now storing the masks of the ice shelves with coverage larger than 70%
    for id in df.index:
        for year in common_years:
            masks_filtered.loc[id,year] = masks_filtered.loc[id,year].astype(bool)
    
    # #save the masks 
    # saving_directory = '/bettik/moncadaf/data/outputs/machine_learning_calving_project/cnn_dataset/'
    # np.save(saving_directory + f'masks_filtered_region_{region_id}.npy', masks_filtered)
    # np.save(saving_directory + f'velocity_x_tif_smoothed_region_{region_id}.npy', velocity_x_tif_smoothed)
    # np.save(saving_directory + f'velocity_y_tif_smoothed_region_{region_id}.npy', velocity_y_tif_smoothed)

    ####################################################################


    #interpolate into the ice shelf mask ########################################
    from skimage.restoration import inpaint

    interpolated_floating_ice_x = pd.DataFrame(index = df.index, columns = common_years)

    for id in df.index:
        for year in common_years:

            # Check if all elements in the mask for the current year are NaN. There are certain regions for particular years where there is absolutely no data.
            if not masks_filtered.loc[id,year].any():
                # If all elements are NaN, set all elements of the interpolated_floating_ice_x to NaN
                interpolated_floating_ice_x.loc[id, year] = np.nan
                print('All elements in the mask for the year', year, 'are NaN')
                continue  # Skip the current year and proceed to the next one

            # Otherwise, perform spatial interpolation 
            data_tmp = np.where(masks_filtered.loc[id, year], velocity_x_tif_smoothed.loc[id, year], np.nan)
            data_tmp = np.where(~masks_filtered.loc[id, year], velocity_x_tif_smoothed.loc[id, year], data_tmp)
            nan_mask = np.isnan(data_tmp)

            interpolated_floating_ice_x.loc[id, year] = inpaint.inpaint_biharmonic(data_tmp, nan_mask)
            interpolated_floating_ice_x.loc[id, year][~masks_filtered.loc[id, year]] = np.nan #Here I re set to nan all the regions which are not ice shelf


    #Now i interpolate the grounded ice for x
    interpolated_ice_x = pd.DataFrame(index = df.index, columns = common_years)
    coverage_ratio_x = pd.DataFrame(index = df.index, columns = common_years)

    for id in df.index:
        for year in common_years:

            #load original data for the land
            data_tmp = velocity_x_tif_smoothed.loc[id,year]

            #load the interpolated data for the floating ice shelves
            data_tmp_all_ice = np.where(masks_filtered.loc[id,year], interpolated_floating_ice_x.loc[id,year], data_tmp)
            nan_mask = np.isnan(data_tmp_all_ice)

            #Calculate the coverage ratio between the non-NaN pixels and the True pixels, in everything which is not sea
            num_non_nan_pixels = np.sum(~np.isnan(data_tmp_all_ice))
            num_true_pixels = np.sum(~sea_mask.loc[id,year])

            coverage_ratio_x.loc[id,year] = num_non_nan_pixels/num_true_pixels

            interpolated_ice_x.loc[id,year] = data_tmp_all_ice

            if coverage_ratio_x.loc[id,year] > 0.7:

                interpolated_ice_x.loc[id,year] = inpaint.inpaint_biharmonic(data_tmp_all_ice, nan_mask)
                interpolated_ice_x.loc[id,year][sea_mask.loc[id,year]] = np.nan     

    #Redoing it for y
    interpolated_floating_ice_y = pd.DataFrame(index = df.index, columns = common_years)

    for id in df.index:
        for year in common_years:

            # Check if all elements in the mask for the current year are NaN
            if not masks_filtered.loc[id,year].any():
                # If all elements are NaN, set all elements of the interpolated_floating_ice_x to NaN
                interpolated_floating_ice_y.loc[id, year] = np.nan
                print('All elements in the mask for the year', year, 'are NaN')
                continue  # Skip the current year and proceed to the next one

            
            data_tmp = np.where(masks_filtered.loc[id,year], velocity_y_tif_smoothed.loc[id,year], np.nan)
            data_tmp = np.where(~masks_filtered.loc[id,year], velocity_y_tif_smoothed.loc[id,year], data_tmp)
            nan_mask = np.isnan(data_tmp)

            interpolated_floating_ice_y.loc[id,year] = inpaint.inpaint_biharmonic(data_tmp, nan_mask)
            interpolated_floating_ice_y.loc[id,year][~masks_filtered.loc[id,year]] = np.nan


    #Now i interpolate the grounded ice for y

    interpolated_ice_y = pd.DataFrame(index = df.index, columns = common_years)
    coverage_ratio_y = pd.DataFrame(index = df.index, columns = common_years)

    for id in df.index:
        for year in common_years:

            #load original data for the land
            data_tmp = velocity_y_tif_smoothed.loc[id,year]

            #load the  interpolated data for the floating ice
            data_tmp_all_ice = np.where(masks_filtered.loc[id,year], interpolated_floating_ice_y.loc[id,year], data_tmp)
            nan_mask = np.isnan(data_tmp_all_ice)

            #Calculate the coverage ratio
            num_non_nan_pixels = np.sum(~np.isnan(data_tmp_all_ice))
            num_true_pixels = np.sum(~sea_mask.loc[id,year])

            coverage_ratio_y.loc[id,year] = num_non_nan_pixels/num_true_pixels

            #print('The ratio of non-NaN pixels to True pixels in y for the year', year, 'is', coverage_ratio_y.loc[id,year])

            interpolated_ice_y.loc[id,year] = data_tmp_all_ice

            if coverage_ratio_y.loc[id,year] > 0.7:

                interpolated_ice_y.loc[id,year] = inpaint.inpaint_biharmonic(data_tmp_all_ice, nan_mask)
                interpolated_ice_y.loc[id,year][sea_mask.loc[id,year]] = np.nan
    ####################################################################


    #Specific treatment for Amundsen Bay ########################################
    #Specific treatment for Pine Island in year 2006 (Coverage was sufficiently good to interpolate the missing values)
    if region_id == 24:
        mask_nan_pi = np.isnan(interpolated_ice_y.loc[region_id,2006][100:175, 420:440])
        interpolated_ice_y.loc[region_id,2006][100:175, 420:440] = inpaint.inpaint_biharmonic(interpolated_ice_y.loc[region_id,2006][100:175, 420:440], mask_nan_pi)

    ####################################################################



    #BLOCK 3

    #Calcualtating the ratios for each pixel on the whole ice #######################################
    #With ratio from now on I will intend the pixel-wise ratio between the ice velocity and the multi year velocity
    ratios_x = pd.DataFrame(index = df.index, columns = common_years) #this variable stores the ratio for each pixel (for each year) between the ice velocity (time dipendent) and the multi year velocity (not depending on time)

    for id in df.index:
        for year in common_years:
            ratios_x.loc[id, year] = interpolated_ice_x.loc[id, year] / velocity_multy_years_x.loc[id, 'image']


    #Now I set to 0 all the values which are not in the see but nan. Because in the multi year velocity we have 0 values not in the sea,
    #and the ratio is therefore impossible to calculate
    for id in df.index:
        for year in common_years:

            if coverage_ratio_x.loc[id, year]  > threshold_interpolation:

                mask_nan_and_not_sea = np.logical_and(~sea_mask.loc[id, year], np.isnan(ratios_x.loc[id, year]))
                ratios_x.loc[id, year][mask_nan_and_not_sea] = 0


    #Same for y
    ratios_y = pd.DataFrame(index = df.index, columns = common_years)

    for id in df.index:
        for year in common_years:

            ratios_y.loc[id, year] = interpolated_ice_y.loc[id, year] / velocity_multy_years_y.loc[id, 'image']

    for id in df.index:
        for year in common_years:

            if coverage_ratio_y.loc[id, year]  > threshold_interpolation:
    
                mask_nan_and_not_sea = np.logical_and(~sea_mask.loc[id, year], np.isnan(ratios_y.loc[id, year]))
                ratios_y.loc[id, year][mask_nan_and_not_sea] = 0

    ####################################################################
                

    #Reshaping the dataset in order to perform pixel by pixel interpolation. ############################     
    raveled_vector_x = np.zeros((len(df.index),len(ratios_x.loc[id, year].ravel()), len(common_years))) 

    index = 0
    for id in df.index:

        time = 0

        for year in common_years:

            matrix = ratios_x.loc[id, year]
            array_raveled = matrix.ravel()
            raveled_vector_x[index, :,time] = array_raveled
            time = time + 1

        index = index + 1


    #Redoing it for y
    raveled_vector_y = np.zeros((len(df.index),len(ratios_y.loc[id, year].ravel()), len(common_years)))
    index = 0

    for id in df.index:
            
            time = 0
        
            for year in common_years:
        
                matrix = ratios_y.loc[id, year]
                array_raveled = matrix.ravel()
                raveled_vector_y[index, :,time] = array_raveled
                time = time + 1
        
            index = index + 1

    ####################################################################



    print('Interpolation pixel by pixel')
    #pixel by pixel interpolation  x ########################################################

    for i in range(len(df.index)): #for each region
        for j in range(len(ratios_x.loc[id, year].ravel())): #for each pixel of the region

            #take the values of the pixel
            x = raveled_vector_x[i, j, :] #here I am taking the ratio time serie for each pixel

            #If the value is all nan, skip to the next pixel
            if np.isnan(x).all() == True:
                continue

            #interpolate the nan values as the average of the two closest values
            x = interpolation_excluding_extreames(x)

            x = fill_first_values(x)

            raveled_vector_x[i, j, :] = x

    #pixel by pixel interpolation  y
        
    for i in range(len(df.index)): #for each region
        for j in range(len(ratios_y.loc[id, year].ravel())): #for each pixel of the region

            #take the values of the pixel
            y = raveled_vector_y[i, j, :] #here I am taking the ratio time serie for each pixel

            #If the value is all nan, skip to the next pixel
            if np.isnan(y).all() == True:
                continue

            #interpolate the nan values as the average of the two closest values
            y = interpolation_excluding_extreames(y)

            y = fill_first_values(y)

            raveled_vector_y[i, j, :] = y
    ####################################################################


    #Re organising the interpolated ratio dataset
    interpolated_ratios_x = pd.DataFrame(index = df.index, columns = common_years)
    index_raveled = 0

    for id in df.index:

        time = 0

        for year in common_years:

            matrix = raveled_vector_x[index_raveled, :, time].reshape(velocity_multy_years_x.loc[id, 'image'].shape)
            interpolated_ratios_x.loc[id, year] = matrix
            time = time + 1

        index_raveled = index_raveled + 1

    #redoing it for y
    interpolated_ratios_y = pd.DataFrame(index = df.index, columns = common_years)
    index_raveled = 0

    for id in df.index:

        time = 0

        for year in common_years:

            matrix = raveled_vector_y[index_raveled, :, time].reshape(velocity_multy_years_y.loc[id, 'image'].shape)
            interpolated_ratios_y.loc[id, year] = matrix
            time = time + 1

        index_raveled = index_raveled + 1
    ####################################################################
        

    #re-creating the velocity dataset, combining the single year velocity and rescaling the multi year velocity with the interpolated ratio

    velocity_multi_and_single_x = pd.DataFrame(index = df.index, columns = common_years)
    velocity_multi_and_single_y = pd.DataFrame(index = df.index, columns = common_years)

    for id in df.index:
        for year in common_years:

            #Here we just use the spatial inteporlated images, where the coverage is good
            if coverage_ratio_x.loc[id, year] >= threshold_interpolation:
                velocity_multi_and_single_x.loc[id, year] = interpolated_ice_x.loc[id, year]

            #When the coverage is not sufficient we rescale the multi year velocity with the interpolated ratio  
            if coverage_ratio_x.loc[id, year] < threshold_interpolation:
                velocity_multi_and_single_x.loc[id, year] = interpolated_ratios_x.loc[id, year] * velocity_multy_years_x.loc[id, 'image']

            #Same for y
                
            if coverage_ratio_y.loc[id, year] >= threshold_interpolation:
                velocity_multi_and_single_y.loc[id, year] = interpolated_ice_y.loc[id, year]

            if coverage_ratio_y.loc[id, year] < threshold_interpolation:
                velocity_multi_and_single_y.loc[id, year] = interpolated_ratios_y.loc[id, year] * velocity_multy_years_y.loc[id, 'image']
    ####################################################################
            


    #BLOCK 4

    #Multy year velocity x and y have the same nans, so we can use the same mask for both.
    #This mask presents the regions where we have nan values in multi years BUT we have floating ice according to Green. 
    #In this way we identify the front advance, since we set the 0 values of the multi-year velocity to nan
    green_and_multi = pd.DataFrame(index = df.index, columns = common_years) ############################

    for id in df.index:

        #To cheange in velocity_multy_years_x.loc[id, 'image']==0 if it has not been set to nan
        mask_nan_in_multi_year = np.isnan(velocity_multy_years_x.loc[id, 'image'])

        for year in common_years:

                green_and_multi.loc[id,year] = np.logical_and(mask_nan_in_multi_year, ice_mask.loc[id, year])
    ####################################################################
                


    #Specific treatment for Amundsen bay region ########################################
    #Here i fill the boarders of the Thwaites ice shelf with the average velocity of the ice shelf region
    if region_id == 24:

        print('Thwaites specific treatment')

        #This first step is useful just for having a correct mask size
        board = boarders_mask.loc[24, 2008]
        ice = ice_mask.loc[24, 2008]

        board_ice = np.logical_and(board, ice)
                    
        #Cooking Thwaites and PI: I create boarder masks and I calculate the average velocity in the boarders
        from scipy import ndimage

        # Define the slices for the regions of interest
        region1_slice = (slice(235, 351), slice(336, 421))
        region2_slice = (slice(500, 686), slice(280,451))

        result_mask = np.full_like(board_ice, fill_value=False)

        # Extract the regions using the slices
        result_mask[region1_slice] = board_ice[region1_slice]
        result_mask[region2_slice] = board_ice[region2_slice]

        pine_island_boarders_average_x = pd.DataFrame(index = df.index, columns = common_years)
        thwaites_boarders_average_x = pd.DataFrame(index = df.index, columns = common_years)

        pine_island_boarders_average_y = pd.DataFrame(index = df.index, columns = common_years)
        thwaites_boarders_average_y = pd.DataFrame(index = df.index, columns = common_years)

        interpo_th_x = []
        interpo_th_y = []

        pine_island_boarder = pd.DataFrame(index = df.index, columns = common_years)
        thwaites_boarders = pd.DataFrame(index = df.index, columns = common_years)

        #Here I calculate the average velocity in the sliced regions

        for id in df.index:
            for year in common_years:

                #Here we look at the average velocity in the boarders of the two regions, usign the original data
                pine_island_boarders_average_x.loc[id,year] = np.nanmean(velocity_x_tif_smoothed.loc[id, year][region1_slice])
                thwaites_boarders_average_x.loc[id,year] = np.nanmean(velocity_x_tif_smoothed.loc[id, year][region2_slice])

                pine_island_boarders_average_y.loc[id,year] = np.nanmean(velocity_y_tif_smoothed.loc[id, year][region1_slice])
                thwaites_boarders_average_y.loc[id,year] = np.nanmean(velocity_y_tif_smoothed.loc[id, year][region2_slice])

                interpo_th_x.append(np.nanmean(velocity_x_tif_smoothed.loc[id, year][region2_slice]))
                interpo_th_y.append(np.nanmean(velocity_y_tif_smoothed.loc[id, year][region2_slice]))

                #Here we load the boarders
                board = boarders_mask.loc[id, year]
                ice = ice_mask.loc[id, year]

                #where we have both ice and boarders
                ice_board = np.logical_and(board, ice)

                #creating background for PI and Thwaites
                back_ground_pi = np.full_like(ice, fill_value=False)
                back_ground_th = np.full_like(ice, fill_value=False)

                #The slices for the regions of interest
                back_ground_pi[region1_slice] = ice_board[region1_slice]
                back_ground_th[region2_slice] = ice_board[region2_slice]

                #saving the boarders
                pine_island_boarder.loc[id, year] = back_ground_pi
                thwaites_boarders.loc[id, year] = back_ground_th


        #Here data coverage is minimal, so I will interpolate the values
        interpo_th_x[7] = np.nan 
        interpo_th_y[7] = np.nan

        #Make the interpolation of the missing boarders values
        interpo_th_x = np.array(interpo_th_x)
        interpo_th_x = interpolation_excluding_extreames(interpo_th_x)

        interpo_th_y = np.array(interpo_th_y)
        interpo_th_y = interpolation_excluding_extreames(interpo_th_y)
    ####################################################################



 
    #Emptying the Green and Multi mask regions and performing the spatial interpolation on those regions

    #Interpolation for x
    v_x_final = pd.DataFrame(index = df.index, columns = common_years)

    for id in df.index:
        for year in common_years:

            prova = velocity_multi_and_single_x.loc[id, year]
            prova[green_and_multi.loc[id, year]] = np.nan
            velocity_x_tif_nan.loc[id, year][green_and_multi.loc[id, year]] = True

            #Inserting manually the boarders for Thwaites
            if region_id == 24:
                prova[thwaites_boarders.loc[id,year]] = thwaites_boarders_average_x.loc[id, year]
                
            mask_nan = np.isnan(prova)
            prova = inpaint.inpaint_biharmonic(prova, mask_nan)
            prova[sea_mask.loc[id, year]] = np.nan           

            v_x_final.loc[id, year] = prova 


    #Same for y
    v_y_final = pd.DataFrame(index = df.index, columns = common_years)

    for id in df.index:
        for year in common_years:

            prova = velocity_multi_and_single_y.loc[id, year]
            prova[green_and_multi.loc[id, year]] = np.nan
            velocity_y_tif_nan.loc[id, year][green_and_multi.loc[id, year]] = True

            if region_id == 24:
                prova[thwaites_boarders.loc[id,year]] = thwaites_boarders_average_y.loc[id, year]

            mask_nan = np.isnan(prova)
            prova = inpaint.inpaint_biharmonic(prova, mask_nan)
            prova[sea_mask.loc[id, year]] = np.nan           

            v_y_final.loc[id, year] = prova 

####################################################################
            

####Creating masks for Twaithes shape############################################       
    if region_id == 24:

        region_th = (slice(350, 690), slice(300, 600)) #Thwaites

        mask_th = pd.DataFrame(index = df.index, columns = common_years)

        for id in df.index:
            for year in common_years:
                result_mask = np.full_like(ice_mask.loc[id,year], fill_value=False)
                result_mask[region_th] = ice_mask.loc[id,year][region_th]
                mask_th.loc[id, year] = result_mask
#########################################################################


    v_x_final_2 = pd.DataFrame(index = df.index, columns = common_years)
    v_y_final_2 = pd.DataFrame(index = df.index, columns = common_years)


    i = 0
    id = region_id

######### Another partcular treatment, for years which were particularly damaged in the dataset.    
    if region_id == 24:
        for year in [2011,2012,2013]:
            
            prova = v_x_final.loc[id, year]
            prova[green_and_multi.loc[id, year]] = np.nan

            #Here I empty the Thwaites region and I fill its boarders

            prova[mask_th.loc[id, year]] = np.nan
            prova[mask_th.loc[id, year]] = velocity_x_tif_smoothed.loc[id, year][mask_th.loc[id, year]]
            prova[thwaites_boarders.loc[id,year]] = interpo_th_x[i]

            mask_nan = np.isnan(prova)
            velocity_x_tif_nan.loc[id, year][mask_nan] = True
            prova = inpaint.inpaint_biharmonic(prova, mask_nan)
            prova[sea_mask.loc[id, year]] = np.nan

            v_x_final_2.loc[id, year] = prova

            #Same for y

            prova = v_y_final.loc[id, year]
            prova[green_and_multi.loc[id, year]] = np.nan

            #Here I empty the Thwaites region and I fill its boarders
            if region_id == 24:
                prova[mask_th.loc[id, year]] = np.nan
                prova[mask_th.loc[id, year]] = velocity_y_tif_smoothed.loc[id, year][mask_th.loc[id, year]]
                prova[thwaites_boarders.loc[id,year]] = interpo_th_y[i]

            mask_nan = np.isnan(prova)
            velocity_y_tif_nan.loc[id, year][mask_nan] = True
            prova = inpaint.inpaint_biharmonic(prova, mask_nan)
            prova[sea_mask.loc[id, year]] = np.nan

            v_y_final_2.loc[id, year] = prova

            i = i + 1


    #Modifying the final dataset for the years 2011, 2012, 2013 for the region of Amundsen Bay
            
    v_x_final_3 = pd.DataFrame(index = df.index, columns = common_years)
    v_y_final_3 = pd.DataFrame(index = df.index, columns = common_years)

    for id in df.index:
        for year in common_years:

            v_x_final_3.loc[id, year] = v_x_final.loc[id, year]
            v_y_final_3.loc[id, year] = v_y_final.loc[id, year]



            if region_id == 24: #specific condition for Pine Island
                if year == 2011 or year == 2012 or year == 2013:

                    v_x_final_3.loc[id, year] = v_x_final_2.loc[id, year]
                    v_y_final_3.loc[id, year] = v_y_final_2.loc[id, year]


    for id in df.index:
        for year in common_years:
    
                v_x_final_3.loc[id, year][sea_mask.loc[id, year]] = 0
                v_y_final_3.loc[id, year][sea_mask.loc[id, year]] = 0  

                velocity_x_tif_nan.loc[id, year][sea_mask.loc[id,year]] = False
                velocity_y_tif_nan.loc[id, year][sea_mask.loc[id,year]] = False

    velocity = pd.DataFrame(index = df.index, columns = common_years)

    for id in df.index:
        for year in common_years:

            velocity.loc[id, year] = np.sqrt(v_x_final_3.loc[id, year]**2 + v_y_final_3.loc[id, year]**2)
    ####################################################################
            
    #Saving the dataset as a npy file
    saving_directory = '/bettik/moncadaf/dataset/produced_data/velocity/'

    np.save(saving_directory + 'v_region_' + str(region_id) + '.npy', velocity)
    np.save(saving_directory + 'v_x_region_' + str(region_id) + '.npy', v_x_final_3)
    np.save(saving_directory + 'v_y_region_' + str(region_id) + '.npy', v_y_final_3)

    saving_directory_nan = '/bettik/moncadaf/dataset/produced_data/interpolated_areas/velocity/'
    np.save(saving_directory_nan + 'v_x_nan_region_' + str(region_id) + '.npy', velocity_x_tif_nan)
    np.save(saving_directory_nan + 'v_y_nan_region_' + str(region_id) + '.npy', velocity_y_tif_nan)

    print('Region', region_id, 'saved for ice velocity')

