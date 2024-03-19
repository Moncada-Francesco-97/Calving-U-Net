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


#Functions ########################################
def interpolation_excluding_extreames(A):

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

    A = np.array(A)

    for i in range(len(A)):

        if np.isnan(A[i]) == False:
            A[:i] = A[i]
            break

    return A
####################################################################

common_years = np.arange(2005,2017)
threshold_interpolation = 0.7

#Get information from the shapefile
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



#Load the masks ########################################################
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

####################################################################


#MULTI YEAR VELOCITY ########################################################
velocity_multy_years_x_path = '/UPDATE/velocity_multi_years_X.tif'
velocity_multy_years_y_path = '/UPDATE/velocity_multi_years_Y.tif'

#dataset
velocity_multy_years_x =pd.DataFrame(index = df.index, columns = ['image'])
velocity_multy_years_y =pd.DataFrame(index = df.index, columns = ['image'])

#Extract the velocity data from the multi year dataset
for id in df.index:

        with rasterio.open(velocity_multy_years_x_path, crs = 'EPSG:3031') as src:
                xmin, ymin, xmax, ymax = df.loc[id, 'boundaries']
                window = rasterio.windows.from_bounds(xmin, ymin, xmax, ymax, src.transform)
                velocity_multy_years_x.loc[id, 'image'] = src.read(1, window=window)
                velocity_multy_years_x.loc[id, 'image'][velocity_multy_years_x.loc[id, 'image']==0] = np.nan

        with rasterio.open(velocity_multy_years_y_path, crs = 'EPSG:3031') as src:
                xmin, ymin, xmax, ymax = df.loc[id, 'boundaries']
                window = rasterio.windows.from_bounds(xmin, ymin, xmax, ymax, src.transform)
                velocity_multy_years_y.loc[id,'image'] = src.read(1, window=window)
                velocity_multy_years_y.loc[id, 'image'][velocity_multy_years_y.loc[id, 'image']==0] = np.nan

####################################################################


#EXTRACTING SINGLE YEAR VELOCITY ########################################
list_x = []
list_y = []

#Velocity
velocity_x_tif = pd.DataFrame(index = df.index, columns = common_years)
velocity_y_tif = pd.DataFrame(index = df.index, columns = common_years)

#FILE LIST (change in cluster)
for i in common_years:
    list_x.append('/UPDATE/' + f"{i}_{i+1}_VX.tif")
    list_y.append('/UPDATE/' + f"{i}_{i+1}_VY.tif")

#calculate the velocity for each year, for each region
for name_x, name_y, year in zip(list_x, list_y, common_years): #for each year

    for id in df.index: #for each region

        xmin, ymin, xmax, ymax = df.loc[id, 'boundaries']

        with rasterio.open(name_x, crs = 'EPSG:3031') as src:
            window = rasterio.windows.from_bounds(xmin, ymin, xmax, ymax, src.transform)
            image_x = src.read(1, window=window)

            #Exclude the 0.01 and 99.99 percentile
            values = np.nanpercentile(image_x.ravel(), [0.05, 99.95])
            image_x[image_x < values[0]] = np.nan
            image_x[image_x > values[1]] = np.nan

            #set to nan all the 0 values
            image_x[image_x == 0] = np.nan
            velocity_x_tif.loc[id, year] = image_x

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

        #thresholding
        velocity_x_tif_smoothed.loc[id, year][abs(velocity_x_tif_smoothed.loc[id, year] - vx_tmp_smoothed) > threshold] = np.nan
        velocity_y_tif_smoothed.loc[id, year][abs(velocity_y_tif_smoothed.loc[id, year] - vy_tmp_smoothed) > threshold] = np.nan

####################################################################



#Create the masks for the ice shelves ########################
from scipy.ndimage import label

ice_shelves_masks = pd.DataFrame(index = df.index, columns = common_years)
number_of_ice_shelves = pd.DataFrame(index = df.index, columns = common_years)

for id in df.index:
    for year in common_years:
        
        ice_shelf_mask = ice_mask.loc[id, year]
        ice_shelf_grounded_ice_mask = grounded_ice_mask.loc[id, year]

        combined_mask = ice_shelf_mask | ice_shelf_grounded_ice_mask

        # Perform connected component labeling on the mask
        labeled_mask, num_labels = label(combined_mask)

        # Create a new array to hold the labeled mask with unique labels for each ice shelf
        labeled_combined = np.zeros_like(labeled_mask)

        # Assign unique labels to each ice shelf
        for label_idx in range(1, num_labels + 1):  # Start from 1 to exclude background label
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
        prova_id = number_of_ice_shelves.loc[id,year]

        prova_filtered = prova.copy()
        ice_shelves_numbers = np.arange(1,prova_id+1,1)

        #From here we are working on the single ice shelf
        j = 0

        for i in range(1,prova_id+1):
            #print(i)

            ice_shelf_mask = prova == ice_shelves_numbers[j] #Get the mask of ice shelf number n

            ice_shelf_mask = ice_shelf_mask.astype(bool) #convert ice_shelf_mask to boolean

            #Select the data from data_in_ice_shelf which are in the region ice_shelf_mask
            data_in_ice_shelf_roi = np.where(ice_shelf_mask, velocity_x_tif.loc[id, year], np.nan)

            # Count the number of non-NaN pixels in the masked data_in_ice_shelf_roi
            num_non_nan_pixels = np.sum(~np.isnan(data_in_ice_shelf_roi))

            # Count the number of True pixels in the ice_shelf_mask
            num_true_pixels = np.sum(ice_shelf_mask)

            # Calculate the ratio of non-NaN pixels to True pixels
            ratio_non_nan_to_true = num_non_nan_pixels / num_true_pixels
            #print('Ratio of non-NaN pixels to True pixels:', ratio_non_nan_to_true)

            if ratio_non_nan_to_true < 0.7:

                prova_filtered = np.where(prova_filtered == ice_shelves_numbers[j], 0, prova_filtered)
                #print('Ice shelf number ', ice_shelves_numbers[i], ' has been filtered out')
            j = j+1

        masks_filtered.loc[id,year] = prova_filtered

#Convert all the mask to the same boolean format
for id in df.index:
    for year in common_years:
        masks_filtered.loc[id,year] = masks_filtered.loc[id,year].astype(bool)

####################################################################
        

#interpolate into the ice shelf mask ########################################
from skimage.restoration import inpaint

interpolated_floating_ice_x = pd.DataFrame(index = df.index, columns = common_years)

for id in df.index:
    for year in common_years:
        print(year)
        
        data_tmp = np.where(masks_filtered.loc[id,year], velocity_x_tif_smoothed.loc[id,year], np.nan)
        data_tmp = np.where(~masks_filtered.loc[id,year], velocity_x_tif_smoothed.loc[id,year], data_tmp)
        nan_mask = np.isnan(data_tmp)

        interpolated_floating_ice_x.loc[id,year] = inpaint.inpaint_biharmonic(data_tmp, nan_mask)
        interpolated_floating_ice_x.loc[id,year][~masks_filtered.loc[id,year]] = np.nan



#Now i interpolate the grounded ice for x
interpolated_ice_x = pd.DataFrame(index = df.index, columns = common_years)
coverage_ratio_x = pd.DataFrame(index = df.index, columns = common_years)

for id in df.index:
    for year in common_years:

        #load original data for the land
        data_tmp = velocity_x_tif_smoothed.loc[id,year]

        #load the  interpolated data for the floating ice
        data_tmp_all_ice = np.where(masks_filtered.loc[id,year], interpolated_floating_ice_x.loc[id,year], data_tmp)
        nan_mask = np.isnan(data_tmp_all_ice)

        #Calculate the coverage ratio
        num_non_nan_pixels = np.sum(~np.isnan(data_tmp_all_ice))
        num_true_pixels = np.sum(~sea_mask.loc[id,year])

        coverage_ratio_x.loc[id,year] = num_non_nan_pixels/num_true_pixels

        print('The ratio of non-NaN pixels to True pixels for the year', year, 'is', coverage_ratio_x.loc[id,year])

        interpolated_ice_x.loc[id,year] = data_tmp_all_ice

        if coverage_ratio_x.loc[id,year] > 0.7:

            interpolated_ice_x.loc[id,year] = inpaint.inpaint_biharmonic(data_tmp_all_ice, nan_mask)
            interpolated_ice_x.loc[id,year][sea_mask.loc[id,year]] = np.nan     

#Redoing it for y
interpolated_floating_ice_y = pd.DataFrame(index = df.index, columns = common_years)

for id in df.index:
    for year in common_years:
        print(year)
        
        data_tmp = np.where(masks_filtered.loc[id,year], velocity_y_tif_smoothed.loc[id,year], np.nan)
        data_tmp = np.where(~masks_filtered.loc[id,year], velocity_y_tif_smoothed.loc[id,year], data_tmp)
        nan_mask = np.isnan(data_tmp)

        interpolated_floating_ice_y.loc[id,year] = inpaint.inpaint_biharmonic(data_tmp, nan_mask)
        interpolated_floating_ice_y.loc[id,year][~masks_filtered.loc[id,year]] = np.nan


#Now i interpolate the grounded ice

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

        print('The ratio of non-NaN pixels to True pixels for the year', year, 'is', coverage_ratio_y.loc[id,year])

        interpolated_ice_y.loc[id,year] = data_tmp_all_ice

        if coverage_ratio_y.loc[id,year] > 0.7:

            interpolated_ice_y.loc[id,year] = inpaint.inpaint_biharmonic(data_tmp_all_ice, nan_mask)
            interpolated_ice_y.loc[id,year][sea_mask.loc[id,year]] = np.nan

####################################################################



#Specific treatment for Pine Island in year 2006 

mask_nan_pi = np.isnan(interpolated_ice_y.loc[1,2006][150:225, 790:812])
interpolated_ice_y.loc[1,2006][150:225, 790:812] = inpaint.inpaint_biharmonic(interpolated_ice_y.loc[1,2006][150:225, 790:812], mask_nan_pi)

####################################################################


#Calcualtating the ratios on the whole ice #######################################
ratios_x = pd.DataFrame(index = df.index, columns = common_years)

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
            

#Reshaping the dataset in order to perform pixel by pixel interpolation ############################     
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




#step by step interpolation  x ########################################################

for i in range(len(df.index)): #for each region
   for j in range(len(ratios_x.loc[id, year].ravel())): #for each pixel of the region

      #take the values of the pixel
      x = raveled_vector_x[i, j, :] #here I am taking the ratio time serie for each pixel

      #If the value is all nan, skip to the next pixel
      if np.isnan(x).all() == True:
          continue

      #interpolate the nan values as the average of the two closest values
      x = interpolation_excluding_extreames(x)

      x = fill_fist_values(x)

      raveled_vector_x[i, j, :] = x

#step by step interpolation  y
      
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
    

#re-creating the velocity dataset, with holes

velocity_multi_and_single_x = pd.DataFrame(index = df.index, columns = common_years)
velocity_multi_and_single_y = pd.DataFrame(index = df.index, columns = common_years)

for id in df.index:
    for year in common_years:

        #Here we just use the inteporlated images, where the coverage is good
        if coverage_ratio_x.loc[id, year] >= threshold_interpolation:
            velocity_multi_and_single_x.loc[id, year] = interpolated_ice_x.loc[id, year]
            
        if coverage_ratio_x.loc[id, year] < threshold_interpolation:
            velocity_multi_and_single_x.loc[id, year] = interpolated_ratios_x.loc[id, year] * velocity_multy_years_x.loc[id, 'image']

        if coverage_ratio_y.loc[id, year] >= threshold_interpolation:
            velocity_multi_and_single_y.loc[id, year] = interpolated_ice_y.loc[id, year]

        if coverage_ratio_y.loc[id, year] < threshold_interpolation:
            velocity_multi_and_single_y.loc[id, year] = interpolated_ratios_y.loc[id, year] * velocity_multy_years_y.loc[id, 'image']

velocity_multi_and_single = pd.DataFrame(index = df.index, columns = common_years)

for id in df.index:
    for year in common_years:

        velocity_multi_and_single.loc[id, year] = np.sqrt(velocity_multi_and_single_x.loc[id, year]**2 + velocity_multi_and_single_y.loc[id, year]**2)
####################################################################
        

#Multy year velocity x and y have the same nans, so we can use the same mask for both.
#This mask presents the regions where we have nan in multi years AND floating ice
green_and_multi = pd.DataFrame(index = df.index, columns = common_years) ############################

for id in df.index:

    #To chenge in velocity_multy_years_x.loc[id, 'image']==0 if it has not been set to nan
    mask_nan_in_multi_year = np.isnan(velocity_multy_years_x.loc[id, 'image'])

    for year in common_years:

            #check when velocity_multy_years_x.loc[id, 'image'][ice_mask.loc[id, year]] is 0
            #velocity_01_x.loc[id, year][ice_mask.loc[id, year]] = 0
            green_and_multi.loc[id,year] = np.logical_and(mask_nan_in_multi_year, ice_mask.loc[id, year])
####################################################################
            


#Specific treatment for Thwaites ########################################
#Here i fill the boarders of the Thwaites ice shelf with the average velocity of the ice shelf region

#This first step is useful just for having a correct mask size
board = boarders_mask.loc[1, 2008]
ice = ice_mask.loc[1, 2008]
board_ice = np.logical_and(board, ice)
            
# Define the slices for the regions of interest
region1_slice = (slice(280, 401), slice(710, 791))
region2_slice = (slice(580, 741), slice(660,826))

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

for id in df.index:
    for year in common_years:

        #Here we look at the average velocity in the boarders of the two regions, usign theoriginal data
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

        #creating background for PI and Th
        back_ground_pi = np.full_like(ice, fill_value=False)
        back_ground_th = np.full_like(ice, fill_value=False)

        #The slices for the regions of interest
        back_ground_pi[region1_slice] = ice_board[region1_slice]
        back_ground_th[region2_slice] = ice_board[region2_slice]

        #Expand the boarders of 10 pixels
        #back_ground_pi = ndimage.binary_dilation(back_ground_pi, iterations=iterations)
        #back_ground_th = ndimage.binary_dilation(back_ground_th, iterations=iterations)

        #saving the boarders
        pine_island_boarder.loc[id, year] = back_ground_pi
        thwaites_boarders.loc[id, year] = back_ground_th


#Make the interpolation of the missing boarders values
interpo_th_x = np.array(interpo_th_x)
interpo_th_x = interpolation_excluding_extreames(interpo_th_x)

interpo_th_y = np.array(interpo_th_y)
interpo_th_y = interpolation_excluding_extreames(interpo_th_y)
####################################################################


#Emptying the Green and Multi mask regions and performing the inteprolation on those regions

#Interpolation for x
v_x_final = pd.DataFrame(index = df.index, columns = common_years)

for id in df.index:
    for year in common_years:

        print(year)

        prova = velocity_multi_and_single_x.loc[id, year]
        prova[green_and_multi.loc[id, year]] = np.nan
        prova[thwaites_boarders.loc[id,year]] = thwaites_boarders_average_x.loc[id, year]
        #prova[pine_island_boarder.loc[id,year]] = pine_island_boarders_average_x.loc[id, year]
        mask_nan = np.isnan(prova)
        prova = inpaint.inpaint_biharmonic(prova, mask_nan)
        prova[sea_mask.loc[id, year]] = np.nan           

        v_x_final.loc[id, year] = prova 


#Interpolation for y
v_y_final = pd.DataFrame(index = df.index, columns = common_years)

for id in df.index:
    for year in common_years:

        print(year)

        prova = velocity_multi_and_single_y.loc[id, year]
        prova[green_and_multi.loc[id, year]] = np.nan
        prova[thwaites_boarders.loc[id,year]] = thwaites_boarders_average_y.loc[id, year]
        #prova[pine_island_boarder.loc[id,year]] = pine_island_boarders_average_x.loc[id, year]
        mask_nan = np.isnan(prova)
        prova = inpaint.inpaint_biharmonic(prova, mask_nan)
        prova[sea_mask.loc[id, year]] = np.nan           

        v_y_final.loc[id, year] = prova 
####################################################################
        

#Modyfiyng Thwaites region for the year 2011, 2012, 2013,
#to maximise the use of the dataset.
        
region_th = (slice(380, 741), slice(600, 1001)) #Thwaites
mask_th = pd.DataFrame(index = df.index, columns = common_years)

for id in df.index:
    for year in common_years:
        result_mask = np.full_like(ice_mask.loc[id,year], fill_value=False)
        result_mask[region_th] = ice_mask.loc[id,year][region_th]
        mask_th.loc[id, year] = result_mask


v_x_final_2 = pd.DataFrame(index = df.index, columns = common_years)
v_y_final_2 = pd.DataFrame(index = df.index, columns = common_years)

i = 0
for year in [2011,2012,2013]:
    
    prova = v_x_final.loc[1, year]
    prova[green_and_multi.loc[id, year]] = np.nan
    prova[mask_th.loc[id, year]] = np.nan
    prova[mask_th.loc[id, year]] = velocity_x_tif_smoothed.loc[id, year][mask_th.loc[id, year]]
    prova[thwaites_boarders.loc[id,year]] = interpo_th_x[i]

    mask_nan = np.isnan(prova)
    prova = inpaint.inpaint_biharmonic(prova, mask_nan)
    prova[sea_mask.loc[id, year]] = np.nan

    v_x_final_2.loc[1, year] = prova

    #Same for y

    prova = v_y_final.loc[1, year]
    prova[green_and_multi.loc[id, year]] = np.nan
    prova[mask_th.loc[id, year]] = np.nan
    prova[mask_th.loc[id, year]] = velocity_y_tif_smoothed.loc[id, year][mask_th.loc[id, year]]
    prova[thwaites_boarders.loc[id,year]] = interpo_th_y[i]

    mask_nan = np.isnan(prova)
    prova = inpaint.inpaint_biharmonic(prova, mask_nan)
    prova[sea_mask.loc[id, year]] = np.nan

    v_y_final_2.loc[1, year] = prova

    i = i + 1


#Modifying the final dataset for the years 2011, 2012, 2013
v_x_final_3 = pd.DataFrame(index = df.index, columns = common_years)
v_y_final_3 = pd.DataFrame(index = df.index, columns = common_years)

for id in df.index:
    for year in common_years:

        v_x_final_3.loc[id, year] = v_x_final.loc[id, year]
        v_y_final_3.loc[id, year] = v_y_final.loc[id, year]

        if year == 2011 or year == 2012 or year == 2013:
            v_x_final_3.loc[id, year] = v_x_final_2.loc[id, year]
            v_y_final_3.loc[id, year] = v_y_final_2.loc[id, year]

velocity = pd.DataFrame(index = df.index, columns = common_years)

for id in df.index:
    for year in common_years:

        velocity.loc[id, year] = np.sqrt(v_x_final_3.loc[id, year]**2 + v_y_final_3.loc[id, year]**2)
####################################################################
        
#Saving the dataset as a npy file

np.save('/UPDATE/velocity.npy', velocity)
np.save('/UPDATE/v_x.npy', v_x_final_3)
np.save('/UPDATE/v_y.npy', v_y_final_3)