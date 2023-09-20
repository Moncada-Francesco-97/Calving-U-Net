import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
import os
import seaborn as sns
import fiona
import sklearn
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score

#My libraries
import importlib
import rf_functions 
import rf_plot 

#Upload the selection of glacier i will use in the model
selection = []

with open('/bettik/moncadaf/data' + '/selecao.txt', 'r') as f:
    for line in f:
        selection.append(int(line.strip()))

common_years = np.arange(2005,2017,1)

print('The shape of the selection is: ', np.shape(selection))



#Extracting the info from the shapefile################
shapefile_path = '/bettik/moncadaf/data/shapefiles_antarctica/ice_shelf.shp'

ids = []
Names = []
regions = []
areas = []
lats = []
lons = []

shapefile=fiona.open(shapefile_path)
print(shapefile.schema['properties'])

for feature in shapefile:

    id = feature['properties']['id']
    name=feature['properties']['name']
    area=feature['properties']['area_fra']
    lat = feature['properties']['latitude']
    lon = feature['properties']['longitude']
    geometry=feature['geometry']
    if geometry is None:
        continue
    region = feature['properties']['regions']

    #Taking the info
    ids.append(id)
    Names.append(name)
    regions.append(region)
    areas.append(area)
    lats.append(lat)
    lons.append(lon)

#Creating the dataframe
df_region = pd.DataFrame(index=ids)
df_region ['id'] = ids
df_region ['Name'] = Names
df_region ['Region'] = regions
df_region ['Area'] = areas
df_region ['Latitude'] = lats
df_region ['Longitude'] = lons

#Selecting the region and sorting the dataframe
df_region = df_region[df_region['id'].isin(selection)]
df_region = df_region.sort_values(by=['id'])
#drop id column
df_region = df_region.drop(['id'], axis=1)
print('I have etracted the info from the shapefile')
############################################


#Loading the dataset###############################
dataset_directory = '/bettik/moncadaf/data/dataset_filtered/'

# Basal Melting
bm = pd.read_csv(dataset_directory + '/bm.csv', index_col=0)
bm = bm.sort_values(by=['id']) #sorting the glaciers by their index
bm = bm.loc[bm.index.isin(selection)] #selecting the glaciers, according to their index
bm = bm[common_years.astype(str)] #selecting the common years
bm = bm.sort_index() #sorting the glaciers by their index

#Load the calving data
calving = pd.read_csv(dataset_directory+ '/df_calving_from_shp_negative_and_positive.csv', index_col=0)
calving = calving.loc[calving.index.isin(selection)]
calving = calving[common_years.astype(str)]
#sort the calving by its index
calving = calving.sort_index()

#Load the ice concentration data
i_c = pd.read_csv(dataset_directory + '/ice_c_avg_extended_front.csv', index_col=0)
i_c = i_c.loc[i_c.index.isin(selection)]
i_c = i_c[common_years.astype(str)]
i_c = i_c.sort_index()

#Load the ice velocity data
i_v = pd.read_csv(dataset_directory + '/velocity_80_percentile_extended_front_2011_2012_linear_trend.csv', index_col=0)
i_v = i_v.loc[i_v.index.isin(selection)]
i_v = i_v[common_years.astype(str)]
i_v = i_v.sort_index()

#Load the ice thickness data
i_t = pd.read_csv(dataset_directory + '/thickness_avg_extended front.csv', index_col=0)
i_t = i_t.loc[i_t.index.isin(selection)]
i_t = i_t[common_years.astype(str)]
i_t = i_t.sort_index()

index = bm.index

print('I have loaded all the datasets')
####################################################




#make a dataset with all the variables################
dataset = pd.concat([bm, calving, i_c, i_v, i_t], axis=1, keys=['bm', 'calving', 'i_c', 'i_v', 'i_t'])
dataset.columns.names = ['Variables', 'Years']
dataset.index.names = ['Glaciers']

#Folder division according to the blocks parameters
folder_1 = [2,4,6,7,8,15,18,16,34,45,135,163,95]
folder_2 = [3,29,10,12,31,30,19,24,35,50,86,104,139]
folder_3 = [65,43,37,17,41,32,38,40,36,52,143,146,125]
folder_4 = [69,67,63,26,46,33,64,44,70,53,123,145,170]
folder_5 = [75,78,89,39,47,81,77,54,71,55,118,101,117]
folder_6 = [76,80,114,58,48,84,83,66,90,56,140,144,164]
folder_7 = [88,82,120,61,57,92,85,105,100,73,158,124,169]
folder_8 = [96,107,121,62,60,127,87,108,102,91,161,122,160]

cv_folders = [folder_1, folder_2, folder_3, folder_4, folder_5, folder_6, folder_7, folder_8]

folder_9 = [129,112,156,68,97,132,93,109,115,98,148,166] #test
folder_10 = [131,147,157,72,99,136,116,111,119,110,162,159] #test

#merge the folders, in order to have a single list
from itertools import chain

train = [folder_1, folder_2, folder_3, folder_4, folder_5, folder_6, folder_7, folder_8]
test = [folder_9, folder_10]
test = list(chain.from_iterable(test))

#Operate block division
cv_block_1 = dataset.loc[folder_1]
cv_block_2 = dataset.loc[folder_2]
cv_block_3 = dataset.loc[folder_3]
cv_block_4 = dataset.loc[folder_4]
cv_block_5 = dataset.loc[folder_5]
cv_block_6 = dataset.loc[folder_6]
cv_block_7 = dataset.loc[folder_7]
cv_block_8 = dataset.loc[folder_8]

cv_blocks = [cv_block_1, cv_block_2, cv_block_3, cv_block_4, cv_block_5, cv_block_6, cv_block_7, cv_block_8]
test_block = dataset.loc[test]

cv = pd.concat(cv_blocks)
######################################################



#Grid Search and Cross Validation#######################
importlib.reload(rf_functions)  # Reload the module
from rf_functions import rf_train_and_fit

grid = {
    'n_estimators': [10, 30, 50],
    'max_depth': [2,5],
    'min_samples_split': [2, 5, 10], #no 1
    'min_samples_leaf': [2,5,10],
    #'criterion': ['squared_error'],
}

X = cv.drop('calving', axis=1)
y = cv['calving']

cv_split = KFold(n_splits=8, shuffle=False, random_state=None)


rf_trained = rf_train_and_fit(X, y, cv_split, grid, 'neg_mean_squared_error')

print('The best max_depth is: ', rf_trained.best_params_['max_depth']
        , 'The best n_estimators is: ', rf_trained.best_params_['n_estimators']
        , 'The best min_samples_split is: ', rf_trained.best_params_['min_samples_split']
        , 'The best min_samples_leaf is: ', rf_trained.best_params_['min_samples_leaf'])


#defining the model with the best parameters
fitted_rf = sklearn.ensemble.RandomForestRegressor(criterion='squared_error',
                                                    max_depth=rf_trained.best_params_['max_depth'],
                                                    n_estimators=rf_trained.best_params_['n_estimators'],
                                                    min_samples_split=rf_trained.best_params_['min_samples_split'],
                                                    min_samples_leaf=rf_trained.best_params_['min_samples_leaf'],
                                                    random_state=42,
                                                    n_jobs=-1
                                                    )

cvl = cross_val_score(fitted_rf, X, y, cv=cv_split, scoring='neg_mean_squared_error', n_jobs=-1)

print('The mean of the cross validation is: ', np.mean(cvl))
print('The std of the cross validation is: ', np.std(cvl))


#Plottiing the results
importlib.reload(rf_plot)  # Reload the module
from rf_plot import plot_gsearch_results

save_dir='/bettik/moncadaf/data/outputs/machine_learning_calving_project/Random_Forest'

plot_gsearch_results(rf_trained, save_dir= save_dir, save_filename='gsearch_results.png')

#Plot the feature importance
importlib.reload(rf_plot)  # Reload the module
from rf_plot import plot_feature_importance

plot_feature_importance(fitted_rf, X, save_dir=save_dir)
