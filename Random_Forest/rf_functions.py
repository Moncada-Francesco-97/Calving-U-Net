from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV


#Thiis function is used to get the best parameters for the random forest 

def get_rf_best_params(X, y, cv_split, param_grid):

    '''Parameters:
    X: the features
    y: the target
    cv_split: the cross validation split
    param_grid: the grid of parameters to test'''

    #defining the model
    rf = RandomForestRegressor(random_state=42, n_jobs=-1)

    #Perform the grid search
    grid_search = GridSearchCV(estimator=rf, param_grid= param_grid, cv=cv_split, verbose=0, n_jobs=-1, scoring='neg_mean_squared_error') #lower values are better

    print(grid_search)

    #Fitting the model
    grid_search.fit(X, y)

    max_depth = grid_search.best_params_['max_depth']
    n_estimators = grid_search.best_params_['n_estimators']
    min_samples_leaf = grid_search.best_params_['min_samples_leaf']
    min_samples_split = grid_search.best_params_['min_samples_split']

    return max_depth, n_estimators, min_samples_leaf, min_samples_split

def print_merda():
	print('Diocane')
    



        