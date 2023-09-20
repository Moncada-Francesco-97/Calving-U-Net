from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV



#Thiis function is used to get the best parameters for the random forest 

def rf_train_and_fit(X, y, cv_split, param_grid, scoring_criterium):

    '''Parameters:
    X: the features
    y: the target
    cv_split: the cross validation split
    param_grid: the grid of parameters to test
    scoring_criterium: the scoring criterium to use for the grid search'''

    #defining the model
    rf = RandomForestRegressor(random_state=42, n_jobs=-1)

    #Perform the grid search
    grid_search = GridSearchCV(estimator=rf, param_grid= param_grid, cv=cv_split, verbose=2, n_jobs=-1, scoring= str(scoring_criterium)) #lower values are better

    print('The parameters to test are: ', param_grid, '\n')
    #'The cross validation score is : ', grid_search.best_score_, '\n'
    #'The best parameters are: ', grid_search.best_params_)
    

    #Fitting the model
    grid_search.fit(X, y)

    max_depth = grid_search.best_params_['max_depth']
    n_estimators = grid_search.best_params_['n_estimators']
    min_samples_leaf = grid_search.best_params_['min_samples_leaf']
    min_samples_split = grid_search.best_params_['min_samples_split']

    return grid_search
    



        