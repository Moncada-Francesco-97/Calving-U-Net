from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV



#Thiis function is used to get the best parameters for the random forest
#It returns a fitted GridSearchCV object

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

    #Fitting the model
    grid_search.fit(X, y)

    return grid_search
    



        