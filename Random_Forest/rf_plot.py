import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
import os
import pandas as pd
import seaborn as sns
import sklearn



def plot_gsearch_results(grid, save_dir=None, save_filename=None):
    """
    Params: 
        grid: A fitted GridSearchCV object. It should include no criterium
        save_dir: The directory where the plot will be saved
        save_filename: The filename of the plot
    """
    ## Results from grid search
    results = grid.cv_results_
    grid.cv_results_
    means_test = results['mean_test_score'] #mean_test_score for all the runs
    stds_test = results['std_test_score']
    #means_train = results['mean_train_score'] Not available in sklearn 0.20
    #stds_train = results['std_train_score']

    ## Getting indexes of values per hyper-parameter
    masks=[]
    masks_names= list(grid.best_params_.keys()) #grid.best_params_.keys() is a dict_keys object. It includes the name of all the parameters used in the grid search

    for p_k, p_v in grid.best_params_.items(): #Here we are iterating in parallel over parameter (p_k) and its corresponding value (p_v)
        masks.append(list(results['param_'+p_k].data==p_v)) #dimension is (n_params, n_runs)

    params=grid.param_grid

    width = len(grid.best_params_.keys())*5

    ## Ploting results
    fig, ax = plt.subplots(1,len(params),sharex='none', sharey='all',figsize=(width,5))
    fig.suptitle('Score per parameter')
    fig.text(0.04, 0.5, 'MEAN SCORE', va='center', rotation='vertical')
    
    for i, p in enumerate(masks_names): #i is the index of the parameter, p is the name of the parameter

        m = np.stack(masks[:i] + masks[i+1:]) #here i select all the masks except the one of the parameter that i am plotting
        best_parms_mask = m.all(axis=0) # Here I check when ALL the other parameters are equal to their best value (looking at columns, when all the values in column are true)
        best_index = np.where(best_parms_mask)[0]
        x = np.array(params[p])
        y_1 = np.array(means_test[best_index])
        e_1 = np.array(stds_test[best_index])
        #y_2 = np.array(means_train[best_index])
        #e_2 = np.array(stds_train[best_index])
        title = 'Best value of ' + p + ' is: ' + str(grid.best_params_[p])
        ax[i].set_title(title)
        ax[i].errorbar(x, y_1, e_1, linestyle='--', marker='o', label='test')
        #ax[i].errorbar(x, y_2, e_2, linestyle='-', marker='^',label='train' )
        ax[i].set_xlabel(p.upper())
        ax[i].grid()

    plt.legend()

    # Save the plot if save_dir and save_filename are provided
    if save_dir and save_filename:
        save_path = os.path.join(save_dir, save_filename)
        plt.savefig(save_path, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
    else:
        plt.show()

    plt.close()  # Close the plot to free up resources


''' Check the function all() in numpy

prova = np.array([[True, False, True],
                  [True, True, False],
                  [True, True, True]])

test = prova.all(axis=0)

test

Output: array([ True, False, False])
'''
    
def plot_feature_importance(rf_trained, X, variables_new = None, save_dir=None):

    '''Parameters:
    fitted_rf: the fitted random forest
    X: the features
    variables_new: potentially the new variables we will use: otherwise we use bm, i_c, i_v, i_t
    save_dir: the directory where the plot will be saved'''

    #First graph

    #Extract just the variable and not the year from the column
    best_rf_estimator = rf_trained.best_estimator_

    # Access the feature importances from the best estimator
    feature_importance = best_rf_estimator.feature_importances_
    sorted_idx = np.argsort(feature_importance) #is sorting the columns
    pos = np.arange(sorted_idx.shape[0]) + 0.5 #just for the graphic
    fig = plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.barh(pos, feature_importance[sorted_idx], align="center")
    plt.yticks(pos, np.array(X.columns)[sorted_idx])
    plt.title("Feature Importance (MDI)")

    #Saving the plot
    if save_dir:
        save_path = os.path.join(save_dir, 'feature_importance.png')
        plt.savefig(save_path, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
    else:
        plt.show()

    ''' 
    #Second graph, which is the cumulative importance per variable

    #Just in case i will include other variables
    if variables_new: 
        var_feat = variables_new
    else:
        var_feat =['bm', 'i_c', 'i_v', 'i_t']

    variables_dataset = X.columns.droplevel(1) #just focus on the variablees
    df = pd.DataFrame(index=var_feat,columns=['importance'])
    df['importance'] = 0

    for i, variable in enumerate(variables_dataset):

        df.loc[str(variable), 'importance'] = df.loc[str(variable),'importance'] + feature_importance[i]
        
    df.plot.bar(
    figsize=(10, 5),
    title='Feature Importance according to Random Forest',
    legend=False,
    grid=True,
    fontsize=12,
    rot=0,
    color='royalblue'
    )

    #add the y label
    plt.ylabel('Cumulative Importance per variable', fontsize=12)

    if save_dir:
        save_path = os.path.join(save_dir, 'feature_importance_per_variable.png')

        plt.savefig(save_path, bbox_inches='tight')
        print(f"Plot saved to {save_path}")

    else:
        plt.show()
    '''
