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
    
    
