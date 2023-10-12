import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
import os
import pandas as pd
import seaborn as sns
import sklearn

from scipy.stats import gaussian_kde
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error


#This function is used to plot the results of the grid search

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

#This function is used to plot the feature importance

def plot_feature_importance(rf_trained, X, variables_new = None, save_dir=None, title = None):

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
    if save_dir and title:
        save_path = os.path.join(save_dir, title)
        plt.savefig(save_path, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
    else:
        plt.show()


#This function is used to plot the results of the model evaluation
#It is a scatter plot of the observed vs the predicted values, with the RMSE, R2 and MSE

def plot_results(y_data, y_modeled, save_dir=None, title = None, axis_lim = None):

    '''Parameters:
    y_data: the observed data
    y_modeled: the modeled data
    save_dir: the directory where the plot will be saved
    axis_lim: the axis limit for the plot'''

    y_data = y_data.values

    xy = np.vstack([y_data,y_modeled])
    z = gaussian_kde(xy)(xy)

    idx = z.argsort()
    ann_plt, y_plt, z = y_data[idx], y_modeled[idx], z[idx]

    fig, ax = plt.subplots()
    plt.title("Model Evaluation ")

    plt.xlabel('y predicted')
    plt.ylabel('y observed')
    sc = plt.scatter(ann_plt, y_plt, c=z, s=5)
    cbar = plt.colorbar(sc)

    textstr = '\n'.join((
        r'$RMSE=%.2f$' % (mean_squared_error(y_data, y_modeled, squared=False), ),
        r'$R^2=%.2f$' % (r2_score(y_data, y_modeled), ),
        r'$MSE=%.2f$' % (mean_squared_error(y_data, y_modeled), )))
    props = dict(boxstyle='round', facecolor='white', alpha=0.5)
    ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=14,
            verticalalignment='top', bbox=props)

    #Now i want to plot the line y = x
    x = np.linspace(*ax.get_xlim())
    ax.plot(x, x, color='black', linestyle='--')

    if axis_lim is not None:
        plt.xlim(-axis_lim, axis_lim)
        plt.ylim(-axis_lim, axis_lim)

    #Add a grid
    plt.grid(True, linestyle='--', linewidth=0.5)

    if save_dir and title:
        save_path = os.path.join(save_dir, title)
        plt.savefig(save_path, bbox_inches='tight')
        print(f"Plot saved to {save_path}")

    plt.show()
