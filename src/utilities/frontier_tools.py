import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#import os
import glob
import scipy.stats

def get_pareto_points(input_df):
    '''
    get the non-dominated set of points (pareto set) in risk-return space
    for a given input_df of backtest results and return the samme
    df with an additional boolean-value column specifying pareto set membership
    '''
    df = input_df.copy()
    df['is_pareto'] = False
    updated_df = df.copy()
    for idx, row in df.iterrows():
        dominated_by = df[(df['excess_return'] > row['excess_return']) & \
                          (df['excess_risk'] < row['excess_risk'])]
        if dominated_by.shape[0] == 0: # if not dominated
            updated_df.loc[idx,'is_pareto'] = True
    return updated_df[updated_df['is_pareto']==True].sort_values(by='excess_risk', ascending=True)

def get_all_backtests_df(backtests_dir):
    '''
    get all DataFrames containing backtest results, concatenate them and
    return as single DataFrame.
    '''
    dfs = []
    for file_dir in glob.glob(f'{backtests_dir}*etc.csv'): # get all .csv files that were created during backtesting
        dfs.append(pd.read_csv(f'{file_dir}'))
    return pd.concat(dfs, ignore_index=True)

def get_all_seed_frontiers(all_results_df, data_dir, market_name, model_name, plot_all_pts=False, alpha=0.2, figsize=(8,5), xlim=None, ylim=None):
    '''
    take a backtest results df containing multiple seed values 
    and plot pareto frontier for each seed set. optionally plot
    all points with given alpha value. also save all_results_df
    with extra column indicating pareto set membership as .csv file
    '''
    all_frontier_dfs = []
    fig, ax = plt.subplots(figsize=figsize)
    for seed in all_results_df['seed'].unique():
        #print(seed)
        results_df = all_results_df[all_results_df['seed']==seed] # get subset of all_results_df for given seed        
        frontier_df = get_pareto_points(results_df) # get frontier df for given seed
        all_frontier_dfs.append(frontier_df)
        if plot_all_pts:
            ax.scatter(results_df['excess_risk'].values, results_df['excess_return'].values, alpha=alpha) # all points in alpha
        ax.plot(frontier_df['excess_risk'], frontier_df['excess_return'], '-o') # frontier
        
    ax.set_xlabel('Excess risk (%)')
    ax.set_ylabel('Excess return (%)')
    plot_title = f'Seeded Backtests: {market_name} {model_name}'
    ax.set_title(plot_title)
    
    if xlim != None:
        ax.set_xlim(xlim)

    if ylim != None:
        ax.set_ylim(ylim)

    save_name = f'{data_dir}{market_name}_{model_name}_seed_frontiers'
    plt.savefig(f'{save_name}.png', dpi=150, facecolor=None, edgecolor=None, bbox_inches='tight')
    print('\tplot saved.', end='')
    all_frontier_df = pd.concat(all_frontier_dfs, ignore_index=True)
    all_frontier_df.to_csv(f'{data_dir}{market_name}_{model_name}_seed_frontiers.csv', index=False)
    print('\tdf saved.')

    return all_frontier_df

def get_critical_t(n, alpha=0.05):
    return scipy.stats.t.ppf(q=1-alpha/2,df=n-1)

def get_mean_margin_of_err(data_list, conf_lvl=0.95):
    data = np.array(data_list)
    n = len(data)
    mean = data.mean()
    stdev = data.std(ddof=1)
    alpha = 1.0 - conf_lvl
    t_star = get_critical_t(n, alpha)
    margin_of_err = t_star * (stdev / np.sqrt(n))
    return mean, margin_of_err

def get_bin_intervals(data, bin_width=3.0):
    '''get bin intervals: select bins by constant bin widths (bin_width)'''
    data_range = max(data) - min(data)
    nb_bins = int(np.ceil(data_range / bin_width))
    bin_intervals = []
    bin_mids = []
    for i in range(nb_bins):
        bin_intervals.append((i*bin_width, (i+1)*bin_width))
        bin_mids.append(( i*bin_width + (i+1)*bin_width ) / 2.0)
    return bin_intervals, bin_mids

def get_bin_means_margin_of_errors(x_data, y_data, bin_width=None, min_in_bin=25, conf_lvl=0.95):
    '''x_data and y_data must be numpy arrays and already sorted by x_data!'''
    
    # for each bin of x-data, get the mean and margin of error of the corresponding y-data
    y_means = []
    y_margin_of_errs = []
    
    # get x-data by bin intervals if bin_width specified
    if bin_width != None:
        bin_intervals, x_mids = get_bin_intervals(x_data, bin_width=bin_width)
    
        for (lower, upper) in bin_intervals:
            try:
                binned_y = y_data[(x_data>lower) & (x_data<=upper)]
                mean, margin_of_err = get_mean_margin_of_err(binned_y, conf_lvl=0.95)
                y_means.append(mean)
                y_margin_of_errs.append(margin_of_err)
            except:
                print(f'Error! probably an empty bin where interval is: ({lower},{upper})')
                
    # otherwise, bins of data by minimum amount of data inside bin
    elif min_in_bin != None:
        nb_bins = int(np.ceil(len(x_data) / min_in_bin))
        #print(f'nb_bins={nb_bins}')
        x_mids = []
        for i in range(nb_bins):
            binned_y = y_data[i*min_in_bin : (i+1)*min_in_bin]
            binned_x = x_data[i*min_in_bin : (i+1)*min_in_bin] 
            
            #print(f'\nbinned_x:\n{binned_x}')
            #print(f'\nbinned_y:\n{binned_y}')
            #break
            
            mean, margin_of_err = get_mean_margin_of_err(binned_y, conf_lvl=0.95)
            x_mids.append(( binned_x[0] + binned_x[-1] ) / 2.0)
            y_means.append(mean)
            y_margin_of_errs.append(margin_of_err)
            
    else:
        raise Exception(f'Error: either bin_width or min_in_bin must have a value that is not None. Current values given: bin_width={bin_width} or min_in_bin={min_in_bin}.')
            
    return np.array(x_mids), np.array(y_means), np.array(y_margin_of_errs)

def get_mean_frontier_with_CI(market_name, model_name, bin_width=None, min_in_bin=30, conf_lvl=0.95, plot=True, save_plot=False, 
                              figsize=(8,6), xlim=(0,30), ylim=(0,30), data_loc=''):
    '''
    for a given market name and model, return risk values with their corresponding 
    mean returns (mean frontier) and margin of error at the given confidence level.
    Optionally plot/save plot. 
    '''
    
    # load df with frontiers of all seeds
    data_dir = f'{data_loc}{market_name}/seeded/{model_name}/backtests/'
    seeded_frontiers_df = pd.read_csv(f'{data_dir}{market_name}_{model_name}_seed_frontiers.csv')
    
    # sort by excess risk and ignore points where excess risk > upper xlim
    data = seeded_frontiers_df.sort_values(by=['excess_risk']).reset_index(drop=True)
    data = data[data['excess_risk']<=xlim[1]]
    
    # get mean line and confidence interval
    x_data = data['excess_risk'].values
    y_data = data['excess_return'].values
    x_mids, y_means, y_margin_of_errs = get_bin_means_margin_of_errors(x_data, y_data, bin_width=bin_width, min_in_bin=min_in_bin, conf_lvl=conf_lvl)
    
    # plot and save if desired
    if plot:
        fig, ax = plt.subplots(figsize=figsize, facecolor=None, edgecolor=None)
        ax.plot(x_mids, y_means, f'C0-', alpha=1.0, label=f'{model_name.replace("_"," ")}')
        ax.fill_between(x_mids, (y_means-y_margin_of_errs), (y_means+y_margin_of_errs), color=f'C0', alpha=0.25)

        ax.set_xlabel('Excess risk (%)')
        ax.set_ylabel('Excess return (%)')
        ax.set_title(f'Mean pareto frontier with {int(conf_lvl*100)}% CI')
        ax.legend();

        ax.set_xlim(left=xlim[0], right=xlim[1])
        ax.set_ylim(bottom=ylim[0], top=ylim[1])
    if plot and save_plot:
        plt.savefig(f'{data_dir}{market_name}_{model_name}_mean_frontier.png', dpi=150, facecolor=None, edgecolor=None, bbox_inches='tight')
        
    # return risk values with their corresponding mean returns (mean frontier) and margin of error
    return x_mids, y_means, y_margin_of_errs