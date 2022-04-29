import json
import papermill as pm
from multiprocessing import Process, cpu_count
import numpy as np
import time
from src.config import market_tickers
import os

def split_gammas(all_gammas, n):
    ''' splits a list of gamma-tuples into n sub-tuples.
    '''
    all_gammas = np.array(all_gammas)
    gammas_sub_lists = [tuple(map(tuple, arr)) for arr in np.array_split(all_gammas, n)]
    return tuple(gammas_sub_lists)


def reproduce_backtest(market_name, from_date, until_date, model_name, constraint_name, gamma_triples):

    assert market_name in ['TEST_5', 'SP_11', 'DOW_30','NIK_25','LA_40', 'SP_500'], 'must choose a valid market name (or update valid list in assertion).'
    assert model_name in ['SPO', 'MPO'], 'must choose a valid model base name ("SPO" or "MPO").'

    start = time.time()
    print(f'\tstarting {model_base_name} on {market_name} [{from_date} - {until_date}] with seeds {seeds}.')
    pm.execute_notebook(
                    input_path='train_template.ipynb',
                    output_path=f'slave_notebooks/{model_base_name}_{market_name}_({seeds[0]}_etc).ipynb',
                    parameters={
                                'SEED_LIST':seeds,
                                'TICKERS':tickers,
                                'MARKET_NAME':market_name,
                                'MODEL_BASE_NAME':model_base_name,
                                'FROM':from_date,
                                'UNTIL':until_date,
                                'NB_EPISODES':nb_episodes,
                                'SAVE_EVERY':save_every,
                               },
                    progress_bar=True,
                   )
    print(f'\tdone with {model_base_name} on {market_name} ({seeds[0]}...) in ', round(time.time() - start,2), 'seconds.')


if __name__ == '__main__':

    # start timer
    #main_start = time.time()

    if not os.path.exists('slave_notebooks'):
        os.makedirs('slave_notebooks')

    # read in config_file
    with open('train_config.json') as json_file:  
        config = json.load(json_file)    
    all_seeds = config['RANDOM_SEEDS']
    all_base_names = config['MODEL_BASE_NAMES']
    nb_episodes = config['NB_EPISODES']
    save_every = config['SAVE_EVERY']
    all_markets = config['MARKETS']
    nb_workers = config['NB_WORKERS']
    if nb_workers == -1:
        nb_workers = cpu_count()    
    print('number of workers chosen:',nb_workers)

    # min amount of workers
    min_workers = len(all_base_names)*len(all_markets) # models * markets = 9
    assert min_workers<=nb_workers, f"number of workers = {nb_workers}. must be greater or equal to (models*markets = {min_workers})"
    assert nb_workers%min_workers == 0, f"number of workers = {nb_workers}. must be divisable by (models*markets = {min_workers})"

    # chop up seeds for workers    
    seed_sets = split_seeds(all_seeds, nb_workers//min_workers)
    processes = []


    # start training in separate process for (markets * models * seed_sets) preocesses
    for market_name, dates in all_markets.items(): # for all markets
        tickers = getattr(market_tickers, market_name+'_TICKER')
        for mod_idx, mod_name in enumerate(all_base_names): # for all models
            for seed_idx in range(len(seed_sets)): # for all seed sets
                proc = Process(target=train_rl, args=(seed_sets[seed_idx], market_name, tickers, mod_name, dates['FROM'], dates['UNTIL'], nb_episodes, save_every))
                processes.append(proc)
                proc.start()
            
    for p in processes:
        p.join()
        
    #print()
    #print('all workers done in', round(time.time() - main_start,2), 'seconds.')