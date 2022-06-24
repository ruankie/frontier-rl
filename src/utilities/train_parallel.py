import json
import papermill as pm
from multiprocessing import Process, cpu_count
import numpy as np
import time
from src.config import market_tickers
import os

MIN_GAMMA_RISKS_PER_WORKER = 1
MIN_SEEDS_PER_WORKER = 1

def split_list(original_list, n):
    ''' splits a list into n sub-lists.
    '''
    original_list = np.array(original_list)
    seeds_sub_lists = [arr.tolist() for arr in np.array_split(original_list, n)]
    return tuple(seeds_sub_lists)


def train_rl(seeds, market_name, tickers, model_base_name, from_date, until_date, 
            gamma_trades, gamma_risks, gamma_holds, nb_episodes, save_every):

    assert market_name in ['TEST_5', 'SP_11', 'DOW_30','NIK_25','LA_40', 'SP_500'], 'must choose a valid market name (or update valid list in assertion).'
    assert model_base_name in ['RL_CNN','RL_str_fcast','RL_all_inp'], 'must choose a valid model base name: "RL_CNN","RL_str_fcast", or "RL_all_inp".'

    # set path to notebooks
    os.chdir(os.path.abspath('../../notebooks/'))

    start = time.time()
    print(f'\tstarting {model_base_name} on {market_name} [{from_date} - {until_date}] with seeds {seeds}.')
    pm.execute_notebook(
                    input_path='train_template.ipynb',
                    output_path=f'slave_notebooks/{model_base_name}_{market_name}_risks_{gamma_risks[0]}_seeds_{seeds[0]}.ipynb',
                    parameters={
                                'SEED_LIST':seeds,
                                'TICKERS':tickers,
                                'MARKET_NAME':market_name,
                                'MODEL_BASE_NAME':model_base_name,
                                'FROM':from_date,
                                'UNTIL':until_date,
                                'NB_EPISODES':nb_episodes,
                                'SAVE_EVERY':save_every,
                                'GAMMA_TRADES':gamma_trades,
                                'GAMMA_RISKS':gamma_risks,
                                'GAMMA_HOLDS':gamma_holds
                               },
                    progress_bar=True,
                   )
    print(f'\tdone with {model_base_name} on {market_name} (risks:{gamma_risks[0]}.. seeds:{seeds[0]}..) in ', round(time.time() - start,2), 'seconds.')


if __name__ == '__main__':

    # change dir to where this file is located 
    # so the context is the same no matter where it's run from
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    # start timer
    #main_start = time.time()

    if not os.path.exists('../../notebooks/slave_notebooks'):
        os.makedirs('../../notebooks/slave_notebooks')

    # read in config_file
    with open('../config/train_config.json') as json_file:  
        config = json.load(json_file)    
    all_seeds = config['RANDOM_SEEDS']
    all_base_names = config['MODEL_BASE_NAMES']
    nb_episodes = config['NB_EPISODES']
    save_every = config['SAVE_EVERY']
    all_markets = config['MARKETS']
    gamma_trades = config['GAMMA_TRADES']
    gamma_risks = config['GAMMA_RISKS']
    gamma_holds = config['GAMMA_HOLDS']
    nb_workers = config['NB_WORKERS']
    if nb_workers == -1:
        nb_workers = cpu_count()    
    print('number of workers chosen:',nb_workers)

    # split up seeds and gamma_risks into sets
    nb_seed_sets = nb_workers//MIN_SEEDS_PER_WORKER
    nb_gamma_risks_sets = nb_workers//MIN_GAMMA_RISKS_PER_WORKER
    seed_sets = split_list(all_seeds, nb_seed_sets)
    gamma_risks_sets = split_list(gamma_risks, nb_gamma_risks_sets)

    # min amount of workers required for workload and config
    min_workers = len(all_markets) * len(all_base_names) * len(seed_sets) * len(gamma_risks_sets)
    assert min_workers<=nb_workers, f"Number of workers allowed: {nb_workers}. Minimum number of workers required: {min_workers}. Allow more workers!)"

    processes = []
    # start training in separate processes
    for market_name, dates in all_markets.items(): # for all markets
        tickers = getattr(market_tickers, market_name+'_TICKER')
        for mod_idx, mod_name in enumerate(all_base_names): # for all models
            for seed_idx in range(len(seed_sets)): # for all seed sets
                for gamma_risk_idx in range(len(gamma_risks_sets)):
                    proc = Process(target=train_rl, args=(
                        seed_sets[seed_idx], 
                        market_name, 
                        tickers, 
                        mod_name, 
                        dates['FROM'], 
                        dates['UNTIL'], 
                        gamma_trades, 
                        gamma_risks_sets[gamma_risk_idx], 
                        gamma_holds,
                        nb_episodes, 
                        save_every))
                    processes.append(proc)
                    proc.start()
            
    for p in processes:
        p.join()
        
    #print()
    #print('all workers done in', round(time.time() - main_start,2), 'seconds.')