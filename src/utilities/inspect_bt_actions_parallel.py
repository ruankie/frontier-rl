import papermill as pm
from multiprocessing import Process
import time
import os

NB_PLOTS = 15

def inspect_actions(market_name, model_base_name, nb_plots):
    assert market_name in ['TEST_5','DOW_30','NIK_25','LA_40'], 'must choose a valid market name (or update valid list in assertion).'
    assert model_base_name in ['RL_CNN','RL_str_fcast','RL_all_inp'], 'must choose a valid model base name (RL_CNN or RL_str_fcast or RL_all_inp).'

    start = time.time()
    print(f'\tstarting {model_base_name} on {market_name} ({nb_plots} plots)...')
    pm.execute_notebook(
                    input_path='../../notebooks/inspect_backtest_actions_template.ipynb',
                    output_path=f'../../notebooks/slave_notebooks/actions/{model_base_name}_{market_name}.ipynb',
                    parameters={
                                'MARKET_NAME':market_name,
                                'MODEL_BASE_NAME':model_base_name,
                                'N':nb_plots,
                               },
                    progress_bar=True,
                   )
    print(f'\tdone with {model_base_name} on {market_name} in ', round(time.time() - start,2), 'seconds.')


if __name__ == '__main__':

    # start timer
    #main_start = time.time()

    if not os.path.exists('../../notebooks/slave_notebooks/actions'):
        os.makedirs('../../notebooks/slave_notebooks/actions')

    all_markets = ['DOW_30', 'LA_40', 'NIK_25']
    all_base_names = ['RL_CNN', 'RL_str_fcast', 'RL_all_inp']
    nb_workers = len(all_markets) * len(all_base_names)
    processes = []

    # start training in separate process for (markets * models) preocesses
    for market_name in all_markets: # for all markets
        for mod_name in all_base_names: # for all models
            proc = Process(target=inspect_actions, args=(market_name, mod_name, NB_PLOTS))
            processes.append(proc)
            proc.start()
            
    for p in processes:
        p.join()
        
    #print()
    #print('all workers done in', round(time.time() - main_start,2), 'seconds.')