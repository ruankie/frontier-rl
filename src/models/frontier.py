###########################################
##   Copyright (c) 2022 Ruan Pretorius   ##
###########################################

import numpy as np
import pandas as pd
#import matplotlib.pyplot as plt

import tensorflow as tf
#from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Concatenate#, Input, Dropout
from tensorflow.keras.optimizers import Adam

# uncomment these two lines to run on GPU
#physical_devices = tf.config.list_physical_devices('GPU') 
#tf.config.experimental.set_memory_growth(physical_devices[0], True)

#from datetime import datetime
import datetime
#import itertools
#import argparse
import os
#import pickle
#import glob


#################
### constants ###
#################
# top 12 US stocks screened for highest 50-day average volume traded on 04-05-2018 and filtered as in Boyd (alphabetical order):
# ['AAPL', 'AMD', 'BAC', 'CMCSA', 'CSCO', 'F', 'GE', 'INTC', 'MSFT', 'MU', 'PFE', 'T'] - remove 'PFE' for top 11
TICKERS = ['AAPL', 'AMD', 'BAC', 'CMCSA', 'CSCO', 'F', 'GE', 'INTC', 'MSFT', 'MU', 'T']
FILE_PERIOD = '1d' # weekly='5d', daily='1d'
#DF_PERIOD = 'D' # weekly='W', daily='D'
#PERIODS_PER_YEAR = 252 # weekly=52, daily=252
DAYS_IN_EPISODE = 30 # 365 for one-year long episodes (conditions checked at end of episode)
RANDOM_SEED = 7
FROM = '2011-08-17' # start of training set
UNTIL = '2018-05-04' # end of training set
EPISODE_DRAW_DISTRIBUTION = 'uniform' # or geometric. select starting point of eposide according to this distribution when generated
HALF_SPREAD = 0.0005/2.0 # 'a' in transaction cost function
NONLIN_COEFF = 1.0 # 'b' transaction cost function
POWER = 1.5 # power for change in poertfolio vector used in transaction cost
BORROW_COST = 0.0001 # the borrowing fee for short trades (page 58 of Boyd et al. (2017))
GAMMA_RISK, GAMMA_TRADE, GAMMA_HOLD = 18, 6.5, 0.1 # relative importance of risk, trading cost, and holding cost
INIT_PORTFOLIO = 100000000.0 # initial portfolio value
model_name = f'REINFORCE_Soft_{UNTIL}' # give model a name to distinguish saved files
#NB_EPISODES = 300 #2000
SAVE_EVERY = 100 # save model weights every time this amount of episodes pass
MODE = 'train' # train or test mode


########################
### helper functions ###
########################

def time_locator(obj, t, as_numpy=False):
    """Retrieve data from a time indexed DF, or a Series or scalar.
    from cvxportfolio/utils/datamanagement.py"""
    if isinstance(obj, pd.DataFrame):
        res = obj.iloc[obj.axes[0].get_loc(t, method='pad')]
        return res.values if as_numpy else res
    elif isinstance(obj, pd.Series):
        return obj.values if as_numpy else obj
    elif np.isscalar(obj):
        return obj
    else:
        raise TypeError(
            'Expected Pandas DataFrame, Series, or scalar. Got:', obj)
        
        
def get_int_index_from_timestamp(df, t):
    '''given a datetime-indexed Dataframe df and a timestamp t,
    return the index (row number) at t. This index can be used
    to locate the value (or others near it) with .iloc[i].
    Useful to get previous (or next) n values as follows:
    df.iloc[i-n:i]
    '''
    if isinstance(t, str):
        return df.reset_index()[df.index==t].index[0]
    elif isinstance(t, datetime.date):
        return df.reset_index()[df.index.date==t].index[0]
    else:
        raise TypeError('timestamp must be in string or datetime.date format')
        
        
def maybe_make_dir(directory):
    '''create directory if it does not exist already
    '''
    if not os.path.exists(directory):
        os.makedirs(directory)
        
        
def play_one_episode(agent, env, mode='train'):
    '''play through one episode
        agent: the agent object that interacts with the environment
        env: the environment object
        mode: select training or testing mode #TODO currently only checks if mode == 'train' (there's no check of it's anything else). If mode is set to anything else, episode will play as normal, just won't update agent weights

    resets the env by drawing a random episode and clearing all tracking variables
        - draw random episode
        - reset to start of episode
        - set done flag to False
        - for each time-step in episode:
            - agent selects action
            - perform action and get next_state, reward, transaction cost, realised returns, info from environment
            - store transition in agent memory
        - after getting trajectory through episode, calculate G
        - update actor weights/parameters with gradient ascent
    '''
    with tf.GradientTape() as tape:

        # getin initial environment state and reset agent memory
        state = env.reset()
        agent.reset_memory()
        done = False

        # get episode trajectory
        while not done:
            action = agent.choose_action(state)
            next_state, reward, transaction_cost, holding_cost, realised_rets, done, info = env.step(action)
            agent.store_transition(state, action, reward, transaction_cost, holding_cost, realised_rets)
            state = next_state

        # notify when done creating episode trajectory
        if env.verbose:
            print('done creating episode trajectory.')
            print('\tnow agent weights will be updated...')

        # calculate expected discounted future rewards for playing the episode following current policy
        G = [tf.convert_to_tensor(0.0, dtype=tf.float32)]
        for i in range(1, len(agent.reward_memory)): # for each time-step recorded in episode ignoring the first because g[-1] = 0
            G.append( agent.reward_memory[-i] + tf.scalar_mul( agent.gamma, G[-1] ) )

        # calculate loss
        G = tf.stack(G, axis=0)
        loss = -tf.reduce_mean(G)

    if mode == 'train':
        gradient = tape.gradient(loss, agent.policy.trainable_variables)
        agent.policy.optimizer.apply_gradients(zip(gradient,agent.policy.trainable_variables))

    return loss


def backtest(agent, env, weights_file_dir=None, verbose=False):
    '''returns realised returns of the agent in the environment for backtest period.
        - agent: agent that will be used for backtest
        - env: environment that agent will interact with during backtest. remember to setup env in a 'backtest' mode
        - weights_file_dir: file path that contains saved model weights for policy network, keep None for random agent
        - verbose: print what happens or not
    Note: the agent specified by the saved weights must have used the same investor preferences, 
            look-back window (tau), and assets as the backtest!
    '''
    # load model weights
    if verbose:
        print(f'loading agent weights from {weights_file_dir}...')

    if weights_file_dir:
        agent.load(weights_file_dir)

    if verbose:
        print('simulating backtest trajectory...')

    # reset environment and agent memory
    state = env.reset()
    agent.reset_memory()
    done = False

    # get backtest trajectory
    while not done:
        action = agent.choose_action(state)
        next_state, reward, transaction_cost, holding_cost, realised_rets, done, info = env.step(action)
        agent.store_transition(state, action, reward, transaction_cost, holding_cost, realised_rets)
        state = next_state

    if verbose:
        print('backtest done.')

    return np.array(agent.relised_ret_memory), np.array(env.prev_actions) #np.array(agent.action_memory)
    


###################
### environment ###
###################
# TODO maybe split environment, agent, policies, and play episode functions into different scripts

class MultiStockEnv:
    """
    multi-stock trading environment with access to all historical data required for states
    and transaction costs.
    State: state space will consist of historical log-rets window, current portfolio weight matrix,
            historical rolling avg volume, and historical rolling avg volatility.
            later on this might include fundamentals and technical indicators
    Action: actions will be portfolio vectors of shape (nb_assets,)
    """
    def __init__(self, tickers=TICKERS, from_date=FROM, until=UNTIL, #nb_episodes=100, 
                 cash_key='USDOLLAR', gamma_risk=GAMMA_RISK, gamma_trade=GAMMA_TRADE, gamma_hold=GAMMA_HOLD,
                 half_spread=HALF_SPREAD, nonlin_coef=NONLIN_COEFF, power=POWER, borrow_costs=BORROW_COST,
                 datadir='../../data/processed_data/', 
                 state_lookback_window=20, distribution=EPISODE_DRAW_DISTRIBUTION,
                 days_duration=DAYS_IN_EPISODE, mode='train', random_seed=RANDOM_SEED,
                 init_portfolio=INIT_PORTFOLIO, period_in_file_name=FILE_PERIOD, 
                 nb_forecasts=None, forecast_type='strong', use_CNN_state=False, verbose=False):
        '''initialise environment
            - tickers: ticker symbols of assets in portfolio
            - from_date: use when backtesting to specify start date of backtest (e.g. '2018-01-01'), otherwise leave as None
            - until: end date of backtests, get data for assets only until this point in time (e.g. '2021-01-01')
            - cash_key: key used in data for rik-free asset
            - gamma_risk: scaling factor for relative importance of risk (risk aversion parameter)
            - gamma_trade: scaling factor for relative importance of trade cost in reward function
            - gamma_hold: scaling factor for relative importance of holding cost in reward function
            - half_spread: 'a' constant in transaction cost function
            - nonlin_coef: 'b' constant in transaction cost function
            - power: power of change in portfolio vector used in transaction cost
            - borrow_costs: the borrowing fee for short trades (unit-less)
            - data_dir: directory where processed historical data can be found
            - state_lookback_window: number of time-steps to include in log-rets for any one state
            - distribution: distribution from which episode start dates are drawn
            - days_duration: specifies episode duration in days
            - mode: to specify if environment will be used for training (to generate episodes) or 
                    for backtesting (generate one long episode of specified length). mode in ['train', 'backtest']
            - random_seed: seed used for random number generation (used for reproducibility)            
            - init_portfolio: initial portfolio value (in USD)
            - period_in_file_name: sampling frequency of data (e.g. weekly='5d', daily='1d') 
            - nb_forecasts: number of time-steps in forecasts used as state observation (None=don't use forecasts; 1=one step ahead; 2=two steps ahead)
            - use_CNN_state: include log-rets window for CNN in state (True/False)
            - forecast_type: to use 'strong' or 'weak' forecasts as state observations
            - verbose: print information or not
        '''
        if verbose:
            print('creating new instance of MultiStockEnv class...')
            print('\tstocks in portfolio:', tickers)

        self.tickers = tickers
        self.from_date = from_date
        self.until = until
        #self.nb_episodes = nb_episodes
        self.cash_key = cash_key
        self.gamma_risk = gamma_risk
        self.gamma_trade = gamma_trade
        self.gamma_hold = gamma_hold
        self.half_spread = half_spread
        self.nonlin_coef = nonlin_coef
        self.power = power
        self.state_lookback_window = state_lookback_window
        self.distribution = distribution
        self.days_duration = days_duration
        self.mode = mode
        self.init_portfolio = tf.convert_to_tensor(init_portfolio, dtype=tf.float32)
        self.verbose = verbose
        if self.cash_key != None:
            self.nb_assets = len(self.tickers) + 1 # total number of assets including cash
        else:
            self.nb_assets = len(self.tickers)        

        # load actual and estimated data (sigmas, returns, log-returns, volumes)
        if self.verbose:
            print('\tnumber of assets in portfolio:', self.nb_assets)
            print('\tloading actual and estimated data (sigmas, returns, volumes)...')

        self.sigmas = pd.read_csv(datadir+'sigmas.csv.gz',index_col=0,parse_dates=[0])
        self.returns = pd.read_csv(datadir+'returns.csv.gz',index_col=0,parse_dates=[0])
        self.log_returns = pd.read_csv(datadir+'log_returns.csv.gz',index_col=0,parse_dates=[0])
        self.volumes = pd.read_csv(datadir+'volumes.csv.gz',index_col=0,parse_dates=[0])
        #self.volume_estimate = pd.read_csv(datadir+'volume_estimate.csv.gz',index_col=0,parse_dates=[0]).dropna()
        #self.sigma_estimate = pd.read_csv(datadir+'sigma_estimate.csv.gz',index_col=0,parse_dates=[0]).dropna()
        self.volume_estimate_scaled = pd.read_csv(datadir+'volume_estimate_scaled.csv.gz',index_col=0,parse_dates=[0]).dropna()
        self.sigma_estimate_scaled = pd.read_csv(datadir+'sigma_estimate_scaled.csv.gz',index_col=0,parse_dates=[0]).dropna()
        self.Sigma = self.returns.rolling(window=10, min_periods=10, closed='neither').cov().dropna()
        self.borrow_costs = borrow_costs

        self.nb_forecasts = nb_forecasts
        self.forecast_type = forecast_type
        self.use_CNN_state = use_CNN_state

        # load forecasts if applicable
        if self.nb_forecasts != None:
            if self.forecast_type == 'strong':
                self.ret_forecasts = pd.read_csv(datadir+'return_estimate.csv.gz',index_col=0,parse_dates=[0])
            elif self.forecast_type == 'weak':
                self.ret_forecasts = pd.read_csv(datadir+'return_estimate_double_noise.csv.gz',index_col=0,parse_dates=[0])
            else:
                raise ValueError(f'MultiStockEnv was given the following, incompatible forecast type: {self.forecast_type}. Forecast type should be either "weak" or "strong".')

        if self.verbose:
            print('\tinstance creation complete.')

        # set seed of random number generator (only sets seed at instance creation, not before every self.reset() call)
        np.random.seed(random_seed)
        tf.random.set_seed(random_seed)

        # reset to beginning of episode and return initial state
        self.reset()


    def _sample_start_date(self):
        '''sample a starting date to use as the start of an episode. return the datetime stamp of start date.
        the range will be sampled from between from_date and until (both inclusive)
        '''
        if self.distribution == 'uniform':
            draw_weights = None # this will result in a uniform distribution
        elif self.distribution == 'geometric':
            k = np.arange(1,self.returns.shape[0],1)
            #p = 0.001 # determines steepness/shape of probability decay
            p = 5e-5 # as in Jiang et al. (2017) [j]
            draw_weights = ( (1-p)**(k-1.0) ) * (p)
            draw_weights = np.flip(draw_weights)
            # take last few weights to match length of returns subset from which will be drawn
            draw_weights = draw_weights[-self.returns[self.from_date:self.until].shape[0]:]            
        return self.returns[self.from_date:self.until].sample(n=1, weights=draw_weights).index


    def _generate_random_episode_date_range(self):
        '''
        returns the start_date and end_date of date range as DatetimeIndex that starts at start_date 
        and has a duration of days_duration (in days).
        use this to index price and returns DataFrames for a randomly generated episode as follows:
        DataFrame.loc[start_date : end_date]
        '''
        if self.verbose:
            print('\tgenerating random episode...')
        start_date = self._sample_start_date()
        start_date_idx = get_int_index_from_timestamp(self.returns, start_date.date[0])
        #end_date = start_date + pd.Timedelta(days=self.days_duration)
        end_date = self.returns.iloc[[start_date_idx + self.days_duration - 1]].index
        return (start_date.date[0], end_date.date[0])

    
    def reset(self):
        '''draws new episode window from historical data
        episode is drawn according to a geometric/uniform distribution
        returns starting state of new episode
        '''
        if self.verbose:
            print('\tresetting environment...')
            
        # reset portfolio value to initial value
        self.portfolio_value = self.init_portfolio

        # draw episode then get data for that episode
        if self.mode == 'train':
            start_date, end_date = self._generate_random_episode_date_range()
        elif self.mode == 'backtest':
            start_date = datetime.datetime.strptime(self.from_date, '%Y-%m-%d').date()
            end_date = datetime.datetime.strptime(self.until, '%Y-%m-%d').date()
        else:
            print('env.mode must be in ["train", "backtest"]') #TODO replace with raising an error instead of just printing
        
        # set  episode/backtest start and end date as environment attributes
        self.episode_start_date = start_date
        self.episode_end_date = end_date
        
        # get list of dates/timestamps of days in episode
        self.episode_times = self.returns.index[
            (self.returns.index.date >= self.episode_start_date) &
            (self.returns.index.date <= self.episode_end_date)
            ].date

        # get number of steps in episode
        self.nb_episode_steps = len(self.episode_times)

        # reset current time-step counter and time-index
        self.curr_time_step = 0
        #self.curr_time = self.episode_start_date
        self.curr_time = self.episode_times[self.curr_time_step]

        # reset list of previous actions
        self.prev_actions = []
        
        # initial portfolio vector (equally weighted and fully invested as in Boyd et al. (2017))
        # w = np.ones(self.nb_assets)
        # w[-1] = 0.0 # none in cash
        # w /= w.sum()        
        
        # initial portfolio vector - fully invested in cash
        if self.cash_key == None:
            raise Exception('Cannot start fully invested in cash when cash asset is disabled!')
        w = np.zeros(self.nb_assets)
        w[-1] = 1.0 # fully invested in cash

        # convert portfolio vector to tensorflow tensor and update action history
        w = tf.convert_to_tensor(w, dtype=tf.float32)
        self.prev_actions.append(w)

        # set done flag to false to indicate that end of episode not reached
        self.done = False

        # return initial observation from start of episode
        return self._get_obs()
    
    
    def _get_obs(self):
        '''
        return current observation (observable state)
        this will be a tuple of tf tensors (log_rets_window, current_weight, volume_est, sigma_est) (depending on policy 
        net used and which states are looked for)
        if no forecasts are given, or (ret_forecast, current_weight, volume_est, sigma_est) otherwise, or combination.
            - log_rets_window: window of log-returns passed to conv net
            - ret_forecast: return forecasts of one or more time-steps into the future
            - current_weight: current portfolio weight vector
            - volume_est: rolling avg volume estimate used for transaction cost estimation
            - sigma_est: rolling avg volatility estimate used for transaction cost estimation
        '''
        if self.verbose:
            print('\tgetting current state observation...')

        # log-rets window / ret_forecast for state
        # only log-rets for CNN
        if (self.use_CNN_state) and (self.nb_forecasts == None): # only log-rets for CNN
            idx = get_int_index_from_timestamp(self.log_returns, self.curr_time)
            log_rets_window = self.log_returns.iloc[idx-self.state_lookback_window : idx]
        # both forecasts and log-rets
        elif (self.use_CNN_state) and (self.nb_forecasts != None): # both forecasts and log-rets
            idx = get_int_index_from_timestamp(self.ret_forecasts, self.curr_time)
            ret_forecast = self.ret_forecasts.iloc[idx : idx+self.nb_forecasts]
            idx = get_int_index_from_timestamp(self.log_returns, self.curr_time)
            log_rets_window = self.log_returns.iloc[idx-self.state_lookback_window : idx]
        # only forecasts
        elif (not self.use_CNN_state) and (self.nb_forecasts != None): # only forecasts
            idx = get_int_index_from_timestamp(self.ret_forecasts, self.curr_time)
            ret_forecast = self.ret_forecasts.iloc[idx : idx+self.nb_forecasts]
        # invalid combination
        else:
            raise ValueError(f'MultiStockEnv was given the following, incompatible state type: use_CNN_state={self.use_CNN_state} with nb_forecasts={self.nb_forecasts}. Select state type where at least one is given.')

        # additional states
        current_weight = self.prev_actions[-1]
        volume_est = self.volume_estimate_scaled.loc[self.curr_time]
        sigma_est = self.sigma_estimate_scaled.loc[self.curr_time]
        
        # convert to tensors with correct shape and return state
        volume_est = tf.convert_to_tensor(volume_est, dtype=tf.float32)
        sigma_est = tf.convert_to_tensor(sigma_est, dtype=tf.float32)
        # only log-rets for CNN
        if (self.use_CNN_state) and (self.nb_forecasts == None): # only log-rets for CNN
            log_rets_window = tf.convert_to_tensor(log_rets_window, dtype=tf.float32)
            return (log_rets_window, current_weight, volume_est, sigma_est)
        # both forecasts and log-rets
        elif (self.use_CNN_state) and (self.nb_forecasts != None): # both forecasts and log-rets
            ret_forecast = tf.convert_to_tensor(ret_forecast, dtype=tf.float32)
            log_rets_window = tf.convert_to_tensor(log_rets_window, dtype=tf.float32)
            return (log_rets_window, ret_forecast, current_weight, volume_est, sigma_est)
        # only forecasts
        elif (not self.use_CNN_state) and (self.nb_forecasts != None): # only forecasts
            ret_forecast = tf.convert_to_tensor(ret_forecast, dtype=tf.float32)
            return (ret_forecast, current_weight, volume_est, sigma_est)

    
    def step(self, action):
        '''take one step:
            - perform action (specified by agent)
            - increment step counter
            - get next historical price/ret
            - calculate portfolio value difference
            - calculate reward (e.g. realised risk-adj. return after transaction cost and holding cost)
            - check if end of episode reached
            - populate info dict with some diagnostic info
        return state, reward, transaction_cost, holding_cost, realised_rets, done, info

        action: portfolio vector that defines action to take for step (tf.tensor)
        '''
        if self.verbose:
            print('\ttaking step in environment...')

        # calculate change in weights, transaction cost and realised returns for step
        prev_action = self.prev_actions[-1]
        delta_w = action - prev_action # both tf.tensors
        delta_w_nc = delta_w[:-1] # without cash. change to u_nc = w_nc @ portfolio value
        
        # transaction cost # (unit-less)
        #transaction_cost = tf.scalar_mul( self.beta, tf.norm(delta_w, ord=1) )
        cost_first_term = self.half_spread * abs(delta_w_nc)

        cost_second_term = self.nonlin_coef \
                            * time_locator(self.sigmas, self.curr_time) \
                            * ( (abs(delta_w_nc) ** self.power) \
                                * ( (self.portfolio_value / time_locator(self.volumes, self.curr_time)) ** (self.power - 1) )
                              )
        cost_second_term = cost_second_term.fillna(0) # whenever volume traded is zero, second transaction cost term will be NaN, replace with zero
        transaction_cost = sum(cost_first_term + cost_second_term) # (unit-less)

        # holding cost # (unit-less)
        # TODO make holding cost for negative cash position negative as well??
        # TODO might have to change operation functions depending on if action is numpy array or tensorflow tensor (if no errors, delete this todo)
        # (see eq 2.4 on page 11 of Boyd et al. (2017))
        # get negative portfolio weights in action (replace others with zero so they have no effect)
        neg_actions = -tf.minimum(action, 0) # or for numpy: np.minimum(action, 0) 
        holding_cost = sum( time_locator(self.borrow_costs, self.curr_time) * neg_actions )
        
        # realised returns # (unit-less)
        rets = tf.tensordot(action, 
                             tf.convert_to_tensor(time_locator(self.returns, self.curr_time), dtype=tf.float32), 
                            axes=1)
        realised_rets = rets - transaction_cost - holding_cost # (unit-less)
        
        # risk function (quadratic risk: w^{T} \Sigma w) # (unit-less)
        risk = tf.tensordot(tf.transpose(action), \
                             tf.tensordot(tf.convert_to_tensor(self.Sigma.loc[self.curr_time], dtype=tf.float32), 
                                          action, axes=1), axes=1)
        
        # immediate reward
        reward = rets \
                 - (self.gamma_trade * transaction_cost) \
                 - (self.gamma_hold * holding_cost) \
                 - (self.gamma_risk * risk)
        
        #u = delta_w * self.portfolio_value # change in portfolio (USD) excluding cash        
        rets_usd = rets * self.portfolio_value
        transaction_cost_usd = transaction_cost * self.portfolio_value
        holding_cost_usd = holding_cost * self.portfolio_value

        # append latest action to list of previous actions
        self.prev_actions.append(action)

        # store the current value of the portfolio in info dict
        info = {'curr_time_step': self.curr_time_step,
                'curr_time': self.curr_time,
                'rets' : rets.numpy(),
                'transaction_cost': transaction_cost.numpy(),
                'realised_rets:': realised_rets.numpy(),
                'risk:': risk.numpy(),
                'reward:': reward.numpy(),
                'rets_usd' : rets_usd.numpy(),
                'transaction_cost_usd' : transaction_cost_usd.numpy(),
                'holding_cost_usd' : holding_cost_usd.numpy(),
                'portfolio_value' : self.portfolio_value.numpy(),
                #'episode_start_date': self.episode_start_date, ############ only for diagnostics - can remove later
                #'episode_end_date': self.episode_end_date, ############ only for diagnostics - can remove later
                }

        # increment time-step counter
        self.curr_time_step += 1
        self.curr_time = self.episode_times[self.curr_time_step]

        # update portfolio value
        self.portfolio_value += rets_usd - transaction_cost_usd - holding_cost_usd

        # check if end of episode reached
        # done if we have run out of data
        done = self.curr_time_step == self.nb_episode_steps - 1  

        # get next state so long and set done=True if state too small (ran out of episode log-rets)
        next_state = self._get_obs()

        return next_state, reward, transaction_cost, holding_cost, realised_rets, done, info



#######################
### policy networks ###
#######################

class LongShortCNNPolicy(tf.keras.Model):
    '''Same as original CNN network but allows short positions: policy network with CNN used to choose agent 
        actions - specified by architecture:
        - n_assets: number assets in portfolio
        - tau: length of sliding time-window considered in conv kernel
        - lookback_window: number of timesteps included in historical log-rets window
        - n_feature_maps: number of feature maps produced by conv layer
        - dropout_rate: dropout probability after flattened layer (implement if necessary)
    return model output

    Note: this model will take inputs in the form: [log_rets, additional_states] as long as they match the input dimensions specified
    TODO: add long-only constraint to cash position - currently all assets can be long or short
    '''
    def __init__(self, n_assets=12, tau=5, lookback_window=20):
        super(LongShortCNNPolicy, self).__init__()
        self.conv = Conv2D(n_assets, (n_assets,tau), activation='relu', input_shape=(1, n_assets, lookback_window, 1), data_format='channels_last')
        self.flatten = Flatten()
        self.concat = Concatenate()
        self.dense = Dense(3*n_assets, activation='relu')
        self.long_short = Dense(n_assets, activation=None) # linear activation

    def call(self, state):
        log_rets_window = tf.expand_dims([tf.transpose(state[0])], axis=-1) # transpose and add batch dim
        additional_inputs = tf.expand_dims( self.concat(state[1:]), 0) # concatente addtional inputs and add batch dim
        x = self.conv(log_rets_window)
        x = self.flatten(x)
        x = self.concat([x, additional_inputs])
        x = self.dense(x)
        #print(f'{"-"*10}\ndense x:\n\n{x}\n{"-"*10}\n\tshape: {x.shape}')
        x = self.long_short(x) # output weights between -1 and 1
        #print(f'{"-"*10}\nlong_short x:\n\n{x}\n{"-"*10}\n\tshape: {x.shape}')
        x = x / tf.math.reduce_sum(x) # ensure weights add to 1
        #print(f'{"-"*10}\nnormalised x:\n\n{x}\n{"-"*10}\n\tshape: {x.shape}')
        return tf.squeeze(x)


class PolicyGradientNetwork(tf.keras.Model):
    '''policy network with CNN used to choose agent actions - specified by architecture:
        - n_assets: number assets in portfolio
        - tau: length of sliding time-window considered in conv kernel
        - lookback_window: number of timesteps included in historical log-rets window
        - n_feature_maps: number of feature maps produced by conv layer
        - dropout_rate: dropout probability after flattened layer (implement if necessary)
    return model output

    Note: this model will take inputs in the form: [log_rets, additional_states] as long as they match the input dimensions specified
    '''
    def __init__(self, n_assets=12, tau=5, lookback_window=20, n_feature_maps=12):
        super(PolicyGradientNetwork, self).__init__()
        self.conv = Conv2D(n_assets, (n_assets,tau), activation='relu', input_shape=(1, n_assets, lookback_window, 1), data_format='channels_last')
        self.flatten = Flatten()
        self.concat = Concatenate()
        self.dense = Dense(3*n_assets, activation='relu')
        self.soft_out = Dense(n_assets, activation='softmax')

    def call(self, state):#, training=False):
        log_rets_window = tf.expand_dims([tf.transpose(state[0])], axis=-1) # transpose and add batch dim
        additional_inputs = tf.expand_dims( self.concat(state[1:]), 0) # concatente addtional inputs and add batch dim
        x = self.conv(log_rets_window)
        x = self.flatten(x)
        x = self.concat([x, additional_inputs])
        x = self.dense(x)
        #if training:
        # x = self.dropout(x, training=training)
        return tf.squeeze(self.soft_out(x))


class PolicyNetworkWithForecast(tf.keras.Model):
    ''' Similar to the PolicyGradientNetwork, but instead of using CNN block
    on log-returns window for implicitly forecasting returns and covariances,
    it takes explicit returns forecasts. This is for more equal comparison to 
    Boyd's SPO/MPO models
    policy network with explicit returns forecasts used to choose agent actions - specified by architecture:
        - n_assets: number assets in portfolio
        - n_forecast_steps: number of steps ahead given as forecast (e.g. 1 or 2)
    return model output

    Note: this model will take inputs in the form: [ret_forecasts, additional_states] as long as they match the input dimensions specified
    '''
    def __init__(self, n_assets=12):#, n_forecast_steps=2):
        super(PolicyNetworkWithForecast, self).__init__()
        #self.conv = Conv2D(n_assets, (n_assets,tau), activation='relu', input_shape=(1, n_assets, lookback_window, 1), data_format='channels_last')
        #self.flatten = Flatten()
        self.concat = Concatenate()
        self.dense = Dense(3*n_assets, activation='relu')
        self.soft_out = Dense(n_assets, activation='softmax')

    def call(self, state):#, training=False):
        forecasts = tf.reshape((state[0]), shape=(-1)) # flatten/combine forecast tensor
        additional_states = self.concat(state[1:]) # combine the rest of the state tensors
        state = tf.expand_dims( self.concat([forecasts, additional_states]), 0) # concatente state vesctor inputs and add batch dim
        x = self.dense(state)
        #if training:
        # x = self.dropout(x, training=training)
        return tf.squeeze(self.soft_out(x))


class PolicyNetworkWithAllInputs(tf.keras.Model):
    '''Combination of PolicyGradientNetwork and PolicyNetworkWithForecast - returns CNN block
    of log-returns window for implicitly forecasting returns and covariances and explicit returns forecasts
    as state input. 
    Note: state input will the following tuple: (log_rets_window, ret_forecast, current_weight, volume_est, sigma_est)
    '''
    def __init__(self, n_assets=12, tau=5, lookback_window=20, n_feature_maps=12):
        super(PolicyNetworkWithAllInputs, self).__init__()
        self.conv = Conv2D(n_assets, (n_assets,tau), activation='relu', input_shape=(1, n_assets, lookback_window, 1), data_format='channels_last')
        self.flatten = Flatten()
        self.concat = Concatenate()
        self.dense = Dense(3*n_assets, activation='relu')
        self.soft_out = Dense(n_assets, activation='softmax')

    def call(self, state):#, training=False):
        log_rets_window = tf.expand_dims([tf.transpose(state[0])], axis=-1) # transpose and add batch dim
        forecasts = tf.reshape((state[1]), shape=(-1)) # flatten/combine forecast tensor
        #additional_inputs = tf.expand_dims( self.concat(state[2:]), 0) # concatente addtional inputs and add batch dim
        additional_states = self.concat(state[2:]) # combine the rest of the state tensors
        x = self.conv(log_rets_window)
        x = self.flatten(x)        
        y = tf.expand_dims( self.concat([forecasts, additional_states]), 0) # concatente and add batch dim        
        x = self.concat([x, y])
        x = self.dense(x)
        #if training:
        # x = self.dropout(x, training=training)
        return tf.squeeze(self.soft_out(x))



#############
### agent ###
#############

# TODO: better specification of policy network - remove all if-else statements and pass policy object at init
class Agent(object):
    def __init__(self, alpha=0.001, gamma=0.99, n_assets=12, tau=5, lookback_window=20, 
                 n_feature_maps=12, use_forecasts=False, use_CNN_state=False, allow_long_short_trades=True):#, dropout_rate=0.2):
        '''
        initialise agent with policy network for choosing actions
            - alpha: learning rate
            - gamma: discount factor for future rewards
            - n_assets: number of assets for specifying output
            - tau: length of sliding time-window considered in conv kernel that operates on historical log-rets window
            - lookback_window: number of timesteps included in historical log-rets window
            - n_feature_maps: number of feature maps produced by conv layer
            - use_forecasts: whether to use forecasts as input to policy network or not
            - use_CNN_state: include log-rets window for CNN in state (True/False)
        '''
        self.gamma = gamma
        self.alpha = alpha
        #self.n_assets = n_assets
        self.state_memory = [] # for recording trajectory through episode
        self.action_memory = [] # for recording trajectory through episode
        self.reward_memory = [] # for recording trajectory through episode
        self.trans_cost_memory = [] # for recording trajectory through episode
        self.holding_cost_memory = [] # for recording trajectory through episode
        self.relised_ret_memory = [] # for keeping track of realised returns (after transaction cost)
        self.use_forecasts = use_forecasts # whether to use forecasts as input to policy network or not
        self.use_CNN_state = use_CNN_state
        self.allow_long_short_trades = allow_long_short_trades

        # allow long and short trades (same as original CNN policy network with different activation function)
        if self.allow_long_short_trades:
            self.policy = LongShortCNNPolicy(n_assets=n_assets, 
                                            tau=tau, 
                                            lookback_window=lookback_window)            
        # only log-rets for CNN
        elif (self.use_CNN_state) and (not self.use_forecasts):
            self.policy = PolicyGradientNetwork(n_assets=n_assets, 
                                                tau=tau, 
                                                lookback_window=lookback_window, 
                                                n_feature_maps=n_feature_maps)
        # both forecasts and log-rets
        elif (self.use_CNN_state) and (self.use_forecasts):
            self.policy = PolicyNetworkWithAllInputs(n_assets=n_assets, 
                                                    tau=tau, 
                                                    lookback_window=lookback_window, 
                                                    n_feature_maps=n_feature_maps)
        # only forecasts
        elif (not self.use_CNN_state) and (self.use_forecasts):
            self.policy = PolicyNetworkWithForecast(n_assets=n_assets)
        # invalid combination
        else:
            raise ValueError(f'Agent was given the following, incompatible state type: use_CNN_state={self.use_CNN_state} with use_forecasts={self.use_forecasts}. Select state type where at least one is given.')
            
        self.policy.compile(optimizer=Adam(learning_rate=self.alpha))


    def choose_action(self, state):
        '''
        returns an action (portfolio vector) as a tf tensor based in some input state made up of 
        a historical log-rets window and additional states such as the current 
        portfolio vector (previous action)
        '''
        return self.policy(state)


    def store_transition(self, state, action, reward, transaction_cost, holding_cost, realised_rets):
        '''
        append state, action, immediate reward, transaction_cost, and realised_rets to memory of current episode
        this represents the trajectory from which G will be calculated later
        '''
        self.state_memory.append(state) # list of tensors
        self.action_memory.append(action) # list of tensors
        self.reward_memory.append(reward) # list of tensors
        self.trans_cost_memory.append(transaction_cost)
        self.holding_cost_memory.append(holding_cost)
        self.relised_ret_memory.append(realised_rets)


    def reset_memory(self):
        '''
        clears agent's memory of: 
            - reward_memory, 
            - action_memory,
            - relised_ret_memory, 
            - action_memory, 
            - state_memory, 
            - trans_cost_memory
            - holding_cost_memory
        this should be done at the end of each episode to prevent endless appending to memory lists
        '''
        self.state_memory = []
        self.action_memory = []
        self.reward_memory = []
        self.trans_cost_memory = []
        self.holding_cost_memory = []
        self.relised_ret_memory = []

    def get_memory(self):
        '''
        returns agent's memory of: 
            - reward_memory, 
            - action_memory,
            - relised_ret_memory, 
            - action_memory, 
            - state_memory, 
            - trans_cost_memory
            - holding_cost_memory
        this memory is returned as a tuple and should be done before mini-batch training 
        in online learning
        '''
        return (self.state_memory, self.action_memory, self.reward_memory, self.trans_cost_memory, self.holding_cost_memory, self.relised_ret_memory)


    def load(self, file_name):
        '''
        load model weights from previously saved file
        '''
        self.policy.load_weights(file_name)
        #self.policy.load_model(file_name)


    def save(self, file_name):
        '''
        save complete model to file
        '''
        self.policy.save_weights(file_name)
        #self.policy.save(file_name) #, save_format='tf'