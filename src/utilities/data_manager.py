import yfinance as yf
#from datetime import datetime
import glob
import pandas as pd
import numpy as np
#import matplotlib.pyplot as plt
import quandl
from src.config.quandl_api_key import my_api_key # use your own Quandl API key please
import os

def _maybe_make_dir(directory):
    '''create directory if it does not exist already
    '''
    if not os.path.exists(directory):
        os.makedirs(directory)

def _load_historical_price_data(ticker_list, start_date='2011', end_date='2017', data_dir='../../data/market/historical_data/'):
    '''return DataFrame with historical price data read from previously saved .csv files
    in data_dir. This includes OHLCV data for all tickers in ticker_list.'''
    all_data_files = glob.glob(data_dir + '*.csv')

    ticker_dfs = []
    file_tickers_list = []

    for filename in all_data_files:
        ticker = filename.split('data\\')[1].split('_')[0] # Linux: .split('data/')  ||  Windows: .split('data\\')        #.split('_'+period_in_file_name)[0]
        #print(f'\t***ticker: {ticker}')
        # load if ticker is in requested load list
        if ticker in ticker_list:
            file_tickers_list.append(ticker)
            df = pd.read_csv(filename, header=0)
            df['Date'] = pd.to_datetime(df['Date'])
            df.set_index('Date', inplace=True)
            ticker_dfs.append(df)

    #pd.concat([df1, df2], axis=1, keys=['ABC', 'XYZ'])
    data = pd.concat(ticker_dfs, axis=1, keys=file_tickers_list)
    return data[start_date:end_date]

def _get_returns(price_data_df): #, set_index_period=DF_PERIOD):
    '''return DataFrame of simple returns of price_data_df containing OHLCV price data
    and set period of returned DataFrame equal to set_index_period.'''
    close_prices = price_data_df.loc[:, (slice(None), 'Adj Close')]
    cols = close_prices.columns
    new_cols = [cols[i][0] for i in range(len(cols))]
    close_prices.columns = new_cols
    rets = close_prices.pct_change() # r = ( p_{t} - p_{t-1} )/p_{t-1} = ( p_{t}/p_{t-1} ) -1
    #rets.index = rets.index.to_period(set_index_period)
    return rets#.dropna()

def _extract_column(OHLCV_data_df, column='Adj Close'): #, set_index_period=DF_PERIOD):
    '''return DataFrame where only the selected column was extracted OHLCV_data_df
    and set period of returned DataFrame equal to set_index_period.'''
    selected_data = OHLCV_data_df.loc[:, (slice(None), column)]
    cols = selected_data.columns
    new_cols = [cols[i][0] for i in range(len(cols))]
    selected_data.columns = new_cols
    return selected_data


class Downloader:
    ''' downloads historical OHLCV data from Yahoo Finance of a list of tickers between a start
    and end date. Cash data can also be downloaded (3-month treasury bill data from Quandl).
    All downloaded data are saved in csv files in a 'historical_data' directory.
    '''
    def __init__(self, ticker_list, start_date, end_date, interval='1d', save_dir='../../data/market/historical_data/', verbose=True): #cash_key='USDOLLAR'
        ticker_list.sort()
        self.ticker_list = ticker_list
        self.start_date = start_date
        self.end_date = end_date
        self.interval = interval
        #self.cash_key = cash_key
        self.save_dir = save_dir
        self.verbose = verbose

    def fetch_data(self):
        if self.verbose:
            print('downloading ticker data from Yahoo Finance...')
        # download ticker data from Yahoo Finance
        data = yf.download(tickers=self.ticker_list,
            interval=self.interval,
            group_by='ticker',
            start=self.start_date,
            end=self.end_date
            )

        # save data for each ticker in separate csv file
        if self.save_dir is not None:
            _maybe_make_dir(self.save_dir)
            for tick in data.columns.levels[0]:
                data[tick].to_csv(f'{self.save_dir}{tick}_{self.interval}.csv')
                if self.verbose:
                    print(f'\t{tick} data saved.')

        if self.verbose:
            print('done fetching data.')

        # return raw data (unprocessed) if not saved
        if self.save_dir == None:
            return data#, usd_rets


class Preprocessor:
    ''' preprocess all downloaded data so that it can be used by different models.
    preprocessed data is saved in a specified directory for later use.
    some preprocessing steps are the same as used in Boyd et al. (2017) and Nystrup et al. (2020)
    '''
    def __init__(self, ticker_list, start_date, end_date, interval='1d', cash_key='USDOLLAR', 
                load_dir='../../data/market/historical_data/', save_dir='../../data/market/preprocessed_data/', verbose=True):
        ticker_list.sort()
        self.ticker_list = ticker_list
        self.start_date = start_date
        self.end_date = end_date
        self.interval = interval
        self.cash_key = cash_key
        self.load_dir = load_dir
        self.save_dir = save_dir
        self.verbose = verbose

    def process_data(self):        
        OHLCV_df = _load_historical_price_data(self.ticker_list, 
                                                start_date=self.start_date, 
                                                end_date=self.end_date, 
                                                data_dir=self.load_dir)

        open_prices = _extract_column(OHLCV_df, column='Open').iloc[1:,:] # this is just to estimate volatility as in Boyd et al. (2017)
        close_prices = _extract_column(OHLCV_df, column='Close').iloc[1:,:] # this is just to estimate volatility as in Boyd et al. (2017)
        volumes = _extract_column(OHLCV_df, column='Volume').iloc[1:,:]
        returns = _get_returns(OHLCV_df)

        # include cash returns: 3-month treasury bill data from Quandl (Risk-Free Asset)
        if self.cash_key is not None:
            if self.verbose:
                print('downloading daily cash returns from Quandl...')
            quandl.ApiConfig.api_key = my_api_key
            usd_rets = quandl.get('FRED/DTB3', start_date=self.start_date, end_date=self.end_date)/(250*100)
            returns[self.cash_key] = usd_rets

        # fill nan values of returns
        returns = returns.fillna(method='ffill').iloc[1:]

        # estimate daily volatility (sigmas) as in Boyd et al. (2017)
        sigmas = np.abs(np.log(open_prices.astype(float))-np.log(close_prices.astype(float)))

        #--------------- apply filtering as in Boyd et al. (2017) ---------------
        # filter NaNs - threshold at 2% missing values
        prices = close_prices.copy()
        bad_assets = prices.columns[prices.isnull().sum()>len(prices)*0.05] # originally was 2% (0.02)
        if len(bad_assets) and self.verbose:
            print('Assets %s have too many NaNs, removing them' % bad_assets)

        prices = prices.loc[:,~prices.columns.isin(bad_assets)]
        sigmas = sigmas.loc[:,~sigmas.columns.isin(bad_assets)]
        volumes = volumes.loc[:,~volumes.columns.isin(bad_assets)]

        nassets=prices.shape[1]

        # days on which many assets have missing values
        bad_days1=sigmas.index[sigmas.isnull().sum(1) > nassets*.9]
        bad_days2=prices.index[prices.isnull().sum(1) > nassets*.9]
        bad_days3=volumes.index[volumes.isnull().sum(1) > nassets*.9]
        bad_days=pd.Index(set(bad_days1).union(set(bad_days2)).union(set(bad_days3))).sort_values()
        if self.verbose:
            print ('Removing these days from dataset:')
            print(pd.DataFrame({'nan price':prices.isnull().sum(1)[bad_days],
                                'nan volumes':volumes.isnull().sum(1)[bad_days],
                                'nan sigmas':sigmas.isnull().sum(1)[bad_days]}))

        prices=prices.loc[~prices.index.isin(bad_days)]
        sigmas=sigmas.loc[~sigmas.index.isin(bad_days)]
        volumes=volumes.loc[~volumes.index.isin(bad_days)]

        # extra filtering
        if self.verbose:
            print(pd.DataFrame({'remaining nan price':prices.isnull().sum(),
                                'remaining nan volumes':volumes.isnull().sum(),
                                'remaining nan sigmas':sigmas.isnull().sum()}))
        prices=prices.fillna(method='ffill')
        sigmas=sigmas.fillna(method='ffill')
        volumes=volumes.fillna(method='ffill')
        if self.verbose:
            print(pd.DataFrame({'remaining nan price':prices.isnull().sum(),
                                'remaining nan volumes':volumes.isnull().sum(),
                                'remaining nan sigmas':sigmas.isnull().sum()}))

        # update returns to contain only filtered assets and risk-free asset
        filtered_cols = list(prices.columns) + [self.cash_key]
        print(filtered_cols)

        returns = returns[filtered_cols]

        # filter out assets with dubious returns
        bad_assets = returns.columns[((-.5>returns).sum()>0)|((returns > 2.).sum()>0)]
        if len(bad_assets) and self.verbose:
            print('Assets %s have dubious returns, removed' % bad_assets)
            
        prices = prices.loc[:,~prices.columns.isin(bad_assets)]
        sigmas = sigmas.loc[:,~sigmas.columns.isin(bad_assets)]
        volumes = volumes.loc[:,~volumes.columns.isin(bad_assets)]
        returns = returns.loc[:,~returns.columns.isin(bad_assets)]

        # save preprocessed data
        _maybe_make_dir(self.save_dir)
        prices.to_csv(self.save_dir+'prices.csv.gz', compression='gzip', float_format='%.3f')
        volumes.to_csv(self.save_dir+'volumes.csv.gz', compression='gzip', float_format='%d')
        returns.to_csv(self.save_dir+'returns.csv.gz', compression='gzip', float_format='%.3e')
        sigmas.to_csv(self.save_dir+'sigmas.csv.gz', compression='gzip', float_format='%.3e')
        if self.verbose:
            print('prices, volumes, returns, and sigmas saved.')
        #------------------------------------------------------------------------

        # calculate log-returns for CNN policy network inputs
        if self.verbose:
            print('calculating log-returns...')
        log_returns = np.log(returns + 1.0)
        log_returns.to_csv(self.save_dir+'log_returns.csv.gz', compression='gzip', float_format='%.3e')

        #------------ estimate returns, volume, volatility ----------------------
        #                as in Boyd et al. (2017) and
        #                   Nystrup et al. (2020)
        if self.verbose:
            print('estimating returns, volume, and volatility...')
            print('\tTypical variance of returns: %g'%returns.var().mean())

        # strong returns (perturbed actual returns) - Boyd et al. (2017)
        sigma2_n=0.02
        sigma2_r=0.0005

        np.random.seed(0)
        noise=pd.DataFrame(index=returns.index, columns=returns.columns, 
                           data=np.sqrt(sigma2_n)*np.random.randn(*returns.values.shape))
        return_estimate= (returns + noise)*sigma2_r/(sigma2_r+sigma2_n)
        return_estimate.USDOLLAR = returns.USDOLLAR

        return_estimate.to_csv(self.save_dir+'return_estimate.csv.gz', compression='gzip', float_format='%.3e')
        if self.verbose:
            print('\tstrong returns estimate saved.')

        # weak returns (perturbed actual returns with double variance) - Nystrup et al. (2020)
        sigma2_n=0.04 # double of original

        np.random.seed(0)
        noise=pd.DataFrame(index=returns.index, columns=returns.columns, 
                           data=np.sqrt(sigma2_n)*np.random.randn(*returns.values.shape))


        return_estimate= (returns + noise)*sigma2_r/(sigma2_r+sigma2_n)
        return_estimate.USDOLLAR = returns.USDOLLAR

        return_estimate.to_csv(self.save_dir+'return_estimate_double_noise.csv.gz', compression='gzip', float_format='%.3e')
        if self.verbose:
            print('\tweak returns estimate saved.')


        # volume and volatisity estimates
        volume_estimate=volumes.rolling(window=10, center=False).mean().dropna()
        volume_estimate.to_csv(self.save_dir+'volume_estimate.csv.gz', compression='gzip', float_format='%d')
        sigmas_estimate = sigmas.rolling(window=10, center=False).mean().dropna()
        # fill zero-estimates with the minimum of the other non-zero estimates
        for col in sigmas_estimate.columns:
            sigmas_estimate[col].replace(to_replace=0, value=sigmas_estimate[sigmas_estimate[col]>0][col].min(), inplace=True)
        sigmas_estimate.to_csv(self.save_dir+'sigma_estimate.csv.gz', compression='gzip', float_format='%.3e')
        if self.verbose:
            print('\tvolume and volatisity estimates saved.')
        #------------------------------------------------------------------------

        # calculate scaled volume-estimate and sigma-estimate for neural net input
        if self.verbose:
            print('calculating scaled volume-estimate and sigma-estimate...')
        volumes_means = volumes.iloc[0:30].mean() # use first 30 days to get scaling value
        sigmas_means = sigmas.iloc[0:30].mean() # use first 30 days to get scaling value
        # if mean is 0, replace by global mean
        if any(sigmas_means == 0):
            sigmas_means = sigmas.mean()
        volume_estimate_scaled = volume_estimate / volumes_means
        sigma_estimate_scaled = sigmas_estimate / sigmas_means
        volume_estimate_scaled.to_csv(self.save_dir+'volume_estimate_scaled.csv.gz', compression='gzip', float_format='%.3e')
        sigma_estimate_scaled.to_csv(self.save_dir+'sigma_estimate_scaled.csv.gz', compression='gzip', float_format='%.3e')
        if self.verbose:
            print('\tscaled volume-estimate and sigma-estimate saved.')

        if self.verbose:
            print('done preprocessing.')


class FactorRiskModel:
    ''' uses the returns calculated from preprocessed data to create a 
    factor risk model (with k factors). For the first day in every month, the asset returns 
    of preceding 730 days are used to compute covariance of returns then 
    eigenvalues/eigenvectors. This follows the same metod used by
    Boyd et al. (2017) -- code from roprisor (2019-refresh).
    Note: ensure that preprocessed returns data exists before using this class. 
    Also, ensure that the returns data includes > 730 days ~ 3 years of data 
    prior to the start date of any backtest (the backtest start date must be 
    given as start_date).
    '''
    def __init__(self, start_date, k=15, load_dir='../../data/market/preprocessed_data/', verbose=True):
        self.start_date = start_date
        self.k = k
        self.load_dir = load_dir
        self.verbose = verbose

    def get_risk_model(self):
        # load preprocessed returns data
        if self.verbose:
            print('loading preprocessed returns...')
        returns = pd.read_csv(self.load_dir+'returns.csv.gz',index_col=0,parse_dates=[0])

        first_days_month=\
            pd.date_range(start=returns.index[next(i for (i,el) in enumerate(returns.index >= self.start_date) if el)-1],
                                         end=returns.index[-1], freq='MS')

        # Use PCA to identify the top k factors generating full covariance matrix
        #k=15
        exposures, factor_sigma, idyos = {}, {}, {}

        if self.verbose:
            print(f'creating {self.k}-factor risk model...')
        # Every first day in each month in period
        for day in first_days_month:
            # Grab asset returns for preceding 730 days, compute covariance of returns then eigenvalues/eigenvectors
            used_returns = returns.loc[(returns.index < day)&
                                (returns.index >= day-pd.Timedelta("730 days"))]
            second_moment=np.cov(used_returns.values.T)
            eival, eivec=np.linalg.eigh(second_moment)
            
            # Factor covariance is diagonal in top eigenvalues
            factor_sigma[day]=pd.DataFrame(data=np.diag(eival[-self.k:]))
            # Exposures are the top k eigenvectors
            exposures[day]=pd.DataFrame(data=eivec[:,-self.k:],index=returns.columns)
            # Idiosyncratic variances
            idyos[day]=pd.Series(data=np.diag(eivec[:,:-self.k]@np.diag(eival[:-self.k])@eivec[:,:-self.k].T),index=returns.columns)

        pd.concat(factor_sigma.values(), axis=0, keys=factor_sigma.keys()).to_hdf(self.load_dir+'risk_model.h5', 'factor_sigma')
        pd.concat(exposures.values(), axis=0, keys=exposures.keys()).to_hdf(self.load_dir+'risk_model.h5', 'exposures')
        pd.DataFrame(idyos).T.to_hdf(self.load_dir+'risk_model.h5', 'idyos')

        if self.verbose:
            print(f'done creating {self.k}-factor risk model.')
