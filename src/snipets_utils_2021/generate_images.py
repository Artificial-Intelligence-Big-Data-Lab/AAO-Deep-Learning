from classes.Market import Market
from classes.Gaf import Gaf
from classes.ImagesMerger import ImagesMerger
#from classes.Downloader import Downloader

#from classes.VggHandler import VggHandler

import time
import pandas as pd

#symbols = ['aapl', 'abt', 'amzn', 'jnj', 'jpm', 'ko', 'mmm', 'msft']
#symbols = ['a', 'aal', 'aap', 'abbv', 'abc', 'acn']
#symbols = ['aapl', 'abt', 'adbe', 'adi', 'adm',  'adp', 'ads', 'adsk', 'aee', 'aep', 'aes', 'afl', 'agn', 'aig','aiv']

#symbols = ['msft', 'amzn', 'aapl', 'googl', 'goog', 'brk_b', 'fb', 'v', 'jnj', 'jpm', 'xom', 'wmt', 'pg', 'bac' 'ma', 'dis', 'pfe', 'vz', 'csco', 'unh', 't', 'cvx', 'ko', 'hd', 'mrk', 'wfc', 'intc', 'ba', 'cmcsa']
#symbols = ['googl', 'goog', 'brk_b', 'fb', 'v', 'jnj', 'jpm', 'xom', 'wmt', 'pg', 'bac', 'ma', 'dis', 'pfe', 'vz', 'csco', 'unh', 't', 'cvx', 'ko', 'hd', 'mrk', 'wfc', 'intc', 'ba', 'cmcsa']
#ymbols = ['bac', 'ma', 'dis', 'pfe', 'vz', 'csco', 'unh', 't', 'cvx', 'ko', 'hd', 'mrk', 'wfc', 'intc', 'ba', 'cmcsa']

#symbols = ['sp500_cet', 'gold_cet', 'crude_oil_cet', 'dax_cet']

#d = Downloader()
#d.run(dataset='dax_cet')

#symbols = ['ibm', 'mrk']
#symbols = ['gold_cet']
#symbols = ['dax_cet']

#symbols = ['sp500_cet', 'dax_cet', 'skew_cet', 'vix_cet']
#symbols = ['skew_cet']
#symbols = ['vix_cet']

symbols = ['sp500_cet']
for symbol in symbols:
    #print("Downloading...")
    #d = Downloader()
    #d.run(dataset=symbol)

    
    print('Start working with ' + symbol)
    market = Market(dataset=symbol)
    five_min_df = market.get()
    label_df = market.get_binary_labels(freq='1d', columns=['delta_current_day', 'delta_next_day', 'close'], thr=-0.5).reset_index()
  
    print('Grouping Datasets...')
    # Raggruppo il dataset per 1h, 1d, 1w, 1m
    #ne_h = market.group(freq='1h', nan=False)
    #four_h = market.group(freq='4h', nan=False)
    #eight_h = market.group(freq='8h', nan=False)
    #one_d = market.group(freq='1d', nan=False)
    
    #one_d = market.get().dropna()

    # Genero solo le gadf per i 4 dataset
    gaf = Gaf()
    
    print('Generating GADF for 1h...')
    #gaf.run(df=one_h, dataset_name=symbol, subfolder="1hours", type='gadf', size=20, columns=['delta_current_day_percentage'])
    #gaf.run_delta_change(df=one_h.copy(), label_df=label_df.copy(), perc=2, dataset_name=symbol, subfolder="1hours", type='gadf', size=20, columns=['delta_current_day_percentage'])
    #gaf.run_delta_change_5_min(df=five_min_df.copy(), label_df=label_df.copy(), perc=0, dataset_name=symbol, subfolder="5min_227", type='gadf', size=227, columns=['delta_current_day_percentage'])
    #gaf.run_delta_change_5_min(df=five_min_df.copy(), label_df=label_df.copy(), perc=2, dataset_name=symbol, subfolder="5min_227", type='gadf', size=227, columns=['delta_current_day_percentage'])
    #gaf.run_delta_change_5_min(df=five_min_df.copy(), label_df=label_df.copy(), perc=10, dataset_name=symbol, subfolder="5min_227", type='gadf', size=227, columns=['delta_current_day_percentage'])
    #gaf.run_delta_change_5_min(df=five_min_df.copy(), label_df=label_df.copy(), perc=50, dataset_name=symbol, subfolder="5min_227", type='gadf', size=227, columns=['delta_current_day_percentage'])
    #gaf.run_delta_change_5_min(df=five_min_df.copy(), label_df=label_df.copy(), perc=100, dataset_name=symbol, subfolder="5min_227", type='gadf', size=227, columns=['delta_current_day_percentage'])
    #gaf.run_delta_change_5_min(df=five_min_df.copy(), label_df=label_df.copy(), perc=200, dataset_name=symbol, subfolder="5min_227", type='gadf', size=227, columns=['delta_current_day_percentage'])
    gaf.run_delta_change_5_min(df=five_min_df.copy(), label_df=label_df.copy(), perc=800, dataset_name=symbol, subfolder="5min_227", type='gadf', size=227, columns=['delta_current_day_percentage'])
    #gaf.run_delta_change_5_min(df=five_min_df.copy(), label_df=label_df.copy(), perc=800, dataset_name=symbol, subfolder="5min", type='gadf', size=96, columns=['delta_current_day_percentage'])
    #gaf.run_delta_change_5_min(df=five_min_df.copy(), label_df=label_df.copy(), perc=4000, dataset_name=symbol, subfolder="5min", type='gadf', size=96, columns=['delta_current_day_percentage'])
    #print('Generating GADF for 4h...')
    #gaf.run(df=four_h, dataset_name=symbol, subfolder="4hours", type='gadf', size=20, columns=['delta_current_day_percentage'])
    #print('Generating GADF for 8h...')
    #gaf.run(df=eight_h, dataset_name=symbol, subfolder="8hours", type='gadf', size=20, columns=['delta_current_day_percentage'])
    #print('Generating GADF for 1d...')
    #gaf.run(df=one_d, dataset_name=symbol, subfolder="1day", type='gadf', size=20, columns=['delta_current_day_percentage'])


    # GENERAZIONE IMG GOLD VS SILVER DA CSV
    #print('Generating GADF for GOLD/SILVER...')
    #df = pd.read_csv('C:/Users/Utente/Desktop/dataset_raff/ratio_Gold_vs_Silver_start2008.csv')
    #df = df.dropna()
    #df['date_time'] = pd.to_datetime(df['date_time'])
    #gaf.run(df=df, dataset_name=symbol, subfolder="1day", type='gadf', size=20, columns=['diff_gold_vs_silver'])
    
    
    
    

    '''
    print('Merging images...')
    imgmerger = ImagesMerger()

    imgmerger.run(
                input_folders=symbol,
                resolutions=['1hours', '4hours', '8hours', '1day'],
                signals=['delta_current_day_percentage'],
                positions=[(0, 0), (20, 0), (0, 20), (20, 20)],
                type='gadf',
                img_size=[40, 40],
                output_path="merge_" + symbol)
    ''' 
    '''
    imgmerger = ImagesMerger()
    imgmerger.run_multivariate(input_folders=symbol,
                resolutions=['1hour'],
                signals=['open', 'close', 'high', 'low'],
                positions=[ [(0, 0)], [(0, 20)], [(20, 0)], [(20, 20)]],
                type='gadf',
                img_size=[40, 40],
                output_path="merge_" + symbol + "_multivariate_ochl")       


    imgmerger = ImagesMerger()
    imgmerger.run_multivariate_multisignal(input_folders=['sp500_cet', 'vix_cet', 'sp500_cet', 'vix_cet'],
                resolutions=['1hours', '1hours', '1day', '1day'],
                signals=['delta_current_day_percentage', 'delta_current_day_percentage', 'delta_current_day_percentage', 'delta_current_day_percentage'],
                positions=[ (0, 0), (20, 0), (0, 20), (20, 20)],
                type='gadf',
                img_size=[40, 40],
                output_path="merge_sp500_vix_hours_days")  



    one_h['OHLC'] = one_h[['high', 'low', 'close', 'open']].mean(axis = 1)
    one_h['HLC'] = one_h[['high', 'low', 'close']].mean(axis = 1)  

    four_h['OHLC'] = four_h[['high', 'low', 'close', 'open']].mean(axis = 1)
    four_h['HLC'] = four_h[['high', 'low', 'close']].mean(axis = 1)  

    eight_h['OHLC'] = eight_h[['high', 'low', 'close', 'open']].mean(axis = 1)
    eight_h['HLC'] = eight_h[['high', 'low', 'close']].mean(axis = 1)  


    one_d['OHLC'] = one_d[['high', 'low', 'close', 'open']].mean(axis = 1)  
    one_d['HLC'] = one_d[['high', 'low', 'close']].mean(axis = 1)  
    '''