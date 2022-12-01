import os
import numpy as np
import pandas as pd
from classes.Market import Market


os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID";
os.environ["CUDA_VISIBLE_DEVICES"]="0";

# Definisco i data range per il test set di 11 walk
test_set_dates = [
            ['2009-08-02', '2010-01-31'],
            ['2010-02-01', '2010-07-30'],
            ['2010-08-01', '2011-01-31'],
            ['2011-02-01', '2011-07-31'],
            ['2011-08-01', '2012-01-31'], 
            ['2012-02-01', '2012-07-31'],
            ['2012-08-01', '2013-01-31'],
            ['2013-02-01', '2013-07-31'],
            ['2013-08-01', '2014-01-31'],
            ['2014-02-02', '2014-07-31'],
            ['2014-08-01', '2015-01-30'] 
        ]


'''
' Calcola l'ensemable sulle colonne (reti) con il 100% di agreement
'''
def full_ensemble(df):
    # Controllo quali righe hanno tutto 1
    m1 = df.eq(1).all(axis=1)

    # Controllo quali righe hanno tutto 0
    m2 = df.eq(-1).all(axis=1)

    # Prevengo sovrascritture di memoria
    local_df = df.copy()
    # Creo una nuova colonna ensemble, mettendo ad 1 se tutte le colonne sono a 1, -1 se sono tutte a 0, 0 altrimenti
    local_df['ensemble'] = np.select([m1, m2], [1, -1], 0)

    # rimuovo tutte le colonne e lascio una sola colonna ensemble che contiene solamente l'operazione da fare (1, -1, 0)
    local_df = local_df.drop(local_df.columns.difference(['ensemble']), axis=1)

    return local_df

'''
' Calcola l'ensemable sulle colonne (reti) con una % di agreement
'''
def perc_ensemble(df, thr=0.7):
    c1 = (df.eq(1).sum(1) / df.shape[1]).gt(thr)
    c2 = (df.eq(-1).sum(1) / df.shape[1]).gt(thr)
    c2.astype(int).mul(-1).add(c1)
    m = pd.DataFrame(np.select([c1, c2], [1, -1], 0), index=df.index, columns=['ensemble'])

    return m


'''
filepath = 'C:/Users/andre/Desktop/CSV RANDOM GUESSING/with 20 nets/'

if not os.path.isdir(filepath):
            os.makedirs(filepath)

sp500 = Market(dataset='sp500')

sp500_one_day = sp500.group(freq='1d')

df = pd.DataFrame()

df_all = pd.DataFrame()

for i, test_set_date in enumerate(test_set_dates):

    df = Market.get_df_by_data_range(df=sp500_one_day, start_date=test_set_date[0], end_date=test_set_date[1])
    df = df[['date_time']]
    df = df.set_index('date_time')

    size = df.shape[0]
    nums = np.ones(size)

    for j in range(0, 20): 
        
        nums = np.random.choice([-1, 1], size=size, p=[.5, .5])

        df['NET_' + str(j)] = nums

    df_all = pd.concat([df_all, df], axis=0)

    df.to_csv(filepath + 'random_guess_walk_' + str(i) + '.csv', header=True, index=True)

df_all.to_csv(filepath + 'random_guess_all_walks.csv', header=True, index=True)
'''

input_filepath = 'C:/Users/andre/Desktop/CSV RANDOM GUESSING/with 20 nets/'
output_filepath = 'C:/Users/andre/Desktop/CSV RANDOM GUESSING/post ensemble/'

thrs = [0.5, 0.6, 0.7, 0.8, 0.9]

files = [   'random_guess_all_walks.csv', 
            'random_guess_walk_0.csv',
            'random_guess_walk_1.csv',
            'random_guess_walk_2.csv',
            'random_guess_walk_3.csv',
            'random_guess_walk_4.csv',
            'random_guess_walk_5.csv',
            'random_guess_walk_6.csv',
            'random_guess_walk_7.csv',
            'random_guess_walk_8.csv',
            'random_guess_walk_9.csv',
            'random_guess_walk_10.csv',
            'random_guess_walk_11.csv'
        ]

for file in files: 
    df = pd.read_csv(input_filepath + file)
    df = df.set_index('date_time')

    df_full_ensemble = full_ensemble(df=df)
    df_full_ensemble.to_csv(output_filepath + '1.0/' + file, header=True, index=True)

    for thr in thrs:
        
        df_perc_ensemble = perc_ensemble(df=df, thr=thr)
        df_perc_ensemble.to_csv(output_filepath + str(thr) + '/' + file, header=True, index=True)
