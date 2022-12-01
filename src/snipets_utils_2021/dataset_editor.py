import pandas as pd 
import numpy as np
from classes.Market import Market
import random

# Dataset originale a 5 minuti
df_5min = pd.read_csv('C:/Users/Utente/Documents/GitHub/PhD-Market-Nets/datasets/sp500_cet.csv')
'''
df_5min['open_hack']    = df_5min['open'].tolist()
df_5min['close_hack']   = df_5min['close'].tolist()

df_5min['open_hack_2']  = df_5min['open'].tolist()
df_5min['close_hack_2'] = df_5min['close'].tolist()
'''
df_5min['date']         = pd.to_datetime(df_5min['date'])

# Dataset a risoluzione giornaliera contenente label_next_day
df_1day = Market(dataset='sp500_cet')
df_1day = df_1day.get_binary_labels(freq='1d', thr=-0.5).reset_index()


weak_class = 0 # da elaborare
strong_class = 1 # da skippare

# 5 ore 
pattern_leght = 12 * 5 

first_pattern = list(range(1, pattern_leght + 1))
second_pattern = random.sample(range(pattern_leght), pattern_leght)

strong_pattern = [100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 
                200, 200, 200, 200, 200, 200, 200, 200, 200, 200, 200, 200, 
                300, 300, 300, 300, 300, 300, 300, 300, 300, 300, 300, 300, 
                200, 200, 200, 200, 200, 200, 200, 200, 200, 200, 200, 200, 
                100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100]

medium_pattern = [10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 
                20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 
                30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 
                20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 
                10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10]


weak_pattern = [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 
                4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 
                5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 
                4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 
                2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2
                ]

# 2 ore
#pattern_leght = 24 
#ultra_weak_pattern = [ 
#                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
#                2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2
#                ]

percentage_multiplier = 0
# +x, -2x,+3x, -2x, +x
percentage_pattern = [
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2,
    3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 
    -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2,
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
]

# ciclo sul dataset a risoluzione daily
for one_day_idx, one_day_row in df_1day[::-1].iterrows():
    pattern_count = 0

    if one_day_row['label_next_day'] == strong_class:
        print("[SKIP] Giorno: ", one_day_row['date_time'].strftime('%Y-%m-%d'), "- Label:", one_day_row['label_next_day'])
        continue
            
    print("[WORK] Giorno: ", one_day_row['date_time'].strftime('%Y-%m-%d'), "- Label:", one_day_row['label_next_day'])

    df_subset = df_5min.loc[ ( df_5min["date"] == one_day_row['date_time'].strftime('%Y-%m-%d')) ]

    # ciclo sul dataset a risoluzione 5 minuti
    #for five_min_idx, five_min_row in df_5min[::-1].iterrows():
    for five_min_idx, five_min_row in df_subset[::-1].iterrows():

        # Quando le date sono uguali, inserisco i pattern del df
        if five_min_row['date'].strftime('%Y-%m-%d') == one_day_row['date_time'].strftime('%Y-%m-%d'):
            if pattern_count < pattern_leght:
                # AGGIUNGO UN DELTA FISSO
                #df_5min.ix[five_min_idx, 'close'] = five_min_row['close'] + third_pattern[pattern_count]
                #df_5min.ix[five_min_idx, 'close'] = five_min_row['close'] + ultra_weak_pattern[pattern_count]
                
                # AGGIUNGO UN DELTA PERCENTUALE
                df_5min.ix[five_min_idx, 'close'] = five_min_row['close'] + ( (five_min_row['close'] / 100) *  (percentage_multiplier * percentage_pattern[pattern_count]))
                '''
                df_5min.ix[five_min_idx, 'open_hack'] = first_pattern[pattern_count]
                df_5min.ix[five_min_idx, 'open_hack_2'] = second_pattern[pattern_count]

                df_5min.ix[five_min_idx, 'close_hack'] = first_pattern[pattern_count] * 2
                df_5min.ix[five_min_idx, 'close_hack_2'] = second_pattern[pattern_count] * 2
                '''
                pattern_count += 1
            
            # per questo giorno ho aggiunto i due pattern, skippo al prossimo giorno
            if pattern_count == pattern_leght: 
                pattern_count = 0 
                break
        
        # Se sfogliando df a 5 minuti vado dietro la data che mi serve, interrompo il ciclo [non dovrebbe essere necessario post ottimizzazione]
        if five_min_row['date'] < pd.to_datetime(one_day_row['date_time'].strftime('%Y-%m-%d')):
            pattern_count = 0 
            break

    #print(df_5min.loc[ ( df_5min["date"] == one_day_row['date_time'].strftime('%Y-%m-%d')) ])
    #input() 


#print("Second pattern:", second_pattern)           
df_5min.to_csv('C:/Users/Utente/Documents/GitHub/PhD-Market-Nets/datasets/sp500_cet_WITH_PATTERN_A_intesita_' + str(percentage_multiplier) + '.csv', header=True, index=False)