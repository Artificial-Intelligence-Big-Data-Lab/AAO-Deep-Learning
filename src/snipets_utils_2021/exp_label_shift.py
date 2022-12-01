import pandas as pd 
from classes.Market import Market

market = Market(dataset='sp500_cet')
market = market.get_binary_labels(freq='1d', columns=['delta_current_day', 'delta_next_day', 'close'], thr=-0.5).reset_index()
market = Market.get_df_by_data_range(df=market.copy(), start_date='2011-01-04', end_date='2020-07-29')

market = market[['date_time', 'label_current_day']]



market['date_time'] = market['date_time'].shift(-1)
market = market.dropna()


market.to_csv('C:/Users/Utente/Desktop/esecuzione n°2 originale 96 sample/shift_label_test/shift_-2.csv', header=True, index=False)
