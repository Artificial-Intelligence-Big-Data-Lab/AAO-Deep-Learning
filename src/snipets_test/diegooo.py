from classes.Market import Market
from classes.Utils import df_date_merger
from classes.Measures import Measures
import pandas as pd 
import numpy as np 

'''
sp500 = Market(dataset='sp500_cet')
aaa = sp500.get_label(freq='1d', columns=['open', 'close', 'high', 'low', 'up', 'down', 'volume', 'delta_current_day', 'delta_next_day', 'delta_current_day_percentage', 'delta_next_day_percentage'])
aaa.to_csv('C:/Users/Andrea/Desktop/Nuova cartella/sp500_1d.csv', header=True, index=True)
'''

namefile = 'turtleoutput.csv'
namefile = 'turtleoutput1.csv'
namefile = 'turtleoutput2.csv'
namefile = 'reinfoutput.csv'

df = pd.read_csv('C:/Users/Utente/Downloads/' + namefile)


df = df_date_merger(df=df.copy(), thr_hold=0.3, columns=['delta_next_day', 'delta_current_day', 'close', 'open', 'high', 'low'], dataset='sp500_cet')

df = Market.get_df_by_data_range(df=df.copy(), start_date='2011-01-01', end_date='2019-12-31')
#print(df)
#input()
equity_line, global_return, mdd, romad, i, j = Measures.get_equity_return_mdd_romad(df.copy(), multiplier=50, type='long_short', stop_loss=0, penalty=0, delta_to_use='delta_next_day')
long, short, hold, general = Measures.get_precision_count_coverage(df=df.copy(), multiplier=50, delta_to_use='delta_next_day', stop_loss=0, penalty=0)

print(namefile)
print("Return:", '{:.3f}'.format(global_return), "- MDD:", '{:.3f}'.format(mdd), "- Romad:", '{:.3f}'.format(romad))