import numpy as np 
import pandas as pd 
from  classes.Utils import df_date_merger
from  classes.Market import Market
from  classes.Measures import Measures
from classes.ResultsHandler import ResultsHandler

'''
' Script usato come prova per capire perché
' i bh itnraday e classici nei plot non tornavano uguali ai report
''
'''
df = pd.read_csv('D:/PhD-Market-Nets/experiments/087 - Test univariate SP500 unico blocco 1h/predictions/final_decision_alg4/ensemble_exclusive/senza-rimozione-reti/long_only/decisions_ensemble_exclusive_0.1.csv')

df_2 = pd.DataFrame()
for i in range(0, 9):
    df_walk = pd.read_csv('D:/PhD-Market-Nets/experiments/087 - Test univariate SP500 unico blocco 1h/predictions/predictions_during_training/test/walk_' + str(i) + '/net_0.csv')
    df_2 = pd.concat([df_2, df_walk], axis=0)

bh_equity_line, bh_global_return, bh_mdd, bh_romad, bh_i, bh_j  = Measures.get_return_mdd_romad_bh(close=df['close'].tolist(), multiplier=50)

bh_2_equity_line, bh_2_global_return, bh_2_mdd, bh_2_romad, bh_2_i, bh_2_j  = Measures.get_equity_return_mdd_romad(df=df.copy(), multiplier=50, type='bh_long', penalty=25, stop_loss=1000, delta_to_use='delta_current_day')
bh_3_equity_line, bh_3_global_return, bh_3_mdd, bh_3_romad, bh_3_i, bh_3_j  = Measures.get_equity_return_mdd_romad(df=df.copy(), multiplier=50, type='bh_long', penalty=0, stop_loss=0, delta_to_use='delta_current_day')

ls_equity_line, ls_global_return, ls_mdd, ls_romad, ls_i, ls_j  = Measures.get_equity_return_mdd_romad(df=df.copy(), multiplier=50, type='long_short', penalty=0, stop_loss=0, delta_to_use='delta_current_day')

print("Final Decision: ")
print("BH Return:", bh_global_return)
print("BH Intraday Return (penalty):", bh_2_global_return)
print("BH Intraday Return (no-penalty):", bh_3_global_return)
#print("Return our approach:", ls_global_return)
print("\n-------------------------------------\n")

df_2 = df_date_merger(df=df_2, columns=['close', 'open', 'high', 'low', 'delta_current_day'], thr_hold=0.3)
df_2 = df_2.rename(columns={'epoch_1': 'decision'})
# allineo le date
df_2['date_time'] = df_2['date_time'].shift(-1)
df_2 = df_2.drop(df_2.index[0])
df_2 = df_2.drop_duplicates(subset='date_time', keep="first")

print(df_2)
input()
bh_equity_line, bh_global_return, bh_mdd, bh_romad, bh_i, bh_j  = Measures.get_return_mdd_romad_bh(close=df_2['close'].tolist(), multiplier=50)

bh_2_equity_line, bh_2_global_return, bh_2_mdd, bh_2_romad, bh_2_i, bh_2_j  = Measures.get_equity_return_mdd_romad(df=df_2.copy(), multiplier=50, type='bh_long', penalty=25, stop_loss=1000, delta_to_use='delta_current_day')
bh_3_equity_line, bh_3_global_return, bh_3_mdd, bh_3_romad, bh_3_i, bh_3_j  = Measures.get_equity_return_mdd_romad(df=df_2.copy(), multiplier=50, type='bh_long', penalty=0, stop_loss=0, delta_to_use='delta_current_day')

ls_equity_line, ls_global_return, ls_mdd, ls_romad, ls_i, ls_j  = Measures.get_equity_return_mdd_romad(df=df_2.copy(), multiplier=50, type='long_short', penalty=0, stop_loss=0, delta_to_use='delta_current_day')
print("Merging walks:")
print("BH Return:", bh_global_return)
print("BH Intraday Return:", bh_2_global_return)
print("BH Intraday Return (no-penalty):", bh_3_global_return)
#print("Return our approach:", ls_global_return)