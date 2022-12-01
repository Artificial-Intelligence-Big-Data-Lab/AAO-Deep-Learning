import pandas as pd 
import numpy as np 
from classes.Utils import df_date_merger
from classes.Market import Market
from classes.Measures import Measures


'''
df_1 = pd.read_csv('C:/Users/Utente/Desktop/73-76-nuovo-criterio/073 - gold and gold silver - pesi bilanciati_long_short_walk_0.csv')
df_2 = pd.read_csv('C:/Users/Utente/Desktop/73-76-nuovo-criterio/076 - ripetizione 75 - due walk_long_short_walk_0.csv')
df_3 = pd.read_csv('C:/Users/Utente/Desktop/73-76-nuovo-criterio/076 - ripetizione 75 - due walk_long_short_walk_1.csv')


#df_1 = pd.read_csv('C:/Users/Utente/Desktop/prova/lh/073 - gold and gold silver - pesi bilanciati_long_walk_0.csv')
#df_2 = pd.read_csv('C:/Users/Utente/Desktop/prova/lh/076 - ripetizione 75 - due walk_long_walk_0.csv')
#df_3 = pd.read_csv('C:/Users/Utente/Desktop/prova/lh/076 - ripetizione 75 - due walk_long_walk_1.csv')



#df_1 = pd.read_csv('C:/Users/Utente/Desktop/prova/lh/073 - gold and gold silver - pesi bilanciati_short_walk_0.csv')
#df_2 = pd.read_csv('C:/Users/Utente/Desktop/prova/lh/076 - ripetizione 75 - due walk_short_walk_0.csv')
#df_3 = pd.read_csv('C:/Users/Utente/Desktop/prova/lh/076 - ripetizione 75 - due walk_short_walk_1.csv')


df_1 = df_1[['date_time', 'decision']]
df_2 = df_2[['date_time', 'decision']]
df_3 = df_3[['date_time', 'decision']]

df_final = pd.concat([df_2, df_3, df_1])
#df_final = pd.concat([df_1])
#df_final = pd.concat([df_2])
#df_final = pd.concat([df_3])


df_final['decision'] = df_final['decision'].shift(1)
df_final = df_final.dropna()
df_final['decision'] = df_final['decision'].astype(int)
df_final.to_csv('C:/Users/Utente/Desktop/73-76-nuovo-criterio/final_long_short_nuovo_criterio_mc.csv', header=True, index=False)
'''

'''
df = pd.read_csv('D:/PhD-Market-Nets/experiments/073 - gold and gold silver - pesi bilanciati/predictions/predictions_during_training/validation/walk_0/net_0.csv')

epoch = df['epoch_112'].tolist()

tot = len(epoch)
long = epoch.count(2)
short = epoch.count(0)
hold = epoch.count(1)

print("Totali:", tot, "- Long:", long, "- Short", short, "- Hold", hold)
print("Totali:", tot, "- Long:", long / tot, "- Short", short / tot, "- Hold", hold / tot)
'''




'''
df_long = pd.read_csv('D:/PhD-Market-Nets/experiments/073 - gold and gold silver - pesi bilanciati/selection/on_finish/walk_0/long_UNselected.csv', delimiter=';', decimal=',')
df_short = pd.read_csv('D:/PhD-Market-Nets/experiments/073 - gold and gold silver - pesi bilanciati/selection/on_finish/walk_0/short_UNselected.csv', delimiter=';', decimal=',')

df_ls = pd.DataFrame()

df_ls['net'] = df_long['net']
df_ls['epoch'] = df_long['epoch']
df_ls['valid_por'] = (df_long['valid_por'] + df_short['valid_por']) / 2
df_ls['valid_romad'] = df_long['valid_romad'] + df_short['valid_romad']
df_ls['valid_return'] = df_long['valid_return'] + df_short['valid_return']
df_ls['valid_mdd'] = df_long['valid_mdd'] + df_short['valid_mdd']
df_ls['valid_cove'] = df_long['valid_cove'] + df_short['valid_cove']
df_ls['test_por'] = (df_long['test_por'] + df_short['test_por']) / 2
df_ls['test_romad'] = df_long['test_romad'] + df_short['test_romad']
df_ls['test_return'] = df_long['test_return'] + df_short['test_return']
df_ls['test_mdd'] = df_long['test_mdd'] + df_short['test_mdd']
df_ls['test_cove'] = df_long['test_cove'] + df_short['test_cove']

df_ls.to_csv('D:/PhD-Market-Nets/experiments/073 - gold and gold silver - pesi bilanciati/selection/on_finish/walk_0/test_Long_short_UNselected.csv', header=True, index=False)
'''


df = pd.read_csv('C:/Users/Utente/Desktop/73-76-nuovo-criterio/final_long_short_nuovo_criterio_mc.csv')

df = df_date_merger(df=df.copy(), columns=['delta_current_day', 'high', 'low', 'open', 'close'], thr_hold=0.3, dataset='gold_cet')


equity_line, global_return, mdd, romad, i, j = Measures.get_equity_return_mdd_romad(df=df.copy(), multiplier=100, type='long_short', penalty=25, stop_loss=1000, delta_to_use='delta_current_day')
daily_returns =  Measures.get_daily_returns(df=df.copy(), multiplier=100, type='long_short', penalty=25, stop_loss=1000, delta_to_use='delta_current_day')

returns = np.add.accumulate(daily_returns)[-1]

risk_free = 0

sr = (daily_returns.mean() - risk_free) / daily_returns.std()
ann_sr = ((daily_returns.mean() - risk_free) / daily_returns.std()) * np.sqrt(240) # 240 / 252

print("Return:", "{:.2f}".format(returns))
print("MDD C-C:", "{:.2f}".format(mdd))
print("Sharpe Ration:", "{:.2f}".format(sr))
print("Annualized Sharpe Ration:", "{:.2f}".format(ann_sr))
