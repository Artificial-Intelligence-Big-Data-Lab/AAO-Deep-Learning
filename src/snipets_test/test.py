from classes.Market import Market
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import numpy as np

'''
sp500 = Market(dataset='sp500')

df = sp500.group(freq='1d', nan=False)

df['label_test'] = 0

df.loc[df['delta_percentage_previous_day'] > 0.3, ['label_test']] = 1
df.loc[df['delta_percentage_previous_day'] < -0.3, ['label_test']] = -1

hold = len(np.where(df['label_test'] == 0)[0])
positive = len(np.where(df['label_test'] == 1)[0])
negative = len(np.where(df['label_test'] == -1)[0])

total = hold + positive + negative

print("Totale: ", total)

print("Positive: ", positive)
print("Negative: ", negative)
print("Hold: ", hold)
#df.to_csv('sp500_percentage.csv', header=True)

dax = pd.read_csv('../datasets/dax_cet.csv')
dax = dax.reset_index()
dax = dax.rename(columns={"index": "id"})
dax.to_csv('../datasets/dax_cet.csv', index=False, header=True)

'''

'''
' Calcola l'ensemable sulle colonne (reti) con una % di agreement
''
def perc_ensemble(df, thr=0.5):
    n_short = (df.eq(0).sum(1) / df.shape[1]).gt(thr)
    n_long = (df.eq(2).sum(1) / df.shape[1]).gt(thr)
    n_hold = (df.eq(1).sum(1) / df.shape[1]).gt(thr)

    #n_long.astype(int).mul(2).add(n_short)
    m = pd.DataFrame(np.select([n_short, n_long], [0, 2], 1), index=df.index, columns=['ensemble'])
    #m = pd.DataFrame(np.select([n_short, n_hold, n_long], [0, 1, 2], 1), index=df.index, columns=['ensemble'])

    return m


df = pd.DataFrame([{ 'col_1': 0, 
                    'col_2': 0, 
                    'col_3': 0, 
                    'col_4': 0, 
                    'col_5': 0, 
                    'col_6': 1, 
                    'col_7': 1, 
                    'col_8': 1, 
                    'col_9': 1, 
                    'col_10': 1,
                    'col_11': 2,
                    'col_12': 2,
                    'col_13': 2,
                    'col_14': 2,
                    'col_15': 2,
                }])
print(df)

thr = 0.3
short_perc = (df.eq(0).sum(1) / df.shape[1])[0]
hold_perc = (df.eq(1).sum(1) / df.shape[1])[0]
long_perc = (df.eq(2).sum(1) / df.shape[1])[0]

print("/nHold perc: ", hold_perc, " | Short perc: ", short_perc, " | Long perc: ", long_perc)
print("Soglia scelta: ", thr)
ens = perc_ensemble(df=df, thr=thr)

print("Risultato ensemble: ", ens.ensemble[0])
'''

'''
def get_mdd_romad(equity_line):
    cumulative = np.maximum.accumulate(equity_line) - equity_line
    
    if all(cumulative == 0): 
        return 999, 999, 999, 999, 999

    # calcolo la posizione i-esima
    i = np.argmax(cumulative) # end of the period

    # calcolo la posizione j-esima
    j = np.argmax(equity_line[:i]) # start of period

    global_return = equity_line[-1]

    # mdd
    mdd = equity_line[j] - equity_line[i] 
    # romad
    romad = global_return / mdd 
    # return totale
    #bh_global_return = sum(series) 

    return global_return, mdd, romad, i, j

def get_mdd_romad_bh(series):

    global_return = series[-1] - series[0]

    cumulative = np.maximum.accumulate(equity_line) - equity_line
    
    if all(cumulative == 0): 
        return 999, 999, 999, 999, 999

    # calcolo la posizione i-esima
    i = np.argmax(cumulative) # end of the period

    # calcolo la posizione j-esima
    j = np.argmax(equity_line[:i]) # start of period

    # mdd
    mdd = series[j] - series[i] 
    # romad
    romad = global_return / mdd 
    # return totale
    #bh_global_return = sum(series) 

    return global_return, mdd, romad, i, j

daily_returns = np.array([11,12,-10, -11, 30, -40, 10, 23])

equity_line = np.add.accumulate(daily_returns)

print((daily_returns))
print(equity_line)

global_return, mdd, romad, i, j = get_mdd_romad(equity_line)
print("Return: ", global_return, " | Mdd: ", mdd, " | Romad: ", romad, " | Lower: ", equity_line[i], " | Upper: ", equity_line[j])

global_return, mdd, romad, i, j = get_mdd_romad_bh(daily_returns)
print("Return: ", global_return, " | Mdd: ", mdd, " | Romad: ", romad, " | Lower: ", equity_line[i], " | Upper: ", equity_line[j])
'''





#sp500 = Market(dataset='sp500')
#oned = sp500.group(freq='1d')
#range_df = Market.get_df_by_data_range(oned, '2011-06-27', '2011-07-27')
#print(range_df)
#input()

#####################



