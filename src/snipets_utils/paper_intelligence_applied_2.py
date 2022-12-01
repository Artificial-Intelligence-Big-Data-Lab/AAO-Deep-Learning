from classes.Market import Market
from classes.Utils import df_date_merger
from classes.Measures import Measures
import pandas as pd 
import numpy as np 

#from classes.Downloader import Downloader
#d = Downloader()
#d.run(dataset='ibm')

'''
' ensemble
' anselmo=True utilizza la sua codifica
'''
def ensemble_magg_classic(df, thr, anselmo=False):
    long_value = 2
    hold_value = 1
    short_value = 0

    if anselmo == True: # anselmo values
        long_value = 1
        hold_value = 0
        short_value = 2

    short = (df.eq(short_value).sum(1) / df.shape[1]).gt(thr)
    hold = (df.eq(hold_value).sum(1) / df.shape[1]).gt(thr)
    long = (df.eq(long_value).sum(1) / df.shape[1]).gt(thr)
    
    long_1 = df.eq(long_value).sum(1) > df.eq(short_value).sum(1)
    long_2 = df.eq(long_value).sum(1) > df.eq(hold_value).sum(1)

    short_1 = df.eq(short_value).sum(1) > df.eq(hold_value).sum(1)
    short_2 = df.eq(short_value).sum(1) > df.eq(long_value).sum(1)

    df = pd.DataFrame(np.select([long & long_1 & long_2, short & short_1 & short_2], [2, 0], default=1), index=df.index, columns=['decision'])

    return df

'''
' Mi genera una lista di iterazioni, 
' la utilizzo per filtrare un sottogruppo
' di iterazioni di anselmo
'
'''
def gen_iterations_list(bottom, top):
    list = []

    for i in range(bottom, top): 
        list.append('iteration' + str(i))
    
    return list

DATASET = "sp500_cet"
STOPLOSS = 0
PENALTY = 0
THR_ENSEMBLE = 0.3
MULTIPLIER = 50


#JPM
results_type = ''
#results_type = 'teste-adadelta-0.5-relu'
#results_type = 'teste-adadelta-0.3-selu'
#results_type = 'teste-adadelta-0.5-sigmoid'

# MRK
#results_type = 'teste-rmsprop-0.9-linear'
#results_type = 'teste-rmsprop-0.9-relu'
#tresults_type = 'teste-rmsprop-0.9-sigmoid'

# OTHERS 
#results_type = 'teste-adadelta-0.3-relu'
#results_type = 'teste-adadelta-0.5-selu'
#results_type = 'teste-adadelta-0.5-tanh'
#results_type = 'teste-adagrad-0.9-tanh'
#results_type = 'teste-adam-0.9-selu'
#results_type = 'teste-adamax-0.5-linear'

# MSFT
#df_anselmo = pd.read_csv('C:/Users/Andrea/Desktop/CSV Anselmo Applied Intelligence/Risultati Anselmo/results-msft/' + results_type + '/walk1ensemble_test.csv')
#df_andrea = pd.read_csv('C:/Users/Andrea/Desktop/CSV Anselmo Applied Intelligence/CSV  CNN/msft/test.csv')

# MKR
#df_anselmo = pd.read_csv('C:/Users/Andrea/Desktop/CSV Anselmo Applied Intelligence/Risultati Anselmo/results-mrk/' + results_type + '/walk1ensemble_test.csv')
#df_andrea = pd.read_csv('C:/Users/Andrea/Desktop/CSV Anselmo Applied Intelligence/CSV  CNN/mrk/test.csv')


# JPM
#df_anselmo = pd.read_csv('C:/Users/Andrea/Desktop/CSV Anselmo Applied Intelligence/Risultati Anselmo/results-jpm/v3/' + results_type + '/walk1ensemble_test.csv')
#df_andrea = pd.read_csv('C:/Users/Andrea/Desktop/CSV Anselmo Applied Intelligence/CSV  CNN/jpm/test.csv')

# SP500 prima versione
df_anselmo = pd.read_csv('C:/Users/Utente/Desktop/CSV Anselmo Applied Intelligence/Risultati Anselmo/results-sp500/v1/output_sp500.csv')
df_andrea = pd.read_csv('C:/Users/Utente/Desktop/CSV Anselmo Applied Intelligence/CSV  CNN/sp500/test_sinica_1.csv')

# SP500 ultima versione
#df_anselmo = pd.read_csv('C:/Users/Utente/Desktop/CSV Anselmo Applied Intelligence/Risultati Anselmo/results-sp500/'v3/ + results_type + '/walk1ensemble_test.csv')
#df_andrea = pd.read_csv('C:/Users/Utente/Desktop/CSV Anselmo Applied Intelligence/CSV  CNN/sp500/test_sinica_3.csv')


# SOTTOINSIEME DI DATE PER LE COMPANIES MSFT, MRK, JPM ETC
#df_anselmo = Market.get_df_by_data_range(df=df_anselmo.copy(), start_date='2007-01-01', end_date='2012-12-31')
#df_andrea = Market.get_df_by_data_range(df=df_andrea.copy(), start_date='2007-01-01', end_date='2012-12-31')

# SOTTOINSIEME DI DATE PER SP500 ULTIMA VERSIONE
df_anselmo = Market.get_df_by_data_range(df=df_anselmo.copy(), start_date='2009-01-01', end_date='2015-12-31')
df_andrea = Market.get_df_by_data_range(df=df_andrea.copy(), start_date='2009-01-01', end_date='2015-12-31')

# SOTTOINSIEME DI DATE PER SP500 ULTIMA VERSIONE
#df_anselmo = Market.get_df_by_data_range(df=df_anselmo.copy(), start_date='2009-08-01', end_date='2014-05-12')
#df_andrea = Market.get_df_by_data_range(df=df_andrea.copy(), start_date='2009-08-01', end_date='2014-05-12')


date_list = df_anselmo['date_time'].tolist()


df_anselmo = df_anselmo.drop(columns=["date_time"])

# riduco il numero di iterazioni
iteration_list = gen_iterations_list(0, 50)
df_anselmo = df_anselmo[iteration_list]

df_andrea = df_andrea.drop(columns=["date_time", 'open', 'close', 'delta_next_day'])


ensemble_anselmo = ensemble_magg_classic(df=df_anselmo.copy(), thr=THR_ENSEMBLE, anselmo=True)
ensemble_andrea = ensemble_magg_classic(df=df_andrea.copy(), thr=THR_ENSEMBLE, anselmo=False)

df_final = pd.DataFrame()

df_final['date_time'] = date_list
#df_final['decision_anselmo'] = ensemble_anselmo['decision'].tolist()
#df_final['decision_andrea'] = ensemble_andrea['decision'].tolist()
df_final = df_date_merger(df=df_final.copy(), thr_hold=0.3, columns=['delta_next_day', 'close', 'open', 'high', 'low'], dataset=DATASET)


# ANSELMO RESULTS
df_final['decision'] = ensemble_anselmo['decision'].tolist()
equity_line, global_return, mdd, romad, i, j = Measures.get_equity_return_mdd_romad(df_final.copy(), multiplier=MULTIPLIER, type='long_short', stop_loss=STOPLOSS, penalty=PENALTY, delta_to_use='delta_next_day')
long, short, hold, general = Measures.get_precision_count_coverage(df=df_final.copy(), multiplier=MULTIPLIER, delta_to_use='delta_next_day', stop_loss=STOPLOSS, penalty=PENALTY)

df_final_2 = df_final[['date_time', 'decision']]
df_final_2['date_time'] = df_final_2['date_time'].shift(-1)
df_final_2 = df_final_2.drop(df_final_2.index[0])
df_final_2['decision'] = df_final_2['decision'].astype(int)
df_final_2.to_csv('C:/Users/Utente/Desktop/CSV Anselmo Applied Intelligence/final_decision.csv', header=True, index=False)

print("(Anselmo " + results_type + ") Market Points:", '{:.3f}'.format(global_return / MULTIPLIER), "- Return:", '{:.3f}'.format(global_return), "- MDD:", '{:.3f}'.format(mdd), "- Romad:", '{:.3f}'.format(romad))
#print("(Anselmo No-Penalty, StopLoss-1000) Long Precision:", '{:.3f}'.format(long['precision']), "- Short Precision:", '{:.3f}'.format(short['precision']), "- coverage:", '{:.3f}'.format(general['total_coverage'])


print("\n")

# ANDREA RESULTS
df_final['decision'] = ensemble_andrea['decision'].tolist()
equity_line, global_return, mdd, romad, i, j = Measures.get_equity_return_mdd_romad(df_final.copy(), multiplier=MULTIPLIER, type='long_short', stop_loss=0, penalty=PENALTY, delta_to_use='delta_next_day')
long, short, hold, general = Measures.get_precision_count_coverage(df=df_final.copy(), multiplier=MULTIPLIER, delta_to_use='delta_next_day', stop_loss=0, penalty=PENALTY)

#print("Risultati nostri soglia", thr)
print("(Andrea No-Penalty, StopLoss) Market Points:", '{:.3f}'.format(global_return / MULTIPLIER), "- Return:", '{:.3f}'.format(global_return)  , "- MDD:", '{:.3f}'.format(mdd), "- Romad:", '{:.3f}'.format(romad))
#print("(Andrea No-Penalty, StopLoss-1000) Long Precision:", '{:.3f}'.format(long['precision']), "- Short Precision:", '{:.3f}'.format(short['precision']), "- coverage:", '{:.3f}'.format(general['total_coverage'])

print("\n")

# BUY & HOLD
equity_line, global_return, mdd, romad, i, j  = Measures.get_return_mdd_romad_bh(close=df_final['close'].tolist(), multiplier=MULTIPLIER)
print("(BH) Market Points:", '{:.3f}'.format(global_return / MULTIPLIER), "- Return:", '{:.3f}'.format(global_return), "- MDD:", '{:.3f}'.format(mdd), "- Romad:", '{:.3f}'.format(romad))
