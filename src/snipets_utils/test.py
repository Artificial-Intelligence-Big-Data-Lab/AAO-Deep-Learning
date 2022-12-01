from classes.Market import Market
from classes.ResultsHandler import ResultsHandler
import pandas as pd 
import numpy as np 
from datetime import timedelta
from classes.Utils import df_date_merger
from classes.Measures import Measures
from sklearn.metrics import accuracy_score

'''
m = Market(dataset='gold_cet')

#g = m.get_label(freq='1d')
#w = m.get_label_next_days(freq='1d', next_days=5, thr=0.8, columns=['open', 'close'])

df_0 = pd.read_csv('D:/PhD-Market-Nets/experiments/050 - Prova GOLD predizioni 5 giorni/predictions/predictions_during_training/test/walk_0/net_5.csv')
df_1 = pd.read_csv('D:/PhD-Market-Nets/experiments/050 - Prova GOLD predizioni 5 giorni/predictions/predictions_during_training/test/walk_1/net_5.csv')
df_2 = pd.read_csv('D:/PhD-Market-Nets/experiments/050 - Prova GOLD predizioni 5 giorni/predictions/predictions_during_training/test/walk_2/net_5.csv')
df_3 = pd.read_csv('D:/PhD-Market-Nets/experiments/050 - Prova GOLD predizioni 5 giorni/predictions/predictions_during_training/test/walk_3/net_5.csv')

df = pd.concat([df_0,df_1, df_2, df_3])

df = df[['date_time', 'epoch_200']]
df = df.rename(columns={'epoch_200': 'decision'})

df['exit_date'] = df['date_time'].shift(-5)
df = df.dropna()
df['unique_id'] = range(0, df.shape[0])
df.to_csv('D:/PhD-Market-Nets/experiments/050 - Prova GOLD predizioni 5 giorni/predictions/final_decisions/final_decision_prova.csv', header=True, index=False)
'''


'''
rh = ResultsHandler(experiment_name="TEST TENSORFLOW2")
net_json = rh.get_result_for_walk_net(start_date='2009-10-1', end_date='2011-01-031', penalty=0, stop_loss=0, index_walk=0, net=0, epoch=2)

print(net_json)
'''


'''
df = pd.read_csv('C:/Users/Utente/Desktop/decision_all_long.csv')

df['decision'] = 2
df.to_csv('C:/Users/Utente/Desktop/decision_all_long.csv', header=True, index=False)
'''

'''
' PER ANSELMO PT 1
' GENERO UN CSV UNICO CON DATA - [10 EPOCHE PER RETE]
''
df_complete = pd.DataFrame()

pred_counter = 1

for i in range(0, 90): 
    print("Running net:", i)

    df = pd.read_csv('D:/PhD-Market-Nets/experiments/027 - ripetizione 026/predictions/predictions_during_training/test/walk_0/net_' + str(i) + '.csv')
    #df = pd.read_csv('D:/PhD-Market-Nets/experiments/066 - SP500 proshort loss estrema/predictions/predictions_during_training/test/walk_0/net_' + str(i) + '.csv')

    if i == 0:
        df_complete['date_time'] = df['date_time'].tolist()
        df_complete = df_date_merger(df=df_complete, columns=['open', 'close', 'delta_next_day'], thr_hold=0.3, dataset='sp500_cet')

    #for j in range (700, 711): 
    for j in range (100, 111):
        lista = df['epoch_' + str(j)].tolist()
        df_complete['pred_' + str(pred_counter)] = Measures.pred_modifier(lista)
        pred_counter = pred_counter + 1

df_complete['label_next_day'] = Measures.pred_modifier(df_complete['label_next_day'].tolist())
df_complete = df_complete.drop('label_current_day', 1)

#df_complete.to_csv('C:/Users/Utente/Desktop/Dati per Anselmo/csv_pred.csv', index=False, header=True)
df_complete.to_csv('C:/Users/Utente/Desktop/Exp 66 nuovo approccio/csv_predizioni.csv', index=False, header=True)
'''


'''
' CONVERTIRE UN PREDICTIONS_TRAINING IN PREDIZIONI PER MC
''
df = pd.read_csv('D:/PhD-Market-Nets/experiments/065 - report criteri SP500 - loss 60/predictions/predictions_during_training/validation/walk_0/net_0.csv')
df = df[['date_time', 'epoch_200']]
df['epoch_200'] = df['epoch_200'].shift(1)
df = df.dropna()
df['epoch_200'] = df['epoch_200'].astype(int)
#df.to_csv('C:/Users/Utente/Desktop/fix_short_sl/predictions.csv', header=True, index=False)
'''


'''
' TEST SU EPOCA 200 EXP 60 PER FIXARE STOP LOSS SHORT
' TEST SU BH INTRADAY
''
rh = ResultsHandler(experiment_name='065 - report criteri SP500 - loss 60')

penalty = 25

no_stop_loss = rh.get_result_for_walk_net(start_date='2009-02-01', end_date='2016-07-31', index_walk=0, net=0, epoch=200, penalty=penalty, stop_loss=1000, set_type='validation')
stop_loss = rh.get_result_for_walk_net(start_date='2009-02-01', end_date='2016-07-31', index_walk=0, net=10, epoch=200, penalty=penalty, stop_loss=1000, set_type='validation')


#print("/nReturn Long Only:", no_stop_loss['lh_return'], "| Stop Loss:", stop_loss['lh_return'])
#print("Return Short Only:", no_stop_loss['sh_return'], "| Stop Loss:", stop_loss['sh_return'])

print("/nBH:", no_stop_loss['bh_return'], "| BH2:", no_stop_loss['bh_2_return'])
print("BH:", stop_loss['bh_return'], "| BH2:", stop_loss['bh_2_return'])
'''

df = pd.DataFrame()

df['prima'] = [1, 2, 3, 4, 5]
df['seconda'] = [5, 4, 3, 2, 1]
df.to_csv('C:/Users/Utente/Desktop/prova/my_csv.csv', mode='w', header=True, index=False)

'''
' TEST SU SELEZIONE SELECTIONS CSV
''

experiment_number = '073 - gold and gold silver - pesi bilanciati'
#experiment_number = '074 - gold - pesi bilanciati controprova exp 73'
rh = ResultsHandler(experiment_name=experiment_number)

#rh.calculate_epoch_selection(test_penalty=30, test_stop_loss=1000, long_min_thr=5, long_step=5, short_min_thr=10, short_step=10)


long_metrics = ['por15_grp20', 'por15_grp50', 'por15_grp100', 'por10_grp20', 'por10_grp50', 'por10_grp100', 'por5_grp20', 'por5_grp50', 'por5_grp100']
short_metrics = ['por30_grp20','por30_grp50','por30_grp100','por20_grp20','por20_grp50','por20_grp100','por10_grp20','por10_grp50','por10_grp100']

#for metric in long_metrics:
rh.get_report_excel(type='selection', metric='valid_romad', second_metric="valid_cove", epoch_selection_policy='long', stop_loss=1000, penalty=30, report_name=experiment_number + "- Rep Selection " + "valid romad Romad BH + Por 10" + " - Long Only - SL1000 Pen30")

#for metric in short_metrics:
#rh.get_report_excel(type='selection', metric='valid_romad', second_metric="valid_cove", epoch_selection_policy='short', stop_loss=1000, penalty=30, report_name=experiment_number + "- Rep Selection " + "valid romad Romad BH + Por 10" + " - Short Only - SL1000 Pen30")


#rh.calculate_correlation(operation='long')
#rh.calculate_correlation(operation='short')
'''


'''
' CHECK RISULTATI ANSELMO PT 2
''


def ensemble_magg(df, thr):
    
    # ANSELMO
    #long = (df.eq(1).sum(1) / df.shape[1]).gt(thr) # 1 long
    #short = (df.eq(2).sum(1) / df.shape[1]).gt(thr) # 2 short
    #hold = (df.eq(0).sum(1) / df.shape[1]).gt(thr) # 0 hold

    # FORMATO NOSTRO
    long = (df.eq(1).sum(1) / df.shape[1]).gt(thr)
    short = (df.eq(-1).sum(1) / df.shape[1]).gt(thr)
    hold = (df.eq(0).sum(1) / df.shape[1]).gt(thr)

    #m = pd.DataFrame(np.select([short, long], [0, 2], default=1), index=df.index, columns=['decision'])
    m = pd.DataFrame(np.select([long, hold, short], [2, 1, 0], default=1), index=df.index, columns=['decision'])

    return m

def majority(df):
    
    ''
    # FORMATO ANSELMO
    long_1 = df.eq(1).sum(1) > df.eq(2).sum(1)  # 1 long 2 short
    long_2 = df.eq(1).sum(1) > df.eq(0).sum(1)  # 1 long 0 hold

    short_1 = df.eq(2).sum(1) > df.eq(1).sum(1) # 2 short 1 long
    short_2 = df.eq(2).sum(1) > df.eq(0).sum(1) # 2 long 0 hold
    ''  
    
    # FORMATO NOSTRO
    long_1 = df.eq(1).sum(1) > df.eq(-1).sum(1) 
    long_2 = df.eq(1).sum(1) > df.eq(0).sum(1) 

    short_1 = df.eq(-1).sum(1) > df.eq(1).sum(1)
    short_2 = df.eq(-1).sum(1) > df.eq(0).sum(1)
    
    #m = pd.DataFrame(np.select([l1 & l2, s1 & s2], [2, 0], default=1), index=df.index, columns=['decision'])
    m = pd.DataFrame(np.select([long_1 & long_2, short_1 & short_2], [2, 0], default=1), index=df.index, columns=['decision'])
    return m

    #m = df.mode(axis=1, dropna=True)
    #new_df=  pd.DataFrame()
    #new_df['decision'] = m[0].astype(int)
    
    #return new_df


def get_equity_return_mdd_romad(df, multiplier, delta_to_use='delta_current_day'):
    y_pred = df['decision'].tolist()
    y_pred = Measures.pred_modifier(y_pred, type='long_short')

    delta = df[delta_to_use].tolist()

    equity_line = np.add.accumulate( np.multiply(y_pred, delta) * multiplier)
    
    global_return, mdd, romad, i, j = Measures.get_return_mdd_romad_from_equity(equity_line=equity_line)    

    return equity_line, global_return, mdd, romad, i, j

def get_precision_count_coverage(df, multiplier): 

    y_pred = df['decision'].tolist()
    delta = df['delta_next_day'].tolist()

    long_count, long_guessed = 0, 0

    short_count, short_guessed = 0, 0

    hold_count = 0
    
    y_true = []
    y_true = df['label_next_day'].tolist()

    accuracy = accuracy_score(y_true, y_pred, normalize=True)

    for i, val in enumerate(y_pred):
        # Long
        if val == 2:
            long_count += 1
            if delta[i] >= 0:
                long_guessed += 1
        # Short 
        elif val == 0:
            short_count += 1
            if delta[i] < 0:
                short_guessed += 1
        #elif val == 1.:
        #    hold_count += 1
        #    if delta[i] > -0.2 and delta[i] < 0.2:
        #        hold_guessed += 1
                
    hold_count = len(y_pred) - long_count - short_count

    # percentuale di long e shorts azzeccate
    long_precision = 0 if long_count == 0 else long_guessed / long_count
    short_precision = 0 if short_count == 0 else short_guessed / short_count

    random_val = Measures.get_delta_coverage(delta=df['delta_next_day'])

    if len(y_pred) > 0:
        # percentuale di operazioni di long e shorts sul totale di operazioni fatte
        long_coverage = long_count / (len(y_pred))
        short_coverage = short_count / (len(y_pred))
        hold_coverage = hold_count / (len(y_pred))
    else: 
        long_coverage = 0
        short_coverage = 0
        hold_coverage = 0

    long = {
        "precision": long_precision, 
        "count": long_count,
        "guessed": long_guessed,
        "coverage": long_coverage,
        "random_perc": random_val['long'],
        "random_count": random_val['long_count']
        }
    
    short = {
        "precision": short_precision, 
        "count": short_count,
        "guessed": short_guessed,
        "coverage": short_coverage,
        "random_perc": random_val['short'],
        "random_count": random_val['short_count']
        }
    hold = {
        "count": hold_count,
        "coverage": hold_coverage,
        "random_perc": random_val['hold'],
        "random_count": random_val['hold_count']
        }

    general = {
        "total_trade": long['count'] + short['count'], 
        "total_guessed_trade": long['guessed'] + short['guessed'],
        "total_operation": long['count'] + short['count'] + hold['count'], 
        "accuracy": accuracy
    }

    return long, short, hold, general


def get_return_mdd_romad_bh(close, multiplier):

    close = [i * multiplier for i in close] 
    global_return = close[-1] - close[0]

    cumulative = np.maximum.accumulate(close) - close
    
    i, j = 0, 0
    mdd = np.nan
    if not all(cumulative == 0):
        i = np.argmax(cumulative)
        j = np.argmax(close[:i])

        mdd =  close[j] - close[i]
        romad = global_return / mdd
    else:
        romad = 0

    return close, global_return, mdd, romad, i, j
#####################


# OUR APPROAC SONO GIA' NEL FORMATO -1, 0, 1
#for thr in [0.45, 0.47, 0.47, 0.48, 0.49, 0.5, 0.51, 0.52, 0.53, 0.54, 0.55]:
#for thr in [0.5]:

    
df = pd.read_csv('C:/Users/Utente/Desktop/Test paper anselmo/our_approach_test_set.csv')

# tolgo le ultime 90 colonne
remove_size = df.shape[1] - 90
df = df.drop(df[df.columns[remove_size:]], axis=1)

# EXP 66
#df = pd.read_csv('C:/Users/Utente/Desktop/Exp 66 nuovo approccio/csv_predizioni.csv')

# OOS
start_date = '2017-02-01'
end_date = '2019-10-30'
#df = df.loc[(df['date_time'] >= start_date) & (df['date_time'] <= end_date)]

subdates = df['date_time'].tolist()

df_predictions = df.drop(['open', 'close', 'delta_next_day', 'label_next_day', 'date_time'], 1)
thr = 0.3
while thr < 1.05:

    decisions = ensemble_magg(df=df_predictions, thr=thr)
    #decisions = majority(df=df_predictions)


    ''
    # ANSELMO
    #df = pd.read_csv('C:/Users/Utente/Desktop/Test paper anselmo/final_results_only_long.csv')
    df = pd.read_csv('C:/Users/Utente/Desktop/Test paper anselmo/final_results.csv')
    df_predictions = df.drop(['date_time'], 1)
    decisions = majority(df=df_predictions)
    #decisions = ensemble_magg(df=df_predictions, thr=0.5)[['decision']]
    ''


    final_df = pd.DataFrame()
    final_df['date_time'] = subdates 
    final_df['decision'] = decisions['decision'].tolist()
    final_df = df_date_merger(df=final_df, columns=['close', 'delta_next_day', 'delta_current_day'], thr_hold=0.3, dataset='sp500_cet')

    ''
    saving = final_df[['date_time', 'decision']]
    saving['decision'] = saving['decision'].shift(1)
    saving = saving.dropna()
    saving['decision'] = saving['decision'].astype(int)
    saving.to_csv('C:/Users/Utente/Desktop/Test paper anselmo/csv_multicharts_oos.csv', header=True, index=False)
    ''


    # NON SERVE PER I NOSTRI DATASET majority

    equity_line, global_return, mdd, romad, i, j = get_equity_return_mdd_romad(df=final_df, multiplier=50, delta_to_use='delta_next_day')
    long, short, hold, general = get_precision_count_coverage(df=final_df, multiplier=50)
    print("thr:", thr)
    print("Return:", global_return, "MDD:", mdd, "Romad", romad)
    print("Long Precision:", long['precision'])
    print("Short Precision", short['precision'])
    print("Accuracy", general['accuracy'])
    print("Coverage Long:", long['coverage'], "Coverage Short:", short['coverage'], "Coverage Totale", long['coverage'] + short['coverage'])
    print("N° Long:", long['count'], " N° Short:", short['count'], "Hold:", hold['count'])
    #print("General:", general)
    #print("Hold:", hold)
    print("/n")

    thr = thr + 0.05


#bh_equity_line, bh_global_return, bh_mdd, bh_romad, bh_i, bh_j = get_return_mdd_romad_bh(close=final_df['close'].tolist(), multiplier=50)
#print("BH Return:", bh_global_return, "BH Mdd:", bh_mdd, "Bh Romad:", bh_romad)
'''



'''
' VERIFICA RETURN DI UN EPOCA
''
df = pd.read_csv('D:/PhD-Market-Nets/experiments/066 - SP500 proshort loss estrema/predictions/predictions_post_ensemble/senza-rimozione-reti/test/ensemble_magg/0.5/walk_0.csv')

epoch_22 = df['epoch_22'].tolist() # no intorno

epoch_22 = df['epoch_26'].tolist() # intorno 1 

long = epoch_22.count(2)
short = epoch_22.count(0)
hold = epoch_22.count(1)

tot = long + short + hold

print(long, short, hold)

print(long/tot, short/tot, hold/tot)
'''