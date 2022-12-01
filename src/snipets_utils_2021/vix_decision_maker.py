import pandas as pd
from classes.Measures import Measures
from classes.Market import Market
from classes.ResultsHandler import ResultsHandler
from classes.Utils import df_date_merger, create_folder
import json 


dataset = 'vix_cet'

thrs = []
exp_name = '001 - Prima prova con Vix'
#exp_name = '002 - Prova con Vix con soglie step 2'
#exp_name = '003 - Prova con Vix con soglie step 1'


for x in range(0, 100, 5):
    thrs.append(x)

training_set = [['2000-01-01', '2009-12-31'], 
        ['2000-01-01', '2010-12-31'], 
        ['2000-01-01', '2011-12-31'], 
        ['2000-01-01', '2012-12-31'], 
        ['2000-01-01', '2013-12-31'], 
        ['2000-01-01', '2014-12-31'], 
        ['2000-01-01', '2015-12-31'], 
        ['2000-01-01', '2016-12-31'], 
        ['2000-01-01', '2017-12-31'], 
        ['2000-01-01', '2018-12-31'] 
 ]

validation_set = [['2010-01-01', '2010-12-31'], 
        ['2011-01-01', '2011-12-31'], 
        ['2012-01-01', '2012-12-31'], 
        ['2013-01-01', '2013-12-31'], 
        ['2014-01-01', '2014-12-31'], 
        ['2015-01-01', '2015-12-31'], 
        ['2016-01-01', '2016-12-31'], 
        ['2017-01-01', '2017-12-31'], 
        ['2018-01-01', '2018-12-31'], 
        ['2019-01-01', '2019-12-31']
 ]

test_set = [['2011-01-01', '2011-12-31'], 
        ['2012-01-01', '2012-12-31'], 
        ['2013-01-01', '2013-12-31'], 
        ['2014-01-01', '2014-12-31'], 
        ['2015-01-01', '2015-12-31'], 
        ['2016-01-01', '2016-12-31'], 
        ['2017-01-01', '2017-12-31'], 
        ['2018-01-01', '2018-12-31'], 
        ['2019-01-01', '2019-12-31'], 
        ['2020-01-01', '2020-07-31']]

iperparameters = { 
    'experiment_name': exp_name,
    'return_multiplier': 50, #25 dax, moltiplicatore per convertire i punti di mercato in $
    'training_set': training_set,
    'validation_set': validation_set,
    'test_set': test_set,
    'input_datasets': ['sp500_cet'],
    'predictions_dataset': 'sp500_cet',
    'input_images_folders': ['merge/merge_sp500_vix_hours_days/gadf/multivariate/'],
    'predictions_images_folder': 'merge/merge_sp500_vix_hours_days/gadf/multivariate/',
    'hold_labeling': 0.3,
    'use_probabilities': False,
    'experiment_path': '/media/unica/HDD 9TB Raid0 - 1/vix-experiments/', 
    'description': '<p>In questo exp le decisioni vengono prese guardando il dataset Vix. Se il <b>close</b> del giorno prima è minore di una certa soglia faccio long, altrimenti Idle. Le soglie vanno da 0 a 90 con step +5. A livello di codice le decisioni sono generate così:</p><pre>df.loc[df["close"] < thr, "decision"] = 2</pre>',
    'thrs': thrs
}

base_path = 'D:/PhD-Market-Nets/vix-experiments/' + exp_name + '/'
path = base_path + 'predictions/final_decisions/'
create_folder(path)

with open(base_path + '/log.json', 'w') as json_file:
    json.dump(iperparameters, json_file, indent=4)


sp500 = Market(dataset='sp500_cet')
sp500 = sp500.group(freq='1d')
sp500 = Market.get_df_by_data_range(df=sp500.copy(), start_date="2011-01-01", end_date="2019-12-31")
sp500 = sp500[['date_time', 'open', 'close', 'delta_current_day', 'delta_next_day', 'high', 'low']]
sp500['decision'] = 2

bh_intraday_equity_line, bh_intraday_global_return, bh_intraday_mdd, bh_intraday_romad, bh_intraday_i, bh_intraday_j = Measures.get_equity_return_mdd_romad(df=sp500.copy(), multiplier=50, type='bh_long', penalty=25, stop_loss=1000, delta_to_use='delta_current_day')

print(bh_intraday_global_return)
print(bh_intraday_mdd)
print(bh_intraday_romad)
input()

market = Market(dataset=dataset)

df = market.group(freq='1d')

df = df[['close']]
df['decision'] = 1
for thr in thrs: 
    df.loc[df['close'] < thr, 'decision'] = 2
    long = len(df[(df.decision == 2)])
    idle = len(df[(df.decision == 1)])
    tot = long + idle
    #print(df)
    print("thr:", thr)
    print("Long totali: ", long, " - in %: ", (long / tot) * 100)
    print("Idle totali: ", idle, " - in %: ", (idle / tot) * 100)

    dff = df[['decision']].shift(1)
    dff = dff.dropna()
    dff['decision'] = dff['decision'].astype(int)

    dff = Market.get_df_by_data_range(df=dff.copy(), start_date=iperparameters['test_set'][0][0], end_date=iperparameters['test_set'][-1][-1])

    dff = df_date_merger(df=dff.copy(), thr_hold=iperparameters['hold_labeling'],
         dataset=iperparameters['predictions_dataset'], columns=['close', 'open', 'delta_current_day', 'delta_next_day', 'high', 'low'])

    dff.to_csv(path + 'final_decision_' + str(thr) + '.csv', header=True, index=True)