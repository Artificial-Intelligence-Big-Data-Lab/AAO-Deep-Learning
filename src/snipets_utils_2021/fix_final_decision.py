import os, shutil
from classes.Market import Market
from classes.Measures import Measures
from classes.Utils import progressBar, df_date_merger, create_folder
import pandas as pd 
import matplotlib.pyplot as plt
from cycler import cycler
import json 

base_path = 'D:/PhD-Market-Nets/experiments/'
experiments = ['130 - SP500 walk 1 anno sino al 2020 con nuovi dati pesi bilanciati']

'''
base_path = '/media/unica/HDD 9TB Raid0 - 1/experiments/'
experiments = [
    '126 - SP500 walk 1 anno sino al 2020',
    '127 - SP500 walk 6 mesi sino al 2020',
    '128 - SP500 walk 1 anno sino al 2020 con nuovi dati',
    '129 - SP500 walk 1 anno sino al 2020 pesi bilanciati',
    '130 - SP500 walk 1 anno sino al 2020 con nuovi dati pesi bilanciati',
    '131 - SP500 walk 1 anno sino al 2020 sp500 e vix pesi long',
    '132 - SP500 walk 1 anno sino al 2020 sp500 e vix pesi bilanciati',
    '133 - SP500 walk 1 anno sino al 2020 sp500 e vix pesi short',
    '134 - SP500 walk 1 anno sino al 2020 sp500 e vix pesi bilanciati'
]
'''
algs = [3, 4]
ensembles = ['ensemble_exclusive', 'ensemble_exclusive_short', 'ensemble_magg']
#ensembles = ['ensemble_exclusive']
iperparameters = {}


##########################
def get_selections_nets(ensemble):
    if ensemble == 'ensemble_exclusive':
        return ['long_only']
    if ensemble == 'ensemble_exclusive_short':
        return ['short_only']
    if ensemble == 'ensemble_magg':
        return ['long_only', 'short_only', 'long_short']


def get_alg(alg): 
    if alg == 3:
        return '/predictions/final_decisions'
    if alg == 4:
        return '/predictions/final_decision_alg4'


def copytree(src, symlinks=False, ignore=None):
    for item in os.listdir(src):
        s = os.path.join(src, item)
        d = os.path.join(src + '_BACKUP', item)

        if os.path.isdir(s):
            shutil.copytree(s, d, symlinks, ignore)
        else:
            shutil.copy2(s, d)

for experiment in experiments:
    print("Start handling with", experiment)

    with open(base_path + experiment + '/log.json') as json_file:
        iperparameters = json.load(json_file)

    # ottimizzo. lo creo una volta e basta
    df_with_dates = pd.DataFrame()
    dates = []

    for alg in algs: 
        alg_str = get_alg(alg)

        copytree(src=base_path + experiment + alg_str)

        for ensemble in ensembles: 
            selections_nets = get_selections_nets(ensemble)
            for selection in selections_nets: 
                full_path = base_path + experiment + alg_str + '/' + ensemble + '/senza-rimozione-reti/' + selection + '/'  
                files = os.listdir(full_path)
                #files = files[:1] #debug

                suffix = "Alg: " + str(alg) + " - Ensemble: " + ensemble + " - Selection: " + selection 
                for file in progressBar(files, suffix=suffix): 
                    df = pd.read_csv(full_path + file)
                    df = df[['date_time', 'decision']]

                    if df_with_dates.empty:
                        df_with_dates = df_date_merger(df=df.copy(), thr_hold=iperparameters['hold_labeling'], columns=['open', 'close', 'high', 'low', 'delta_current_day'], dataset=iperparameters['predictions_dataset'])
                        df_with_dates = df_with_dates.drop(columns=['decision'])

                        dates = df_with_dates['date_time'].tolist()

                    dates_current = df['date_time'].tolist()
                    #print("DF base")
                    #print(df.shape)

                    diff1 = list(set(dates_current) - set(dates))
                    diff2 = list(set(dates) - set(dates_current))

                    df = df[~df.date_time.isin(diff1)]
                    df = df[~df.date_time.isin(diff2)]
                    
                    #print("DF after elimination diff")
                    #print(df.shape)

                    #df = pd.merge([df, df_with_dates], axis=1, join='inner', keys=['date_time', 'decision'])
                    df = df.merge(df_with_dates, left_on='date_time', right_on='date_time', how='inner')
                    #df =    df_date_merger(df=df.copy(), thr_hold=iperparameters['hold_labeling'], columns=['open', 'close', 'high', 'low', 'delta_current_day'], dataset=iperparameters['predictions_dataset'])
                    #print("DF final")
                    #print(df)

                    df = df.to_csv(full_path + file, header=True, index=False)
                #print("working for", full_path)
            