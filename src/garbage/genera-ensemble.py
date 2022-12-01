import os
import numpy as np 
import pandas as pd 
from os import listdir
from os.path import isfile, join
from classes.Market import Market

import time
import re

def atoi(text):
    return int(text) if text.isdigit() else text

def natural_keys(text):
    return [ atoi(c) for c in re.split(r'(\d+)', text) ]

'''
'
'''
def get_date_epochs_walk(path, walk): 
    date_list = []
    epochs_list = []

    full_path = path + walk + '/'

    for filename in os.listdir(full_path):
        df = pd.read_csv(full_path + filename)
        date_list = df['date_time'].tolist()
        epochs_list = df.columns.values
        epochs_list = np.delete(epochs_list, [0, 1])
        break       

    return np.array(date_list), np.array(epochs_list)

'''
' Calcola l'ensemable sulle colonne (reti) con una % di agreement
'''
def perc_ensemble(df, thr=0.5):
    df = df.drop(['date_time'], axis=1)

    n_short = (df.eq(0).sum(1) / df.shape[1]).gt(thr)
    n_long = (df.eq(2).sum(1) / df.shape[1]).gt(thr)

    

    n_long.astype(int).mul(2).add(n_short)

    m = pd.DataFrame(np.select([n_short, n_long], [0, 2], 1), index=df.index, columns=['ensemble'])

    return m

'''
'
'''
def elimination_ensemble(df):
    df = df.drop(['date_time'], axis=1)

    return pd.DataFrame( df.eq(2).sum(1) + (-df.eq(0).sum(1)), index=df.index, columns=['ensemble'] )



### DA SETTARE ###

#experiment_name = 'exp_BH_walk_1_mese_SGD_BS_500_GOLDCET'
#experiment_name = 'exp_BH_walk_DEBUG_CSV'
#experiment_name = 'exp_BH_walk_1_mese_SGD_BS_500_2'
#experiment_name = 'exp_sp500_prova_soglia_hold_0.4'
experiment_name = 'exp_sp500_prova_soglia_hold_0.3'


### FINE SETTINGS ###
thr = 0.5
dataset_path = 'C:/Users/Utente/Documents/GitHub/PhD-Market-Nets/experiments/'

validation_folder = '/predictions_validation/'
test_folder = '/predictions_test/'

validation_output_folder = '/predictions_ensemble_validation_' + str(thr) + '/'
test_output_folder = '/predictions_ensemble_test_' + str(thr) + '/'

validation_path = dataset_path + experiment_name + validation_folder
test_path = dataset_path + experiment_name + test_folder

validation_output_path = dataset_path + experiment_name + validation_output_folder
test_output_path = dataset_path + experiment_name + test_output_folder

walks_list = os.listdir(validation_path)
nets_list = os.listdir(validation_path + walks_list[0])

walks_list.sort(key=natural_keys)
nets_list.sort(key=natural_keys)

#walks_list = ['walk_0', 'walk_1', 'walk_2']

######################### ENSEMBLE  ###############################

################################################# 
# VALIDATION SET 
################################################# 

# per ogni walk 
for walk in walks_list:
    start_walk = time.time()
    date_list, epochs_list = get_date_epochs_walk(path=validation_path, walk=walk)

    # Qui salvo il risultato dell'ensemble di ogni rete per ogni epoca
    df_ensemble_magg = pd.DataFrame(columns=[epochs_list])
    df_ensemble_magg['date_time'] = date_list

    df_ensemble_el = pd.DataFrame(columns=[epochs_list])
    df_ensemble_el['date_time'] = date_list

    # Qui inserisco la predizione di ogni rete giorno per giorno 
    df_net = pd.DataFrame(columns=[nets_list])
    df_net['date_time'] = date_list

    reti = []
    for net in range(0, len(nets_list)):
        # Leggo per ogni rete le decisioni, e per la specifica epoca le inserisco nel DF delle predizioni delle reti
        reti.append(pd.read_csv(validation_path + walk + '/' + nets_list[net], engine='c', na_filter=False))

    # per ogni epoca
    for idx, epoch in enumerate(epochs_list):

        start = time.time()

        # per ogni rete
        for i, net in enumerate(nets_list):
            # Leggo per ogni rete le decisioni, e per la specifica epoca le inserisco nel DF delle predizioni delle reti
            df_predizioni = reti[i]
            df_net[net] = df_predizioni[epoch]

        end = time.time()
        print(walk + " | epoca: " + str(idx + 1) + " | ETA: " + "{:.3f}".format(end-start))
            

        # Stampo le predizioni per tutte le reti di questa epoca
        #df_net['date_time'] = pd.to_datetime(df_net['date_time'])
        
        ensemble_magg = perc_ensemble(df=df_net, thr=thr)

        ensemble_el = elimination_ensemble(df=df_net)

        df_ensemble_magg[epoch] = ensemble_magg['ensemble']
        df_ensemble_el[epoch] = ensemble_el['ensemble']
        
        #df_net['ensemble_magg'] = ensemble_magg['ensemble']
        #df_net['ensemble_el'] = ensemble_el['ensemble']
        #result = pd.merge(df_net, ensemble, on='date_time')
    
    # salvo l'ensemble di quest'epoca per questo walk 
    if not os.path.isdir(validation_output_path + 'ensemble_magg/'):
        os.makedirs(validation_output_path + 'ensemble_magg/')

    if not os.path.isdir(validation_output_path + 'ensemble_el/'):
        os.makedirs(validation_output_path + 'ensemble_el/')

    df_ensemble_magg.to_csv(validation_output_path + 'ensemble_magg/ ' + walk + '.csv', header=True, index=False)
    df_ensemble_el.to_csv(validation_output_path + 'ensemble_el/ ' + walk + '.csv', header=True, index=False)

    end_walk = time.time()
    print("TOTAL WALK ELAPSED TIME: "+ "{:.3f}".format(end_walk-start_walk))


################################################# 
# TEST SET 
################################################# 

# per ogni walk 
for walk in walks_list:
    date_list, epochs_list = get_date_epochs_walk(path=test_path, walk=walk)

    # Qui salvo il risultato dell'ensemble di ogni rete per ogni epoca
    df_ensemble_magg = pd.DataFrame(columns=[epochs_list])
    df_ensemble_magg['date_time'] = date_list

    df_ensemble_el = pd.DataFrame(columns=[epochs_list])
    df_ensemble_el['date_time'] = date_list

    reti = []
    for net in range(0, len(nets_list)):
        # Leggo per ogni rete le decisioni, e per la specifica epoca le inserisco nel DF delle predizioni delle reti
        reti.append(pd.read_csv(test_path + walk + '/' + nets_list[net], engine='c', na_filter=False))
    
    # per ogni epoca
    for idx, epoch in enumerate(epochs_list): 

        start = time.time()

        # Qui inserisco la predizione di ogni rete giorno per giorno 
        df_net = pd.DataFrame(columns=[nets_list])
        df_net['date_time'] = date_list

        # per ogni rete
        for i, net in enumerate(nets_list):
            # Leggo per ogni rete le decisioni, e per la specifica epoca le inserisco nel DF delle predizioni delle reti
            df_predizioni = reti[i]
            df_net[net] = df_predizioni[epoch]
            
        end = time.time()
        print(walk + " TEST | epoca: " + str(idx + 1) + " | ETA: " + "{:.3f}".format(end-start))

        # Stampo le predizioni per tutte le reti di questa epoca
        #df_net['date_time'] = pd.to_datetime(df_net['date_time'])
        
        ensemble_magg = perc_ensemble(df=df_net, thr=thr)

        ensemble_el = elimination_ensemble(df=df_net)

        df_ensemble_magg[epoch] = ensemble_magg['ensemble']
        df_ensemble_el[epoch] = ensemble_el['ensemble']
        
        #df_net['ensemble_magg'] = ensemble_magg['ensemble']
        #df_net['ensemble_el'] = ensemble_el['ensemble']
        #result = pd.merge(df_net, ensemble, on='date_time')
    
    # salvo l'ensemble di quest'epoca per questo walk 
    if not os.path.isdir(test_output_path + 'ensemble_magg/'):
        os.makedirs(test_output_path + 'ensemble_magg/')

    if not os.path.isdir(test_output_path + 'ensemble_el/'):
        os.makedirs(test_output_path + 'ensemble_el/')

    df_ensemble_magg.to_csv(test_output_path + 'ensemble_magg/ ' + walk + '.csv', header=True, index=False)
    df_ensemble_el.to_csv(test_output_path + 'ensemble_el/ ' + walk + '.csv', header=True, index=False)

    