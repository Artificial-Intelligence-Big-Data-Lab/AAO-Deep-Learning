import os
import numpy as np 
import pandas as pd 
from os import listdir
from os.path import isfile, join
from classes.Market import Market
import matplotlib.pyplot as plt
import statistics 
from classes.Market import Market
import time
import re

### DA SETTARE ###
#experiment_name = 'exp_BH_walk_DEBUG_CSV'
#experiment_name = 'exp_BH_walk_1_mese_SGD_BS_500_2'
experiment_name = 'exp_BH_walk_1_mese_SGD_BS_500_SP500_thr0.3'
experiment_name = 'Esperimento 16 Walks 6mesi , SP500_CET, SGD BS 300, Labeling hold 0.3, Salvatore Capodanno (seba)/'

### FINE SETTINGS ###
def atoi(text):
    return int(text) if text.isdigit() else text

def natural_keys(text):
    return [ atoi(c) for c in re.split(r'(\d+)', text) ]

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

    cumulative = np.maximum.accumulate(series) - series
    
    if all(cumulative == 0): 
        return 999, 999, 999, 0, 0

    # calcolo la posizione i-esima
    i = np.argmax(cumulative) # end of the period

    # calcolo la posizione j-esima
    j = np.argmax(series[:i]) # start of period

    # mdd
    mdd = series[j] - series[i] 
    # romad
    romad = global_return / mdd 
    # return totale
    #bh_global_return = sum(series) 

    return global_return, mdd, romad, i, j

#thrs = [0.33, 0.35, 0.4, 0.45, 0.475, 0.45, 0.5]
#thrs = [0.4, 0.45, 0.475, 0.45, 0.5]
thrs = [0.35]

'''
for thr in thrs:
    df_GLOBALE = pd.DataFrame(columns=['date_time', 'close', 'delta', 'label'])

    dataset_path = '/home/unica/PhD-Market-Nets/experiments/'
    dataset_path = 'C:/Users/Utente/Documents/GitHub/PhD-Market-Nets/experiments/Esperimenti Vacanze/'

    validation_input_folder = '/predictions_ensemble/validation/' + str(thr) + '/'
    validation_input_path = dataset_path + experiment_name + validation_input_folder + 'ensemble_magg/'


    test_input_folder = '/predictions_ensemble/test/' + str(thr) + '/'
    test_input_path = dataset_path + experiment_name + test_input_folder + 'ensemble_magg/'

    # sp500 wrapper
    sp500 = Market(dataset='sp500_cet')
    sp500_label = sp500.get_label_next_day(freq='1d', columns=['open', 'close', 'delta'])
    sp500_label = sp500_label.reset_index()
    sp500_label['date_time'] = sp500_label['date_time'].astype(str)

    walks_list = sorted(os.listdir(validation_input_path))
    walks_list.sort(key=natural_keys)

    # calcolo il return walk per walk e epoca per epoca
    for index_walk, walk in enumerate(walks_list): 

        #if index_walk is not 14:
        #    continue

        ################################################# 
        # VALIDATION SET 
        #################################################

        # leggo le predizioni fatte con l'esnemble
        df_ensemble = pd.read_csv(validation_input_path + walk)

        # mergio con le label, così ho un subset del df con le date che mi servono e la predizione 
        df_merge_with_label = pd.merge(df_ensemble, sp500_label, how="inner")

        number_of_epochs = df_ensemble.shape[1] - 1 # conto le epoche - 1 (c'è il date_time)

        return_epochs = np.zeros(number_of_epochs)

        # calcolo il return per un epoca
        for i in range(1, number_of_epochs): 
            
            delta = df_merge_with_label['delta'].tolist()
            datetime = df_merge_with_label['date_time'].tolist()
            y_pred = df_merge_with_label['epoch_' + str(i)].tolist()

            delta =  delta[1:]
            delta.append(0)

            # Computing total return
            pred = y_pred
            pred = [-1 if x==0 else x for x in pred]
            pred = [0 if x==1 else x for x in pred] 
            pred = [1 if x==2 else x for x in pred]

            return_tot = np.sum(np.multiply(pred, delta) * 50)
            return_epochs[i] = return_tot

        
        # DA SETTARE L'INTORNO DESIDERATO 
        intorno = 15

        # Ritorno medio con l'intorno
        mean_returns = []

        i = 0
        for s_return in return_epochs: 
            if i >= intorno and i <= (number_of_epochs - intorno): 
                mean_returns.append(statistics.mean(return_epochs[i-intorno:i+intorno]))

            i = i + 1

        # seleziono l'epoca con il return migliore a cui aggiungo l'intorno per selezionarlo int est
        max_index = np.argmax(mean_returns) +  intorno
        #print("Walk", index_walk, "Seleziono epoca n°", max_index)


        ################################################# 
        # TEST SET 
        #################################################
        walk = walk.replace(" ", "")
        df_ensemble = pd.read_csv(test_input_path + walk)
        #df_ensemble['date_time'] = pd.to_datetime(df_ensemble['date_time'])
        
        ''
        start_date = pd.to_datetime(df_ensemble.iloc[0].date_time)
        end_date = (start_date + pd.DateOffset(months=1))

        df_ensemble = Market.get_df_by_data_range(df=df_ensemble, start_date=start_date, end_date=end_date)
        

        #torno alla stringa per fare il merge
        df_ensemble['date_time'] = df_ensemble['date_time'].dt.date
        df_ensemble['date_time'] = df_ensemble['date_time'].apply(str)
        ''

        # mergio con le label, così ho un subset del df con le date che mi servono e la predizione 
        df_merge_with_label = pd.merge(df_ensemble, sp500_label, how="inner")
        #df_merge_with_label = df_merge_with_label.set_index('index')
        subset_column = df_merge_with_label[['epoch_' + str(max_index), 'close', 'delta', 'date_time']]
        subset_column = subset_column.rename(columns={"epoch_" + str(max_index): "label"})

        df_GLOBALE = pd.concat([df_GLOBALE, subset_column])

        # TEMPORARY
        delta = subset_column['delta'].tolist()
        datetime = subset_column['date_time'].tolist()

        y_pred = subset_column['label'].tolist()

        delta =  delta[1:]
        delta.append(0)

        # Computing total return
        pred = y_pred
        pred = [-1 if x==0 else x for x in pred]
        pred = [0 if x==1 else x for x in pred] 
        pred = [1 if x==2 else x for x in pred]

        long_count = 0
        long_guessed = 0
        short_count = 0
        short_guessed = 0

        for i, val in enumerate(y_pred):
            if val == 2:
                long_count += 1
                if delta[i] >= 0:
                    long_guessed += 1
            elif val == 0.:
                short_count += 1
                if delta[i] < 0:
                    short_guessed += 1
                    

        # percentuale di long e shorts azzeccate
        longs_precision = 0 if long_count == 0 else long_guessed / long_count
        shorts_precision = 0 if short_count == 0 else short_guessed / short_count

        daily_returns = np.multiply(pred, delta)

        equity_line = np.add.accumulate(daily_returns)
        return_tot = np.sum(daily_returns)
        
        global_return, mdd, romad, i, j = get_mdd_romad(equity_line)
        print("Global return: ", global_return)
        print("Return tot: ", return_tot)
        print("Walk n°", index_walk, "Soglia di ensemble: ", thr)
        #print("Walk n°", index_walk, " | Return: ", global_return, " | Mdd: ", round(mdd, 2), " | Romad: ", round(romad, 2), " | Lower: ", equity_line[i], " | Upper: ", equity_line[j])
        print("Walk n°", index_walk, " | Return: ", global_return, " | Mdd: ", round(mdd, 2), " | Romad: ", round(romad, 2))

        print("Numero di long: ", y_pred.count(2), "Precision long: ", longs_precision)
        print("Numero di short: ", y_pred.count(0), "Precision short: ", shorts_precision)
        print("Numero di hold: ", y_pred.count(1))
        print("\n ----- \n")

    # RIMUOVO EVENTUALI DATE DUPLICATE + SHIFTO LA LABEL DI 1 GIORNO COSI' A FINANCO DI OGNI DATA HO LA DECISIONE DA PRENDERE 
    df_GLOBALE = df_GLOBALE.drop_duplicates(subset='date_time', keep="first")

    df_GLOBALE['label'] = df_GLOBALE['label'].shift(1)
    df_GLOBALE = df_GLOBALE.drop(columns=['close', 'delta'], axis=1)
    df_GLOBALE = df_GLOBALE.drop(df_GLOBALE.index[0])
    df_GLOBALE['label'] = df_GLOBALE['label'].astype(int)
    df_GLOBALE.to_csv(dataset_path + experiment_name + '/final_decisions/decisions_ensemble_' + str(thr) + '.csv', header=True, index=False)
'''


    df_GLOBALE = pd.read_csv(dataset_path + experiment_name + '/final_decisions/decisions_ensemble_' + str(thr) + '.csv')

    close_BH = df_GLOBALE['close'].tolist()
    close_BH = [x * 50 for x in close_BH]
    global_return, mdd, romad, i, j = get_mdd_romad_bh(close_BH)
    print("Return BH: ", global_return, " | Mdd BH: ", round(mdd, 2), " | Romad BH: ", round(romad, 2), " | Lower BH: ", close_BH[i], " | Upper BH: ", close_BH[j])

    delta = df_GLOBALE['delta'].tolist()
    datetime = df_GLOBALE['date_time'].tolist()

    y_pred = df_GLOBALE['label'].tolist()

    delta =  delta[1:]
    delta.append(0)

    # Computing total return
    pred = y_pred
    pred = [-1 if x==0 else x for x in pred]
    pred = [0 if x==1 else x for x in pred] 
    pred = [1 if x==2 else x for x in pred]

    daily_returns = np.multiply(pred, delta) * 50

    equity_line = np.add.accumulate(daily_returns)
    return_tot = np.sum(daily_returns)


    global_return, mdd, romad, i, j = get_mdd_romad(equity_line)


    #print("return_tot: ", return_tot)
    print("Return: ", global_return, " | Mdd: ", round(mdd, 2), " | Romad: ", round(romad, 2), " | Lower: ", equity_line[i], " | Upper: ", equity_line[j])
    #print("Return Ann: ", round(global_return / 7, 2), " | Mdd Ann: ", round(mdd / 7, 2), " | Romad Ann: ", round(romad / 7, 2))


#norm_eqt = [float(i)/max(equity_line) for i in equity_line]
#norm_bh = [float(i)/max(close_BH) for i in close_BH]
#x_vect = range(0, len(equity_line))
#plt.figure(figsize=(15,12))
#plt.xlabel("Dates")
#plt.ylabel("Dollars")
#plt.plot(x_vect, equity_line)
#plt.plot(x_vect[i], equity_line[i], color='red', linestyle='dashed', marker='o')
#plt.plot(x_vect[j], equity_line[j], color='red', linestyle='dashed', marker='o')
#plt.plot(range(0, len(equity_line)), close_BH, color='red')
#plt.plot(range(0, len(equity_line)), norm_eqt)
#plt.plot(range(0, len(equity_line)), norm_bh, color='red')
#plt.show()
