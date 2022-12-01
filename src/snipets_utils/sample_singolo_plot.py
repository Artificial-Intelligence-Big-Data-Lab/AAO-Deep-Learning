import os.path
from os import path
import time
import pandas as pd 
import numpy as np
from datetime import timedelta
from classes.Market import Market
from classes.Utils import create_folder, df_date_merger
import cv2
from sklearn.metrics import accuracy_score
import json
import threading
import matplotlib.pyplot as plt
from cycler import cycler
import statistics

'''
' Calcolo il file di log con l'accuracy sul campione 
'''
def calculate_champ_acc(df, target_day, nets=30, column='rmse', thr=0.8):
    #print("Calculating accuracy for champ with", column, "metric, thr:" + str(thr) + "...")

    y_true_global = []
    y_pred_global = []
    tot_nets = 0
    
    for net in range(0, nets):
        rows = df.loc[df['net_' + str(net)] >= thr]

        seleced_epochs = rows['epoch'].tolist()
        tot_nets = tot_nets + len(seleced_epochs)

        for epoch in seleced_epochs: 
            y_pred_global.append(target_row_df['epoch_' + str(epoch)].iloc[0])
            y_true_global.append(target_label)

    if len(y_true_global) > 0:
        global_acc = accuracy_score(y_true_global, y_pred_global)
    else:
        global_acc = 0


    return global_acc





########################
#  CONFIGURAGION ZONE 
########################
notebook = False
user = 'Utente'
if notebook == True: 
    user = 'Andrea'

dataset = 'sp500_cet'

#target_exp = '087 - Test univariate SP500 unico blocco 1h'
target_exp = '080 - Test multivariate SP500 con tutti i blocchi'

target_epoch = 'epoch_1'
target_net = 0
target_walk = 7
label = 'label_next_day'

# se voglio solo le long = 2, short = 0, hold = 1
# disattivare = -1
ONLY_SAMPLE = 0

########################
#  END CONFIGURAGION ZONE 
########################

'''
' Creo i dataframe con le serie storiche per ogni giorno 
' ogni colonna sarà un giorno e le righe saranno la serie storica
' associata a quel giorno [da lanciare una volta per pc]
'''
#create_dataset_csv(small=True)
#create_dataset_csv()


log = ''
with open('D:/PhD-Market-Nets/experiments/' + target_exp + '/log.json') as json_file:
    log = json.load(json_file)

predictions_locale = pd.read_csv('D:/PhD-Market-Nets/experiments/' + target_exp + '/predictions/predictions_during_training/test/walk_' + str(target_walk) + '/net_' + str(target_net) + '.csv')
predictions_locale = df_date_merger(df=predictions_locale, thr_hold=0.3, columns=[], dataset=dataset)

# i giorni del walk
first_day_available = predictions_locale['date_time'].tolist()[0]
last_day_available = predictions_locale['date_time'].tolist()[-1]

# tolgo i primi due mesi, così posso avere più samples per fare confronti
predictions_locale = predictions_locale[40:]

# prendo la lista dei giorni da ciclare
date_time_list = predictions_locale['date_time'].tolist()

print("THR - Acc media metrica 1 (RMSE) - Acc media metrica 2 (tm_sqdiff_normed)")
string = "THR - Acc media metrica 1 (RMSE) - Acc media metrica 2 (tm_sqdiff_normed)\n"
for thr in [0.95, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3]:

    global_acc_rmse = []
    global_acc_tm = []

    diff = 0 
    #print("Calculatin accuracy day by day for thr:", thr)
    for target_day in date_time_list:

        # dataframe contenente la riga del target_day
        target_row_df = predictions_locale.loc[predictions_locale['date_time'] == target_day]
        target_prediction = target_row_df.iloc[0][target_epoch]
        target_label = target_row_df.iloc[0][label]

        if ONLY_SAMPLE != -1:
            if target_label != ONLY_SAMPLE: 
                diff += 1
                continue

        if  os.path.isfile('C:/Users/' + user + '/Desktop/Risultati sample singolo/' + target_exp + '/walk_7/' + target_day + '/rmse.csv') == False:
            diff +=1
            continue

        # per velocizzare il debug
        rmse_accuracies = pd.read_csv('C:/Users/' + user + '/Desktop/Risultati sample singolo/' + target_exp + '/walk_7/' + target_day + '/rmse.csv')
        tm_sqdiff_normed_accuracies = pd.read_csv('C:/Users/' + user + '/Desktop/Risultati sample singolo/' + target_exp + '/walk_7/' + target_day + '/tm_sqdiff_normed.csv')

        # RISULTATI MEDIE ACCURACY CAMPIONE
        rmse_accuracy = calculate_champ_acc(df=rmse_accuracies, target_day=target_day, nets=log['number_of_nets'], column='rmse', thr=thr)
        global_acc_rmse.append(rmse_accuracy)
    
        tm_accuracy = calculate_champ_acc(df=tm_sqdiff_normed_accuracies, target_day=target_day, nets=log['number_of_nets'], column='tm_sqdiff_normed', thr=thr)
        global_acc_tm.append(tm_accuracy)

    
    print(str(thr) + " - " + "{:.3f}".format(statistics.mean(global_acc_rmse)) + " - " +  "{:.3f}".format(statistics.mean(global_acc_tm)) )
    string += str(thr) + " - " + "{:.3f}".format(statistics.mean(global_acc_rmse)) + " - " +  "{:.3f}".format(statistics.mean(global_acc_tm)) + "\n"

    plt.figure(figsize=(30,18))

    # plot the training loss and accuracy
    plt.style.use("ggplot")
    plt.rcParams['axes.prop_cycle'] = cycler(color='bgrcmyk')
    plt.subplot(2, 1, 1)

    plt.xlabel("Days")
    plt.ylabel("Accuracy")
    plt.plot(np.arange(0, len(date_time_list) - diff), global_acc_rmse, label="RMSE")
    #plt.xticks(rotation=90)
    plt.legend(loc="upper left")

    plt.subplot(2, 1, 2)
    plt.xlabel("Days")
    plt.ylabel("Accuracy")

    plt.plot(np.arange(0, len(date_time_list) - diff), global_acc_tm, label="tm_sqdiff_normed")
    #plt.xticks(rotation=90)
    plt.legend(loc="upper left")


    base_path = 'C:/Users/' + user + '/Desktop/Risultati sample singolo/' + target_exp + '/walk_7/accuracy_plot_medie/'

    if ONLY_SAMPLE == 2: 
        base_path = base_path + 'solo long/'
    
    if ONLY_SAMPLE == 0: 
        base_path = base_path + 'solo short/'
    
    if ONLY_SAMPLE == 1: 
        base_path = base_path + 'solo hold/'

    if ONLY_SAMPLE == -1: 
        base_path = base_path + 'tutti i sample/'

    create_folder(base_path)

    # salvo il log
    text_file = open(base_path + "medie.txt", "w")
    text_file.write(string)
    text_file.close()
    
    plt.savefig(base_path + 'all_accuracies_thr_' + str(thr) + '.png')

    plt.close('all')