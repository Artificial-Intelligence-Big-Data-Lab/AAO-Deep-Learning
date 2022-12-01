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

'''
' Metodo di supporto a create_dataset_csv
'''
def run(df, size, target_df):
    # Tecnicamente non dovrebbero esserci valori na, per sicurezza rieseguo
    df = df.dropna().reset_index()

    # Prendo la data dell'ultima riga e ci sommo un giorno.
    # In questo modo all'interno del ciclo posso effettuare un controllo sul giorno corrente
    # ed assicurarmi di calcolare le gadf e le gasf una volta per giorno. E' importante perché
    # se ho raggruppato il dataset per 1 ora, avrei 24 righe per giorno e di conseguenza lo script proverebbe
    # a calcolare 24 diverse immagini per giorno. Invece, aggiungendo il controllo sulla data, skippo tutte le righe
    # che hanno la stessa data del giorno che ho già calcolando, shiftando semplicemente di 1 giorno ogni volta
    # che calcolo le gadf o gasf
    last_iterated_day = (df['date_time'][df.index[-1]]).date() + timedelta(days=1)

    # Itero per ogni riga. Per ogni giorno calcolo a ritrovo le GADF e GASF
    # in base ai parametri specificati in precedenza
    for index, row in df[::-1].iterrows():

        # Ottengo il giorno corrente della riga
        current_day = row['date_time'].date()

        # Come specificato nel commento alla variabile last_iterated_day
        # controllo se ho già calcolato questo giorno controllando che il giorno corrente
        # sia maggiore o uguale all'ultimo iterato.
        if current_day >= last_iterated_day:
            continue

        # Uso la maschera per prendere solo le righe a partire dalla data corrente
        mask = df['date_time'] <= row['date_time']
        subset = df.loc[mask]

        dates = subset['date_time'].tolist()
        # Rimuovo la colonna date_time che non mi serve per il calcolo della trasformata
        subset = subset.drop('date_time', axis=1)
        # Ottengo un sottoinsieme del DataFrame, prendendo le n righe specificate con size
        df_range = subset.tail(size)

        # Controllo se la dimensione e' la stessa. Se non lo è vuol dire che il ciclo
        # e' arrivato quasi alla fine, quindi blocco l'esecuzione
        if df_range.shape[0] == size:
            print("Running:", current_day)
            target_df[current_day] = df_range['delta_current_day'].tolist()

            last_iterated_day = last_iterated_day - timedelta(days=1)

        # Se tail restituisce un subset di dimensione minore a quello specificato in size, devo uscire
        else:
            break

'''
' Calcola RMSE tra due timeseries
'''
def rmse(predictions, targets):
    return np.sqrt(((predictions - targets) ** 2).mean())

'''
' Calcola tm_sqdiff_normed tra due img
'''
def tm_sqdiff_normed(template, img):
    return cv2.matchTemplate(templ=template, image=img, method=cv2.TM_SQDIFF_NORMED)[0][0]

'''
' Genera il datasets con ogni colonna un giorno
' e la serie storica associata ad una gaf di quel giorno
'''
def create_dataset_csv(dataset='sp500_cet', freq='1d', small=False):
    datasets_gaf = pd.DataFrame()

    m = Market(dataset='sp500_cet')
    one_day = m.group(freq='1d')

    if small == True: 
        one_day = Market.get_df_by_data_range(df=one_day, start_date='2010-01-01', end_date='2011-12-31')

    start = time.time()
    run(df=one_day, size=20, target_df=datasets_gaf)
    end = time.time()

    create_folder('C:/Users/' + user + '/Documents/GitHub/PhD-Market-Nets/datasets/sp500_cet_gaf_df/')

    if small == True: 
        datasets_gaf.to_csv('C:/Users/' + user + '/Documents/GitHub/PhD-Market-Nets/datasets/sp500_cet_gaf_df/one_day_small.csv', header=True, index=False)
    else:
        datasets_gaf.to_csv('C:/Users/' + user + '/Documents/GitHub/PhD-Market-Nets/datasets/sp500_cet_gaf_df/one_day_full.csv', header=True, index=False)
    print("Total walk elapsed time: " + "{:.3f}".format(end - start) + " seconds.")


'''
' Restituisce due dataframe 
' con i tot x sample più simili a quello scelto come target_day
'''
def get_df_with_metrics(target_day, top_number=10):
    
    basepath = 'C:/Users/' + user + '/Documents/GitHub/PhD-Market-Nets/images/sp500_cet/1hours/gadf/delta_current_day/'

    # leggo il dataframe contenente per ogni colonna la data e la serie storica
    df = pd.read_csv('C:/Users/' + user + '/Documents/GitHub/PhD-Market-Nets/datasets/sp500_cet_gaf_df/one_day_full.csv')

    # la serie storica del target_day
    target_list = df[target_day].tolist()
    target_img = cv2.imread(basepath + target_day + '.png')

    # La struttura è la seguente
    # target_date è il giorno che uso come campione per essere paragonato agli altri
    # date_time sono gli altri giorni. la colonna si chiama così per poter riutilizzare altri metodi del FW per filtrare dati
    # RMSE contiene la distanza tra le serie storiche associate alle due colonne
    # tm_sqfidd_normed la similarità tra le due img associalte alle due colonne 
    df_final = pd.DataFrame(columns=['target_date', 'date_time', 'rmse', 'tm_sqdiff_normed'])

    print("Generating dataframe date - rmse - tm_sqdiff_normed....")
    for name, value in df.iteritems():
        current_img = cv2.imread(basepath + name + '.png')

        rmse_val = rmse(target_list, value)
        tm_sqdiff_normed_val = tm_sqdiff_normed(target_img, current_img)
        #print("RMSE", target_day, "-", name, '{:.3f}'.format(rmse_val), "- TM_SQDIFF_NORMED:", '{:.3f}'.format(tm_sqdiff_normed_val))

        df_final = df_final.append(pd.DataFrame({"target_date": [target_day],  "date_time": [name], "rmse": [rmse_val], "tm_sqdiff_normed": [tm_sqdiff_normed_val] })) 

    #df_final.to_csv('C:/Users/Andrea/Desktop/df_final/prova.csv', header=True, index=False)
    #input("salvato")
    #df_final = pd.read_csv('C:/Users/Andrea/Desktop/df_final/prova.csv')

    df_final = df_final.sort_values('date_time')
    
    # in questo caso tutti i sample simili sono antecedenti al giorno scelto come campione
    df_final = Market.get_df_by_data_range(df=df_final, start_date=first_day_available, end_date=target_day)
    # in questo caso i sample simili possono essere antecedenti o successivi al giorno scelto come campione
    #df_final = Market.get_df_by_data_range(df=df_final, start_date=first_day_available, end_date=last_day_available)

    # ascendente, più è piccolo il valore più è simile
    sort_rmse = df_final.sort_values('rmse', ascending=True)
    sort_rmse = sort_rmse.iloc[1:] # elimino il confronto con se stesso
    sort_rmse = sort_rmse.head(top_number) # prendo i 10 valori più vicini
    sort_rmse = df_date_merger(df=sort_rmse, thr_hold=0.3, columns=[], dataset=dataset)
    sort_rmse = sort_rmse.drop(columns=['label_current_day', 'tm_sqdiff_normed'])

    # decrescente, più è grande il valore più è simile
    sort_tm_sqdiff_normed = df_final.sort_values('tm_sqdiff_normed', ascending=False)
    sort_tm_sqdiff_normed = sort_tm_sqdiff_normed.iloc[1:] # elimino il confronto con se stesso
    sort_tm_sqdiff_normed = sort_tm_sqdiff_normed.head(top_number) # prendo i 10 valori più vicini
    sort_tm_sqdiff_normed = df_date_merger(df=sort_tm_sqdiff_normed, thr_hold=0.3, columns=[], dataset=dataset)
    sort_tm_sqdiff_normed = sort_tm_sqdiff_normed.drop(columns=['label_current_day', 'rmse'])

    return sort_rmse, sort_tm_sqdiff_normed

'''
' Calcolo l'accuracy per i 10 sample simili al sample campione
' l'accuracy è calcolata per ogni rete (colonna) e ogni epoca (riga)
'''
def gen_csv_accuracy_for_a_day(df_sorted, target_day, epochs=500, nets=30, column='rmse'):
    print("\nGenerating accuracy csv for", column + "...")
    # la colonna target day è il giorno del giorno che sto usando come campione
    # la colonna date_time è il giorno del sample simile
    # label_next_day è la label associata al giorno di date_tim
    date_list_sorted = df_sorted['date_time'].tolist()

    accuracy_df = pd.DataFrame()
    accuracy_df['epoch'] = range(1, epochs + 1)

    for j in range (0, nets):
    #for j in range (0, 2): # debug 
        print(column, "- Running net n°", j)
        predictions = pd.read_csv('D:/PhD-Market-Nets/experiments/' + target_exp + '/predictions/predictions_during_training/test/walk_' + str(target_walk) + '/net_' + str(j) + '.csv')
        predictions = df_date_merger(df=predictions, thr_hold=0.3, columns=[], dataset=dataset)
 
        accuracy_val = []
        for i in range(1, epochs + 1):
            y_true = []
            y_pred = []
            
            for day in date_list_sorted: 
                sel = predictions.loc[predictions['date_time'] == day]
                pred = sel.iloc[0]['epoch_' + str(i)]
                label_next_day = sel.iloc[0][label]

                y_true.append(label_next_day)
                y_pred.append(pred)
                #print("Day:", day, "- Prediction:", pred, "- Label", label_next_day)
            accuracy_val.append(accuracy_score(y_true, y_pred))
            #print("Epoch:", i, " - Accuracy sulle img simili:", accuracy_score(y_true, y_pred))
        
        accuracy_df['net_' + str(j)] = accuracy_val

    #print(accuracy_df)  
    path = 'C:/Users/' + user + '/Desktop/Risultati sample singolo/' + target_exp + '/walk_' +  str(target_walk) + '/' + target_day + '/'
    create_folder(path)
    accuracy_df.to_csv(path + '/' + column + '.csv', index=False, header=True)
    
    return accuracy_df

'''
' Calcolo il file di log con l'accuracy sul campione 
'''
def calculate_champ_acc(df, target_day,nets=30, column='rmse'):
    print("Calculating accuracy for champ with", column, "metric...")

    path = 'C:/Users/' + user + '/Desktop/Risultati sample singolo/' + target_exp + '/walk_' +  str(target_walk) + '/' + target_day + '/'
    thrs = [0.8, 0.7, 0.6, 0.5, 0.4, 0.3]
    
    string = ''
    
    if column == 'rmse':
        # reseto il file di log
        text_file = open(path + "risultati_sample_campione.txt", "w")
        text_file.write(string)
        text_file.close()

        string += "Exp: " + target_exp + "\n" + "Target day: " + target_day + " con label: " + str(target_label) + " e predizione: " + str(target_prediction) + "\n\n"
    string += "Metric: " + column + "\n"

    # RISULTATI PER RMSE
    for thr in thrs:
        #thr = 0.5

        y_true_global = []
        y_pred_global = []
        tot_nets = 0
        #print("Selezionate le epoche la cui accuracy sui sample simili è >=", thr)
        for net in range(0, nets):
        #for net in range(0, 2): #debug
            rows = df.loc[df['net_' + str(net)] >= thr]

            seleced_epochs = rows['epoch'].tolist()
            tot_nets = tot_nets + len(seleced_epochs)

            y_true = []
            y_pred = []

            for epoch in seleced_epochs: 
                #print("epoch:", target_day_df['epoch_' + str(epoch)].iloc[0])
                y_pred.append(target_row_df['epoch_' + str(epoch)].iloc[0])
                y_true.append(target_label)

                y_pred_global.append(target_row_df['epoch_' + str(epoch)].iloc[0])
                y_true_global.append(target_label)


            if len(y_pred) > 0:
                acc = accuracy_score(y_true, y_pred)
                #print("Net", net, "- Accuracy media:", '{:.4f}'.format(acc), "su un totale di", len(y_pred), "epoche")
            #else:
                #print("Net", net, "- Nessuna epoca ha superato la soglia di selezione")
        if len(y_true_global) > 0:
            global_acc = accuracy_score(y_true_global, y_pred_global)
        else:
            global_acc = 0

        string += "THR: " + str(thr) + " - Accuracy media: " + '{:.3f}'.format(global_acc) + " (" + str(tot_nets) +  " reti)\n"

    string += "\n"

    text_file = open(path + "risultati_sample_campione.txt", "a")
    text_file.write(string)
    text_file.close()

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
numero_di_vicini = 20

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

date_time_list.reverse()

for target_day in date_time_list:

    #if target_day == '2018-11-21':
    #    break
    # dataframe contenente la riga del target_day
    target_row_df = predictions_locale.loc[predictions_locale['date_time'] == target_day]
    target_prediction = target_row_df.iloc[0][target_epoch]
    target_label = target_row_df.iloc[0][label]

    print("-------------------------------------------")
    print("Target exp:", target_exp)
    print("Target day:", target_day, "Target Label:", target_label, " -  Prediction:", target_prediction)
    #rint("\nTarget net:", target_net, "- Target walk:", target_walk)
    #print("\nFirst day of the walk:", first_day_available, " - Last day of the walk:", last_day_available)
    
    print("-------------------------------------------")
    #input("Premi per continuare...")
    
    # Prendo i 10 giorni più simili per rmse e l'altra metrica
    sort_rmse, sort_tm_sqdiff_normed = get_df_with_metrics(target_day=target_day, top_number=numero_di_vicini)

    
    t1 = threading.Thread(target=gen_csv_accuracy_for_a_day, args=([sort_rmse, target_day, log['epochs'], log['number_of_nets'], 'rmse']))
    t2 = threading.Thread(target=gen_csv_accuracy_for_a_day, args=([sort_tm_sqdiff_normed, target_day, log['epochs'], log['number_of_nets'], 'tm_sqdiff_normed']))        
    
    t1.start()
    t2.start()

    t1.join()
    t2.join()
    
    
    # sequenziale senza thread
    #rmse_accuracies =  gen_csv_accuracy_for_a_day(df_sorted=sort_rmse, target_day=target_day, epochs=log['epochs'],
    #                                                nets=log['number_of_nets'], column='rmse')

    #tm_sqdiff_normed_accuracies= gen_csv_accuracy_for_a_day(df_sorted=sort_tm_sqdiff_normed, target_day=target_day, epochs=log['epochs'], 
    #                                                nets=log['number_of_nets'], column='tm_sqdiff_normed')


    # per velocizzare il debug
    rmse_accuracies = pd.read_csv('C:/Users/' + user + '/Desktop/Risultati sample singolo/' + target_exp + '/walk_7/' + target_day + '/rmse.csv')
    tm_sqdiff_normed_accuracies = pd.read_csv('C:/Users/' + user + '/Desktop/Risultati sample singolo/' + target_exp + '/walk_7/' + target_day + '/tm_sqdiff_normed.csv')

    # RISULTATI MEDIE ACCURACY CAMPIONE
    calculate_champ_acc(df=rmse_accuracies, target_day=target_day, nets=log['number_of_nets'], column='rmse')

    calculate_champ_acc(df=tm_sqdiff_normed_accuracies, target_day=target_day, nets=log['number_of_nets'], column='tm_sqdiff_normed')