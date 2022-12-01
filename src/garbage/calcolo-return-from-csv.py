import os
import numpy as np 
import pandas as pd 
from os import listdir
from os.path import isfile, join
from classes.Market import Market
import matplotlib.pyplot as plt
import statistics 
from classes.Market import Market
import re
from cycler import cycler

### DA SETTARE ###
#experiment_name = 'exp_BH_walk_DEBUG_CSV'
#experiment_name = 'exp_BH_walk_1_mese_SGD_BS_500_2'
experiment_name = 'exp_BH_walk_1_mese_SGD_BS_500_SP500_thr0.3'
#experiment_name = 'exp_BH_walk_1_mese_SGD_BS_500_SP500CET'

def atoi(text):
    return int(text) if text.isdigit() else text

def natural_keys(text):
    return [ atoi(c) for c in re.split(r'(\d+)', text) ]

### FINE SETTINGS ###
thrs = [0.34, 0.36, 0.38, 0.40, 0.45, 0.50]

thrs = [0.5]
for thr in thrs:

    dataset_path = '/home/unica/PhD-Market-Nets/experiments/'

    validation_input_folder = '/predictions_ensemble_validation_' + str(thr) + '/'
    test_input_folder = '/predictions_ensemble_test_' + str(thr) + '/'

    validation_input_path = dataset_path + experiment_name + validation_input_folder +  'ensemble_magg/' # da cambiare se si vuole usare quello ad eliminazione 
    test_input_path = dataset_path + experiment_name + test_input_folder + 'ensemble_magg/'

    validation_plots_path = dataset_path + experiment_name + validation_input_folder +  'plots/ensemble_magg/'
    test_plots_path = dataset_path + experiment_name + test_input_folder +  'plots/ensemble_magg/'

    if not os.path.isdir(validation_plots_path):
        os.makedirs(validation_plots_path)

    if not os.path.isdir(test_plots_path):
        os.makedirs(test_plots_path)

    # sp500 wrapper
    sp500 = Market(dataset='sp500')
    sp500_label = sp500.get_label_next_day(freq='1d', columns=['open', 'close', 'delta'])
    sp500_label = sp500_label.reset_index()
    sp500_label['date_time'] = sp500_label['date_time'].astype(str)

    walks_list = sorted(os.listdir(validation_input_path))
    walks_list.sort(key=natural_keys)
    # calcolo il return walk per walk e epoca per epoca
    for index_walk, walk in enumerate(walks_list): 

        ################################################# 
        # VALIDATION SET 
        #################################################

        # leggo le predizioni fatte con l'esnemble
        df_ensemble = pd.read_csv(validation_input_path + walk)
        # mergio con le label, così ho un subset del df con le date che mi servono e la predizione 
        df_merge_with_label = pd.merge(df_ensemble, sp500_label, how="inner")

        number_of_epochs = df_ensemble.shape[1] - 1 # conto le epoche - 1 (c'è il date_time)

        return_epochs = np.zeros(number_of_epochs)
        long_prec_epochs = np.zeros(number_of_epochs)
        short_prec_epochs = np.zeros(number_of_epochs)

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
            
            # conto quante long, short e hold azzecco
            long_count = 0
            long_guessed = 0
            short_count = 0
            short_guessed = 0
            hold_count = 0
            hold_guessed = 0

            for i, val in enumerate(y_pred):
                if val == 2.:
                    long_count += 1
                    if delta[i] >= 0:
                        long_guessed += 1
                elif val == 0.:
                    short_count += 1
                    if delta[i] < 0:
                        short_guessed += 1
                #elif val == 1.:
                #    hold_count += 1
                #    if self.delta[i] > -0.2 and self.delta[i] < 0.2:
                #        hold_guessed += 1
                        
            hold_count = len(y_pred) - long_count - short_count

            # percentuale di long e shorts azzeccate
            longs = 0 if long_count == 0 else long_guessed / long_count
            shorts = 0 if short_count == 0 else short_guessed / short_count
            #holds = 0 if hold_count == 0 else hold_guessed / hold_count

            print("LONG_PREC: ", longs)
            print("SHORT_PREC: ", shorts)
            long_prec_epochs[i] = longs
            short_prec_epochs[i] = shorts

            # percentuale di operazioni di long e shorts sul totale di operazioni fatte
            longs_perc = long_count / (len(y_pred))
            shorts_perc = short_count / (len(y_pred))
            holds_perc = hold_count / (len(y_pred))

        
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
        # può servire (??)
        max_value = return_epochs[max_index]

        # salvo il plots
        x = np.arange(0, number_of_epochs)
        #plt.figure(figsize=(15,12))
        #plt.xlabel("Epoch #")
        #plt.ylabel("Dollars")
        #plt.plot(x, return_epochs)
        #plt.plot(x[max_index], return_epochs[max_index], color='red', linestyle='dashed', marker='o')
        #plt.savefig(validation_plots_path + 'walk_' + str(index_walk) + '.png')
        #plt.close('all')
        #plt.show()

        plt.figure(figsize=(15,12))
        
        plt.figtext(0.1, 0.97, "Walk " + str(walk), fontsize='xx-large')
        #plt.figtext(0.1, 0.96, "Train set: " + str(self.__training_set[walk][0]) + " - " + str(self.__training_set[walk][1]))
        #plt.figtext(0.1, 0.94, "Valid set: " + str(self.__validation_set[walk][0]) + " - " + str(self.__validation_set[walk][1]))
        #plt.figtext(0.1, 0.92, "Epoche: " + str(self.__epochs) + " - Numero reti: " + str(self.__number_of_nets) + " - Batch Size: " + str(self.__bs))

        # plot the training loss and accuracy
        plt.style.use("ggplot")
        plt.rcParams['axes.prop_cycle'] = cycler(color='bgrcmyk')
        plt.subplot(2, 1, 1)
        plt.xlabel("Epoch #")
        plt.ylabel("Dollars")
        
        plt.plot(x, return_epochs)
        plt.plot(x[max_index], return_epochs[max_index], color='red', linestyle='dashed', marker='o')
        plt.plot(x, np.zeros(number_of_epochs), color="black")
        
        plt.subplot(2, 1, 2)
        plt.xlabel("Epoch #")
        plt.ylabel("Per-class val. precision")

        
        plt.plot(x, long_prec_epochs, label="Long precision")
        plt.plot(x, short_prec_epochs, label="Short precision")
        #plt.plot(np.arange(0, self.__epochs), hold_val_acc, label="Hold precision")
        plt.plot(x, np.full((number_of_epochs, 1), 0.5), color="black")
        plt.legend(loc="upper left")
        

        '''
        plt.subplot(2, 2, 3)
        plt.xlabel("Epoch #")
        plt.ylabel("Global accuracy")

        plt.plot(np.arange(0, self.__epochs), acc, label="Training accuracy")
        #plt.plot(np.arange(0, self.__epochs), H.history["accuracy"], label="train_acc")
        plt.plot(np.arange(0, self.__epochs), val_acc, label="Validation accuracy")
        #plt.title("Training Loss")
        plt.legend(loc="upper left")

        plt.subplot(2, 2, 4)
        #plt.xlabel("Epoch #")
        #plt.ylabel("Global loss")
        #plt.plot(np.arange(0, self.__epochs), loss, label="Training loss")
        #plt.plot(np.arange(0, self.__epochs), val_loss, label="Validation loss")
        plt.xlabel("Epoch #")
        plt.ylabel("Perc Operations")
        plt.plot(np.arange(0, self.__epochs),long_perc, label='% of long operations')
        plt.plot(np.arange(0, self.__epochs), short_perc, label="% of short operations")
        plt.plot(np.arange(0, self.__epochs), hold_perc, label="% of hold operations")
        #plt.title("Training Loss")
        plt.legend(loc="upper left")
        '''
        plt.savefig(validation_plots_path + 'walk_' + str(index_walk) + '.png')
        plt.close('all')


        '''
        ################################################# 
        # TEST SET 
        ################################################# 

        # leggo le predizioni fatte con l'esnemble
        df_ensemble = pd.read_csv(test_input_path + walk)
        df_ensemble['date_time'] = pd.to_datetime(df_ensemble['date_time'])
        
        start_date = pd.to_datetime(df_ensemble.iloc[0].date_time)
        end_date = (start_date + pd.DateOffset(months=1))

        df_ensemble = Market.get_df_by_data_range(df=df_ensemble, start_date=start_date, end_date=end_date)

        #torno alla stringa per fare il merge
        df_ensemble['date_time'] = df_ensemble['date_time'].dt.date
        df_ensemble['date_time'] = df_ensemble['date_time'].apply(str)

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


        # salvo il plots
        x = np.arange(0, number_of_epochs)
        plt.figure(figsize=(15,12))
        plt.xlabel("Epoch #")
        plt.ylabel("Dollars")
        plt.plot(x, return_epochs)
        plt.plot(x[max_index], return_epochs[max_index], color='red', linestyle='dashed', marker='o')
        plt.savefig(test_plots_path + 'walk_' + str(index_walk) + '.png')
        plt.close('all')
        #plt.show()


        return_selezionato = return_epochs[max_index]
        y_pred = df_merge_with_label['epoch_' + str(max_index)].tolist()

        short_totali = y_pred.count(0)
        hold_totali = y_pred.count(1)
        long_totali = y_pred.count(2)

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

        print("Walk n° ", walk)
        print("Return in test set: ", return_selezionato)
        
        print("Long in test set: ", long_totali, " | Precision delle longs: ", longs_precision * 100, " | Long in %: ", long_totali / len(y_pred) * 100 )
        print("Short in test set: ", short_totali, " | Precision delle short: ", shorts_precision * 100, " | Short in % : ", short_totali / len(y_pred) * 100 )
        print("Hold in test set: ", hold_totali, " | Hold in % : ", hold_totali / len(y_pred) * 100 )

        print("Long + short: ", short_totali + long_totali)   
        print("Long + short + hold: ", len(y_pred))   
        print("\n")

        f=open(test_plots_path + 'return_test_set.txt','a')
        f.write("Walk n°: " + str(walk) + "\n")
        f.write("Return in test set: " + str(return_selezionato) + "\n")
        f.write("Long in test set: " + str(long_totali)  + " | Precision delle longs: " + str(longs_precision * 100) + " | Long in %: " + str(long_totali / len(y_pred) * 100) + "\n")
        f.write("Short in test set: " + str(short_totali)  + " | Precision delle short: " + str(shorts_precision * 100) + " | Short in %: " + str(short_totali / len(y_pred) * 100) + "\n")
        f.write("Hold in test set: " + str(hold_totali)  + " | Hold in %: " + str(hold_totali / len(y_pred) * 100) + "\n")

        f.write("Long + short:  " + str(short_totali + long_totali) + "\n")
        f.write("Long + short + hold:  " + str(len(y_pred)) + "\n")
        f.write("\n")
        f.close()
        '''