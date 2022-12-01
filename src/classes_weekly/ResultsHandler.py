import os
import numpy as np
import pandas as pd
from datetime import timedelta
from os import listdir
from os.path import isfile, join
from classes.Market import Market
from classes.Measures import Measures
from classes.Utils import natural_keys, df_date_merger, do_plot
import statistics
import time
import threading
import xlsxwriter
import matplotlib
import matplotlib.pyplot as plt
import statistics
import json 
from classes.Utils import create_folder
import platform




class ResultsHandler:

    platform = platform.platform()

    experiment_name = ''

    #experiment_original_path = 'C:/Users/Utente/Documents/GitHub/PhD-Market-Nets/experiments/Esperimenti Vacanze/'
    #experiment_original_path = 'C:/Users/Utente/Documents/GitHub/PhD-Market-Nets/experiments/'  # locale
    
    if platform == 'Linux-4.15.0-45-generic-x86_64-with-Ubuntu-16.04-xenial': 
        experiment_original_path = '/media/unica/HDD 9TB Raid0 - 1/experiments/'
    else: 
        experiment_original_path = '../experiments/'  # locale


    prediction_base_folder = ''
    original_predictions_validation_folder = ''
    original_predictions_test_folder = ''
    ensemble_base_folder = ''
    final_decision_folder = ''

    dataset = ''

    walks_list = []
    nets_list = []

    def __init__(self, experiment_name):
        self.experiment_name = experiment_name
        self.prediction_base_folder = self.experiment_original_path + experiment_name + '/predictions/'

        self.original_predictions_validation_folder = self.experiment_original_path + experiment_name + '/predictions/predictions_during_training/validation/'
        self.original_predictions_test_folder = self.experiment_original_path + experiment_name + '/predictions/predictions_during_training/test/'
        self.ensemble_base_folder = self.experiment_original_path + experiment_name + '/predictions/predictions_post_ensemble/'
        self.final_decision_folder = self.experiment_original_path + experiment_name + '/predictions/final_decisions/'

        with open(self.experiment_original_path + experiment_name + '/log.json') as json_file:
            self.iperparameters = json.load(json_file)

        '''
        print(self.iperparameters)
        input()

        print(self.iperparameters['predictions_dataset'])
        input()
        '''

        self.dataset = self.iperparameters['predictions_dataset']

        # Prendo la lista degli walk e il numero di reti utilizzate leggendo i file
        self.walks_list = os.listdir(self.original_predictions_validation_folder)
        self.nets_list = os.listdir(self.original_predictions_validation_folder + self.walks_list[0])

        self.walks_list.sort(key=natural_keys)
        self.nets_list.sort(key=natural_keys)

    '''
    █▀▀ █▀▀▄ █▀▀ █▀▀ █▀▄▀█ █▀▀▄ █░░ █▀▀
    █▀▀ █░░█ ▀▀█ █▀▀ █░▀░█ █▀▀▄ █░░ █▀▀
    ▀▀▀ ▀░░▀ ▀▀▀ ▀▀▀ ▀░░░▀ ▀▀▀░ ▀▀▀ ▀▀▀
    '''

    '''
    ' Questo metodo elimina tutte le colonne che hanno tutto 0 o tutto 1
    ' L'assunzione di base è che quella specifica rete non abbia imparato nulla. 
    ' Restituisce il df risultante, senza le colonne delle reti inutili
    '''
    def get_useless_epoch(self, df):
        nunique = df.apply(pd.Series.nunique)
        cols_to_drop = nunique[nunique == 1].index
        return cols_to_drop

    '''
    ' Restituisco per un csv preso in input la
    ' lista delle date presenti in quel file (un walk)
    ' ed il numero di epoche
    '''
    def get_date_epochs_walk(self, path, walk):
        date_list = []
        epochs_list = []

        full_path = path + walk + '/'

        for filename in os.listdir(full_path):
            df = pd.read_csv(full_path + filename)
            date_list = df['date_time'].tolist()
            epochs_list = df.columns.values
            # rimuovo il primo ed il secondo elemento che sono rispettivamente col0, e date_time
            epochs_list = np.delete(epochs_list, [0, 1])
            break

        return np.array(date_list), np.array(epochs_list)

    '''
    ' Calcola l'ensemable sulle colonne (reti) con una % di agreement
    ' Inserendo nel calcolo della percentuale anche il numero di hold
    '''
    def generate_triple(self, df):
        m = pd.DataFrame()
        m['ensemble'] = df.eq(0).sum(1).astype(str) + ';' + df.eq(1).sum(1).astype(str) + ';' + df.eq(2).sum(1).astype(str) 
        
        return m

    '''
    ' Calcola l'ensemable sulle colonne (reti) con una % di agreement
    ' Inserendo nel calcolo della percentuale anche il numero di hold
    '''
    def ensemble_magg(self, df, thr):
        
        for column in df:            
            n_short  = ((df[column].str.split(';').str[0]).astype(int) / len(self.nets_list)).gt(thr)
            n_hold  = ((df[column].str.split(';').str[1]).astype(int) / len(self.nets_list)).gt(thr)
            n_long  = ((df[column].str.split(';').str[2]).astype(int) / len(self.nets_list)).gt(thr)


            #df_global['decision'] = df_global['decision'].apply(lambda x: 2 if x > jolly_number else 1)
            # n_long.astype(int).mul(2).add(n_short)

            # DECOMMENTARE QUESTA RIGA PER ESCLUDERE LE HOLD DAL CALCOLO
            #m = pd.DataFrame(np.select([n_short, n_long], [0, 2], 1), index=df.index, columns=['ensemble'])
            df[column] = np.select([n_short, n_hold, n_long], [0, 1, 2], 1)

        return df

    '''
    ' Calcolo l'ensemble ad eliminazione
    ' Se il numero è positivo sarà una Long, altrimenti sarà una short
    ' Es: 10 long, 2 short = 8
    ' Es: 10 short, 3 long = -7
    ' Es: 10 long, 10 short = 0, hold
    '''
    def ensemble_elimination(self, df, thr):
        jolly_number = int((len(self.nets_list) * (thr * 100) / 100))
        
        for column in df:          
            df[column]  = df[column].apply(lambda x: 2 if (int(x.split(';')[2]) - int(x.split(';')[0])) > jolly_number else 1)

        return df
    '''
    '
    '''
    def ensemble_exclusive(self, df, thr):
        jolly_number = int((len(self.nets_list) * (thr * 100) / 100))

        for column in df:          
            df[column]  = df[column].apply(lambda x: 2 if int(x.split(';')[2]) > jolly_number else 1)

        return df

    '''
    ' Ensamble che forzi le hold a un 30-40 % di coverage in tutti e 3 i casi:  
    ' Per ogni giorno entro se: A & B sono vere: 
    ' a) % hold sotto una certa soglia per quel gg (va deciso il parametro) 
    ' b) num reti che fanno long > 2* num reti che fanno short.
    '''
    def offset_thr_ensemble(self, df, hold_thr=0.3):
        m1 = df.eq(1).mean(1) < hold_thr  # hold sotto una certa soglia
        m2 = df.eq(2).sum(1) > 2 * df.eq(1).sum(1)  # long > 2 * short
        m3 = df.eq(0).sum(1) > df.eq(2).sum(1)  # short > long

        m = pd.DataFrame(np.select([m1 & m2, m1 & m3], [
                         2, 0], default=1), index=df.index, columns=['ensemble'])

        return m

    '''
    ' Calcola gli ensemble a maggioranza
    '''
    def calculate_ensemble_magg(self, thrs, type, remove_nets=False):
        remove_nets_str = 'con-rimozione-reti/'

        if remove_nets == False:
            remove_nets_str = 'senza-rimozione-reti/'

        # A seconda del set creo input e output path
        if type == 'Validation':
            input_path = self.original_predictions_validation_folder
            output_path = self.ensemble_base_folder + remove_nets_str + 'validation/'

        if type == 'Test':
            input_path = self.original_predictions_test_folder
            output_path = self.ensemble_base_folder + remove_nets_str + 'test/'

        for thr in thrs:
            if not os.path.isdir(output_path + 'ensemble_magg/' + str(thr) + '/'):
                os.makedirs(output_path + 'ensemble_magg/' + str(thr) + '/')
        
            # per ogni walk
            for walk in self.walks_list:
                start_walk_time = time.time()
                date_list, epochs_list = self.get_date_epochs_walk(path=input_path, walk=walk)

                # Qui salvo il risultato dell'ensemble di ogni rete per ogni epoca
                df_ensemble_magg = pd.DataFrame(columns=[epochs_list])
                df_ensemble_magg['date_time'] = date_list

                reti = []
                new_nets_list = []

                # leggo le reti e scarto quelle che hanno un numero di epoche "rotte" sopra il 60%
                for net in range(0, len(self.nets_list)):
                    rete = pd.read_csv(input_path + walk + '/' + self.nets_list[net], engine='c', na_filter=False)

                    if remove_nets is True:
                        dummy_epoch = self.get_useless_epoch(rete)
                        count_of_dummy = dummy_epoch.shape[0] * 100 / rete.shape[1]

                        if count_of_dummy < 60.0:
                            # Leggo per ogni rete le decisioni, e per la specifica epoca le inserisco nel DF delle predizioni delle reti
                            reti.append(rete)
                            new_nets_list.append(self.nets_list[net])
                    else:
                        reti.append(rete)
                        new_nets_list.append(self.nets_list[net])

                nets_list = new_nets_list

                # Qui inserisco la predizione di ogni rete giorno per giorno
                df_net = pd.DataFrame(columns=[nets_list])
                df_net['date_time'] = date_list

                # per ogni epoca
                for idx, epoch in enumerate(epochs_list):
                    start = time.time()

                    # per ogni rete
                    for i, net in enumerate(nets_list):
                        # Leggo per ogni rete le decisioni, e per la specifica epoca le inserisco nel DF delle predizioni delle reti
                        df_predizioni = reti[i]
                        df_net[net] = df_predizioni[epoch]

                    ensemble_magg = self.ensemble_magg(df=df_net)

                    df_ensemble_magg[epoch] = ensemble_magg['ensemble']
                    
                    end = time.time()
                    print(self.experiment_name + " | Ensemble magg | " + type + " | " + walk + " | epoca: " + str(idx + 1) + " | THR: " + str(thr) + " | ETA: " + "{:.3f}".format(end-start))

                df_ensemble_magg.to_csv(output_path + 'ensemble_magg/' + str(thr) + '/' + walk + '.csv', header=True, index=False)
                end_walk_time = time.time()
                print(self.experiment_name + " | Ensemble magg | " + type + " | THR: | " + str(thr) + " | Total walk elapsed time:" + "{:.3f}".format(end_walk_time - start_walk_time))

    '''
    ' Calcola in contemporanea gli ensemble ad 
    ' eliminazione, exclusive ed offset
    '''
    def calculate_ensembles(self, type, remove_nets=False):
        remove_nets_str = 'con-rimozione-reti/'

        if remove_nets == False:
            remove_nets_str = 'senza-rimozione-reti/'

        # A seconda del set creo input e output path
        if type == 'Validation':
            input_path = self.original_predictions_validation_folder
            output_path = self.ensemble_base_folder + remove_nets_str + 'validation/'

        if type == 'Test':
            input_path = self.original_predictions_test_folder
            output_path = self.ensemble_base_folder + remove_nets_str + 'test/'

        if not os.path.isdir(output_path + 'ensemble_el/'):
            os.makedirs(output_path + 'ensemble_el/')

        if not os.path.isdir(output_path + 'ensemble_offset/'):
            os.makedirs(output_path + 'ensemble_offset/')

        if not os.path.isdir(output_path + 'ensemble_exclusive/'):
            os.makedirs(output_path + 'ensemble_exclusive/')

        # per ogni walk
        for walk in self.walks_list:
            start_walk_time = time.time()
            date_list, epochs_list = self.get_date_epochs_walk(path=input_path, walk=walk)

            # Qui salvo il risultato dell'ensemble di ogni rete per ogni epoca
            df_ensemble_el = pd.DataFrame(columns=[epochs_list])
            df_ensemble_el['date_time'] = date_list

            df_ensemble_exclusive = pd.DataFrame(columns=[epochs_list])
            df_ensemble_exclusive['date_time'] = date_list

            df_ensemble_offset = pd.DataFrame(columns=[epochs_list])
            df_ensemble_offset['date_time'] = date_list

            reti = []
            new_nets_list = []

            # leggo le reti e scarto quelle che hanno un numero di epoche "rotte" sopra il 60%
            for net in range(0, len(self.nets_list)):
                rete = pd.read_csv(input_path + walk + '/' + self.nets_list[net], engine='c', na_filter=False)

                if remove_nets is True:
                    dummy_epoch = self.get_useless_epoch(rete)
                    count_of_dummy = dummy_epoch.shape[0] * 100 / rete.shape[1]

                    if count_of_dummy < 60.0:
                        # Leggo per ogni rete le decisioni, e per la specifica epoca le inserisco nel DF delle predizioni delle reti
                        reti.append(rete)
                        new_nets_list.append(self.nets_list[net])
                else:
                    reti.append(rete)
                    new_nets_list.append(self.nets_list[net])

            nets_list = new_nets_list

            # Qui inserisco la predizione di ogni rete giorno per giorno
            df_net = pd.DataFrame(columns=[nets_list])
            df_net['date_time'] = date_list

            # per ogni epoca
            for idx, epoch in enumerate(epochs_list):

                start = time.time()

                # per ogni rete
                for i, net in enumerate(nets_list):
                    # Leggo per ogni rete le decisioni, e per la specifica epoca le inserisco nel DF delle predizioni delle reti
                    df_predizioni = reti[i]
                    df_net[net] = df_predizioni[epoch]

                ensemble_el = self.elimination_ensemble(df=df_net)
                ensemble_exclusive = self.elimination_ensemble_exclusive(df=df_net)

                hold_thr = 0.2
                ensemble_offset = self.offset_thr_ensemble(df=df_net, hold_thr=hold_thr)

                df_ensemble_el[epoch] = ensemble_el['ensemble']
                df_ensemble_offset[epoch] = ensemble_offset['ensemble']
                df_ensemble_exclusive[epoch] = ensemble_exclusive['ensemble']

                end = time.time()
                print(self.experiment_name + " | Ensembles: " + type + " | " + walk + " | epoca: " + str(idx + 1) + " | ETA: " + "{:.3f}".format(end-start))

            df_ensemble_el.to_csv(output_path + 'ensemble_el/' + walk + '.csv', header=True, index=False)
            df_ensemble_offset.to_csv(output_path + 'ensemble_offset/' + walk + '.csv', header=True, index=False)
            df_ensemble_exclusive.to_csv(output_path + 'ensemble_exclusive/' + walk + '.csv', header=True, index=False)

            end_walk_time = time.time()
            print(self.experiment_name + " | Ensembles: " + type + "| Total walk elapsed time:" + "{:.3f}".format(end_walk_time - start_walk_time))

    '''
    ' Calcola per ogni soglia gli ensemble a maggioranza
    ' eliminazione ed exclusive
    '''
    def run_ensemble(self, type, thrs_ensemble_magg=[], thrs_ensemble_exclusive=[], thrs_ensemble_elimination=[], remove_nets=False): 
        remove_nets_str = 'con-rimozione-reti/'

        if remove_nets == False:
            remove_nets_str = 'senza-rimozione-reti/'

        if type == 'validation':
            output_path = self.ensemble_base_folder + remove_nets_str + 'validation/'

        if type == 'test':
            output_path = self.ensemble_base_folder + remove_nets_str + 'test/'

        for walk in self.walks_list:
            df_walk = pd.read_csv(self.prediction_base_folder + 'triple_csv/' + type + '/' + walk + '.csv')  

            date_time = df_walk['date_time'].tolist()
            df_walk = df_walk.drop(columns=['date_time']) 

            for thr in thrs_ensemble_magg:
                print("Running ensemble magg | Walk:", walk, " | thr:", thr, "| type:", type)
                create_folder(output_path + 'ensemble_magg/' + str(thr) + '/')
                df_ensemble_magg = self.ensemble_magg(df=df_walk.copy(), thr=thr)
                df_ensemble_magg['date_time'] = date_time
                df_ensemble_magg.to_csv(output_path + 'ensemble_magg/' + str(thr) + '/' + walk + '.csv', header=True, index=False)

            for thr in thrs_ensemble_exclusive:
                print("Running ensemble exclusive | Walk:", walk, " | thr:", thr, "| type:", type)
                create_folder(output_path + 'ensemble_exclusive/' + str(thr) + '/')
                df_ensemble_exclusive = self.ensemble_exclusive(df=df_walk.copy(), thr=thr)
                df_ensemble_exclusive['date_time'] = date_time
                df_ensemble_exclusive.to_csv(output_path + 'ensemble_exclusive/' + str(thr) + '/' + walk + '.csv', header=True, index=False)
                
            for thr in thrs_ensemble_elimination:
                print("Running ensemble elimination | Walk:", walk, " | thr:", thr, "| type:", type)
                create_folder(output_path + 'ensemble_el/' + str(thr) + '/')
                df_ensemble_elimination = self.ensemble_elimination(df=df_walk.copy(), thr=thr)
                df_ensemble_elimination['date_time'] = date_time
                df_ensemble_elimination.to_csv(output_path + 'ensemble_el/' + str(thr) + '/' + walk + '.csv', header=True, index=False)

    '''
    ' Calcola tutti gli ensemble su due thread separati
    '''
    def generate_ensemble(self, thrs_ensemble_magg=[], thrs_ensemble_exclusive=[], thrs_ensemble_elimination=[], remove_nets=False):
        t1 = threading.Thread(target=self.run_ensemble, args=(['validation', thrs_ensemble_magg, thrs_ensemble_exclusive, thrs_ensemble_elimination, remove_nets]))
        t2 = threading.Thread(target=self.run_ensemble, args=(['test', thrs_ensemble_magg, thrs_ensemble_exclusive, thrs_ensemble_elimination, remove_nets]))        
        
        t1.start()
        t2.start()

        t1.join()
        t2.join()


    '''
    ' lancio due thread in parallelo per validation e test set
    '''
    def generate_triple_csv(self, remove_nets=False):
        
        t1 = threading.Thread(target=self.calculate_triple_csv, args=(['validation']))
        t2 = threading.Thread(target=self.calculate_triple_csv, args=(['test']))        
        
        t1.start()
        t2.start()

        t1.join()
        t2.join()
    
    '''
    ' Nuovo ensemble che genera un csv di triple per ogni epoca / giorno
    '''
    def calculate_triple_csv(self, type):
        # A seconda del set creo input e output path
        if type == 'validation':
            input_path = self.original_predictions_validation_folder
            output_path = self.prediction_base_folder + 'triple_csv/validation/'

        if type == 'test':
            input_path = self.original_predictions_test_folder
            output_path = self.prediction_base_folder + 'triple_csv/test/'

        if not os.path.isdir(output_path):
            os.makedirs(output_path)

        # per ogni walk
        for walk in self.walks_list:
            start_walk_time = time.time()
            date_list, epochs_list = self.get_date_epochs_walk(path=input_path, walk=walk)

            # Qui salvo il risultato dell'ensemble di ogni rete per ogni epoca
            df_ensemble_magg = pd.DataFrame(columns=[epochs_list])
            df_ensemble_magg['date_time'] = date_list

            nets = []
            # leggo le reti e scarto quelle che hanno un numero di epoche "rotte" sopra il 60%
            for net in range(0, len(self.nets_list)):
                nets.append(pd.read_csv(input_path + walk + '/' + self.nets_list[net], engine='c', na_filter=False))

            # Qui inserisco la predizione di ogni rete giorno per giorno
            df_net = pd.DataFrame(columns=[self.nets_list])
            df_net['date_time'] = date_list

            # per ogni epoca
            for idx, epoch in enumerate(epochs_list):
                start = time.time()

                # per ogni rete
                for i, net in enumerate(self.nets_list):
                    # Leggo per ogni rete le decisioni, e per la specifica epoca le inserisco nel DF delle predizioni delle reti
                    df_predizioni = nets[i]
                    df_net[net] = df_predizioni[epoch]

                ensemble_magg = self.generate_triple(df=df_net)

                df_ensemble_magg[epoch] = ensemble_magg['ensemble']
                
                end = time.time()
                print(self.experiment_name + " | CSV Triple | " + type + " | " + walk + " | epoca: " + str(idx + 1) + " | ETA: " + "{:.3f}".format(end-start))

            df_ensemble_magg.to_csv(output_path + walk + '.csv', header=True, index=False)

            end_walk_time = time.time()
            print(self.experiment_name + " | CSV Triple | " + type + " | Total walk elapsed time:" + "{:.3f}".format(end_walk_time - start_walk_time))


    '''
    █▀▀█ █▀▀ ▀▀█▀▀ █░░█ █▀▀█ █▀▀▄ █▀▀   ░ ░   █▀▄▀█ █▀▀▄ █▀▀▄   ░ ░   █▀▀█ █▀▀█ █▀▄▀█ █▀▀█ █▀▀▄
    █▄▄▀ █▀▀ ░░█░░ █░░█ █▄▄▀ █░░█ ▀▀█   ▀ ▀   █░▀░█ █░░█ █░░█   ▀ ▀   █▄▄▀ █░░█ █░▀░█ █▄▄█ █░░█
    ▀░▀▀ ▀▀▀ ░░▀░░ ░▀▀▀ ▀░▀▀ ▀░░▀ ▀▀▀   ░ ░   ▀░░░▀ ▀▀▀░ ▀▀▀░   ░ ░   ▀░▀▀ ▀▀▀▀ ▀░░░▀ ▀░░▀ ▀▀▀░
    '''

    '''
    ' Genero i file di decisioni finali per i vari ensemble
    ' type = [ensemble_magg, ensemble_el_longonly, ensemble_exclusive]
    '
    ' ensemble_magg'
    ' get_final_decision_from_ensemble(type='ensemble_magg', ensemble_thr=0.35, remove_nets=False)
    '
    '
    ' ensemble_el:  
    ' get_final_decision_from_ensemble(type='ensemble_el', perc_agreement=0.4, remove_nets=False, validation_thr=15)
    '
    ' Calcolo le decisioni finali sull'ensemble el, facendo il long only usando la % 
    ' Viene calcolato sull'ensemble generato da elimination_ensemble (quindi facendo +1 long -1 short)
    '
    ' ensemble_exclusive
    ' get_final_decision_from_ensemble(type='ensemble_exclusive', num_agreement=10, remove_nets=False, validation_thr=15)
    '
    ' Calcolo le decisioni finali sull'ensemble el, facendo il long only usando la % 
    ' Viene calcolato sull'ensemble generato da elimination_ensemble_longonly 
    '(quindi facendo +1 long e basta)
    ' # TOLTO num_agreement
    '''
    def get_final_decision_from_ensemble(self, type='ensemble_magg', validation_thr=15, validation_metric='romad', epoch_selection_policy='long_short', 
        thr_ensemble_magg=0.35, thr_ensemble_exclusive=0.3, thr_ensemble_elimination=0.3, remove_nets=False, stop_loss=1000, penalty=32):
        
        remove_nets_str = 'con-rimozione-reti/'

        if remove_nets == False:
            remove_nets_str = 'senza-rimozione-reti/'

        if type is 'ensemble_magg':
            print("Generating final decision | type:", type, "| thr:", thr_ensemble_magg, "| Epoch selection policy:", epoch_selection_policy)
            output_path = self.final_decision_folder + 'ensemble_magg/' + remove_nets_str + epoch_selection_policy + '/'
            validation_input_path = self.ensemble_base_folder + remove_nets_str + 'validation/ensemble_magg/' + str(thr_ensemble_magg) + '/'
            test_input_path = self.ensemble_base_folder  + remove_nets_str + 'test/ensemble_magg/' + str(thr_ensemble_magg) + '/'


        if type is 'ensemble_el_long_only':
            print("Generating final decision | type:", type, "| thr:", thr_ensemble_elimination, "| Epoch selection policy:", epoch_selection_policy)
            output_path = self.final_decision_folder + 'ensemble_el_longonly/' + remove_nets_str + epoch_selection_policy + '/'
            validation_input_path = self.ensemble_base_folder + remove_nets_str + 'validation/ensemble_el/' #+ str(thr_ensemble_elimination) + '/'
            test_input_path = self.ensemble_base_folder + remove_nets_str + 'test/ensemble_el/' #+ str(thr_ensemble_elimination) + '/'

        if type is 'ensemble_exclusive': 
            print("Generating final decision | type:", type, "| thr:", thr_ensemble_exclusive, "| Epoch selection policy:", epoch_selection_policy)
            output_path = self.final_decision_folder + 'ensemble_exclusive/' + remove_nets_str + epoch_selection_policy + '/'
            validation_input_path = self.ensemble_base_folder + remove_nets_str + 'validation/ensemble_exclusive/' + str(thr_ensemble_exclusive) + '/'
            test_input_path = self.ensemble_base_folder + remove_nets_str + 'test/ensemble_exclusive/' + str(thr_ensemble_exclusive) + '/'
        
        # creo la path finale     
        if not os.path.isdir(output_path):
            os.makedirs(output_path)

        df_global = pd.DataFrame(columns=['date_time', 'close', 'delta_current_day', 'delta_next_day', 'label'])

        dataset = Market(dataset=self.dataset)
        dataset_label = dataset.get_label(freq='1d', columns=['open', 'close', 'delta_current_day', 'delta_next_day', 'high', 'low'], thr=self.iperparameters['hold_labeling'])
        dataset_label = dataset_label.reset_index()
        dataset_label['date_time'] = dataset_label['date_time'].astype(str)

        for index_walk, walk in enumerate(self.walks_list):
            df_ensemble_val = pd.read_csv(validation_input_path + walk + '.csv')
            df_merge_with_label = pd.merge(df_ensemble_val, dataset_label, how="inner")
            
            val_idx, romad, return_value, mdd_value = self.get_max_idx_from_validation(df_merge_with_label=df_merge_with_label, validation_thr=validation_thr, metric='romad', epoch_selection_policy=epoch_selection_policy, stop_loss=stop_loss, penalty=penalty)
            
            #DEBUG IS1 E IS2
            #val_idx = 400 # DEBUG
            #df_calcoli_debug = df_merge_with_label[['epoch_' + str(val_idx), 'date_time', 'delta_next_day', 'low', 'open', 'close']]
            #df_calcoli_debug = df_calcoli_debug.rename(columns={'epoch_' + str(val_idx) : 'decision'})
            #vlong, vshort, vhold, vgeneral = Measures.get_precision_count_coverage(df=df_calcoli_debug, multiplier=50, delta_to_use='delta_next_day', stop_loss=1000, penalty=25)
            #print("Walk n° " +  str(index_walk) + " - ID validation scelto: " + str(val_idx))
            #input()

            df_test = pd.read_csv(test_input_path + walk + '.csv')

            # mergio con le label, così ho un subset del df con le date che mi servono e la predizione
            df_test = pd.merge(df_test, dataset_label, how="inner")
            #df_merge_with_label = df_merge_with_label.set_index('index')
            df_test = df_test[['epoch_' + str(val_idx), 'open', 'close', 'delta_current_day', 'delta_next_day', 'date_time', 'high', 'low']]
            df_test = df_test.rename(columns={"epoch_" + str(val_idx): "decision"})

            df_global = pd.concat([df_global, df_test], sort=True)

            #tlong, tshort, thold, tgeneral = Measures.get_precision_count_coverage(df=df_test, multiplier=50, delta_to_use='delta_next_day', stop_loss=1000, penalty=25)
            #print("Walk n°", index_walk, " Epoca migliore selezionata: ", val_idx, " | Soglia Ensemble Exclusive:", thr_ensemble_exclusive, "Valid Long Coverage:", vlong['coverage'], " Test Long Coverage:", tlong['coverage'])

        df_global = df_global.drop_duplicates(subset='date_time', keep="first")

        df_global['date_time'] = df_global['date_time'].shift(-1)
        #df_global = df_global.drop(columns=['close', 'delta'], axis=1)
        df_global = df_global[['date_time', 'decision']]
        df_global = df_global.drop(df_global.index[0])

        df_global = df_date_merger(df=df_global, columns=['open', 'close', 'high', 'low', 'delta_current_day', 'delta_next_day'], dataset=self.iperparameters['predictions_dataset'], thr_hold=self.iperparameters['hold_labeling'])
        df_global['decision'] = df_global['decision'].astype(int)

        #ttlong, ttshort, tthold, ttgeneral = Measures.get_precision_count_coverage(df=df_global, multiplier=50, delta_to_use='delta_next_day', stop_loss=1000, penalty=25)
        #print("% OPERAZIONI SUL TEST SET TOTALE: ", ttlong['coverage'])

        number_of_nets = len(self.nets_list)

        if type is 'ensemble_el_longonly':
            df_global.to_csv(output_path + 'decisions_ensemble_el_long_only_' + str(thr_ensemble_elimination) + '.csv', header=True, index=False)

        if type is 'ensemble_exclusive': 

            df_global.to_csv(output_path + 'decisions_ensemble_exclusive_' + str(thr_ensemble_exclusive) + '.csv', header=True, index=False)

        if type is 'ensemble_magg': 
            df_global.to_csv(output_path + 'decisions_ensemble_magg_' + str(thr_ensemble_magg) + '.csv', header=True, index=False)


    '''
    ' Scelgo l'epoca migliore dal validation utilizzando come metrica o il return, 
    ' il romad oppure l'mdd
    ' return: epoch_index, romad_self.iperparameters['epochs'][epoch_index], return_self.iperparameters['epochs'][epoch_index], mdd_self.iperparameters['epochs'][epoch_index]
    '''
    def get_max_idx_from_validation(self, df_merge_with_label, validation_thr, epoch_selection_policy='long_short', metric='return', stop_loss=1000, penalty=32):
        # mergio con le label, così ho un subset del df con le date che mi servono e la predizione
        #df_merge_with_label = pd.merge(net, dataset_label, how="inner")

        # conto le epoche - 1 (c'è il date_time e l'id)
        number_of_epochs = self.iperparameters['epochs']

        romad_epochs = np.full(number_of_epochs, -100.0)
        return_epochs = np.full(number_of_epochs, -1000000.0)
        mdd_epochs = np.full(number_of_epochs, 1000000.0)

        # calcolo il return per un epoca, lascio volontariamente il primo elemento a 0
        for i in range(1, number_of_epochs + 1):
            df_epoch_rename = df_merge_with_label.copy()
            df_epoch_rename = df_epoch_rename.rename(columns={'epoch_' + str(i): 'decision'})

            equity_line, global_return, mdd, romad, ii, j = Measures.get_equity_return_mdd_romad(df=df_epoch_rename, multiplier=self.iperparameters['return_multiplier'], 
                                penalty=penalty, stop_loss=stop_loss, type=epoch_selection_policy, delta_to_use='delta_next_day')
            #print("Epoca N° ", i, " Romad: ", romad)
            #print("Epoca N°", i, " - Return: ", global_return, " - Romad: ", romad)
            romad_epochs[i-1] = romad
            return_epochs[i-1] = global_return
            mdd_epochs[i-1] = mdd

        # Ritorno medio con l'intorno
        
        array_values = []  # la uso per salvare la metrica scelta

        if metric is 'return':
            mean_values = np.full(number_of_epochs, -100.0)
            array_values = return_epochs
        if metric is 'romad':
            mean_values = np.full(number_of_epochs, -1000000.0)
            array_values = romad_epochs
        if metric is 'mdd':
            mean_values = np.full(number_of_epochs, 1000000.0)
            array_values = mdd_epochs

        for index_for, value in enumerate(array_values):
            # entro solo tra i valori compresi dalla soglia, se soglia = 15, parto dal 15° sino a i-15
            if index_for >= validation_thr and index_for <= (number_of_epochs - validation_thr):
                mean_values[index_for] = statistics.mean(array_values[index_for-validation_thr:index_for+validation_thr])
                #print("mean_values id", index_for, "valore: ", mean_values[index_for])

        # seleziono l'epoca con il return migliore a cui aggiungo l'intorno per selezionarlo int est
        if metric is 'mdd':
            epoch_index = np.argmin(mean_values) + 1
        else:
            epoch_index = np.argmax(mean_values) + 1

        #print(mean_values)
        #print("epoca selezionata:", epoch_index)
        #input()
        return epoch_index, romad_epochs[epoch_index], return_epochs[epoch_index], mdd_epochs[epoch_index]

    '''
    ' Leggo il file con le decisioni ultime
    ' Quindi calcolo tutte le metriche per quel CSV (unico per tutti gli walk)
    ' Utilizzo il delta_current_day poiché il vettore delle label è già 
    ' allineato al giorno corrente (giorno in cui deve venir fatta l'operazione)
    '''
    def get_results(self, ensemble_type='ensemble_magg', epoch_selection_policy='long_short', thr_ensemble_magg=0.35, thr_ensemble_exclusive=0.3, thr_ensemble_elimination=0.3, remove_nets=False, penalty=32, stop_loss=1000, insample=[]):
        remove_nets_str = 'con-rimozione-reti/'

        if remove_nets == False:
            remove_nets_str = 'senza-rimozione-reti/'

        #ensemble magg
        if ensemble_type is 'ensemble_magg':
            df = pd.read_csv(self.final_decision_folder + 'ensemble_magg/' + remove_nets_str + epoch_selection_policy + '/decisions_ensemble_magg_' + str(thr_ensemble_magg) + '.csv')

        if ensemble_type is 'ensemble_el_long_only':
            df = pd.read_csv(self.final_decision_folder + 'ensemble_el_longonly/' + remove_nets_str + epoch_selection_policy + '/decisions_ensemble_el_long_only_' + str(thr_ensemble_elimination) + '.csv')

        if ensemble_type is 'ensemble_exclusive':
            df = pd.read_csv(self.final_decision_folder + 'ensemble_exclusive/' + remove_nets_str + epoch_selection_policy + '/decisions_ensemble_exclusive_' + str(thr_ensemble_exclusive) + '.csv')

        # without ensemble
        if ensemble_type is 'without_ensemble':
            df = pd.read_csv(self.final_decision_folder + 'without_ensemble/decisions_without_ensemble.csv')

        close = df['close'].tolist()
        dates = df['date_time'].tolist()

        # BLOCCO DI CODICE IS1 - IS2
        if insample != []:
            df = Market.get_df_by_data_range(df=df.copy(), start_date=insample[0], end_date=insample[1])
            close = df['close'].tolist()
            dates = df['date_time'].tolist()
            #df = Market.get_df_by_data_range(df=df, start_date='2009-08-02', end_date='2014-01-31') # INSAMPLE 1 009
            #df = Market.get_df_by_data_range(df=df, start_date='2014-01-01', end_date='2016-12-31') # INSAMPLE 2 009
            #df = Market.get_df_by_data_range(df=df, start_date='2017-01-01', end_date='2018-01-31') # INSAMPLE 2 009
            #close = df['close'].tolist()
            #dates = df['date_time'].tolist()
            #df_min = df[['date_time', 'decision']]
            #if ensemble_type == 'ensemble_magg':
            #    df_min.to_csv('C://Users/Utente/Desktop/prova/' + ensemble_type + '_' + str(thr_ensemble_magg) + 'excl.csv', header=True, index=False)
                #input("Salvato " + ensemble_type + " - " + str(thr_ensemble_magg))

            #if ensemble_type == "ensemble_exclusive":
            #    df_min.to_csv('C://Users/Utente/Desktop/prova/' + ensemble_type + '_' + str(thr_ensemble_exclusive) + 'excl.csv', header=True, index=False)
                #input("Salvato " + ensemble_type + " - " + str(thr_ensemble_exclusive))
        
        # calcolo il b&h
        bh_equity_line, bh_global_return, bh_mdd, bh_romad, i, j = Measures.get_return_mdd_romad_bh(close=close, multiplier=self.iperparameters['return_multiplier'])

        # calcolo tutte le info per long + short, long + hold e short + hold
        long_info, short_info, hold_info, general_info = Measures.get_precision_count_coverage(df=df.copy(), penalty=penalty, stop_loss=stop_loss, multiplier=self.iperparameters['return_multiplier'], delta_to_use='delta_current_day')
        
        ls_equity_line, ls_global_return, ls_mdd, ls_romad, ls_i, ls_j = Measures.get_equity_return_mdd_romad(df=df.copy(), multiplier=self.iperparameters['return_multiplier'], type='long_short', penalty=penalty, stop_loss=stop_loss, delta_to_use='delta_current_day')
        lh_equity_line, lh_global_return, lh_mdd, lh_romad, lh_i, lh_j = Measures.get_equity_return_mdd_romad(df=df.copy(), multiplier=self.iperparameters['return_multiplier'], type='long_only', penalty=penalty, stop_loss=stop_loss, delta_to_use='delta_current_day')
        sh_equity_line, sh_global_return, sh_mdd, sh_romad, sh_i, sh_j = Measures.get_equity_return_mdd_romad(df=df.copy(), multiplier=self.iperparameters['return_multiplier'], type='short_only',penalty=penalty, stop_loss=stop_loss, delta_to_use='delta_current_day')

        
        ls_results = {
            "return": ls_global_return,
            "mdd": ls_mdd,
            "romad": ls_romad,
            "equity_line": ls_equity_line
        }

        l_results = {
            "return": lh_global_return,
            "mdd": lh_mdd,
            "romad": lh_romad,
            "equity_line": lh_equity_line,

        }

        s_results = {
            "return": sh_global_return,
            "mdd": sh_mdd,
            "romad": sh_romad,
            "equity_line": sh_equity_line,

        }

        bh_results = {
            "return": bh_global_return,
            "mdd": bh_mdd,
            "romad": bh_romad,
            "equity_line": bh_equity_line
        }

        general_info = {
            "long_info": long_info,
            "short_info": short_info,
            "hold_info": hold_info,
            "total_trade": general_info['total_trade'],
            "total_guessed_trade": general_info['total_guessed_trade'],
            "total_operation": general_info['total_operation'],
            "dates": dates
        }

        return ls_results, l_results, s_results, bh_results, general_info

    
    '''
    ' Leggo tutte le metriche in output da un tipo di ensemble con get_results
    ' e salvo un file excel con tute le informazioni che mi servono
    '''
    def get_report_excel(self, report_name, epoch_selection_policy='long_short', thr=0.3, remove_nets=False, type='ensemble_magg', stop_loss=1000, penalty=32):
        print("Generating Report Excel for:", type, "| Epoch selection policy:", epoch_selection_policy, "| Stop Loss:", stop_loss, "| Penalty:", penalty)
        type_str = ''

        if type is 'ensemble_magg':
            ls_results, l_results, s_results, bh_results, general_info = self.get_results(ensemble_type=type, epoch_selection_policy=epoch_selection_policy, remove_nets=remove_nets, thr_ensemble_magg=thr, stop_loss=stop_loss, penalty=penalty)
            type_str = 'Ens Magg'

        if type is 'ensemble_elimination':
            ls_results, l_results, s_results, bh_results, general_info = self.get_results(ensemble_type=type, epoch_selection_policy=epoch_selection_policy, remove_nets=remove_nets, thr_ensemble_elimination=thr, stop_loss=stop_loss, penalty=penalty)
            type_str = 'Ens El'

        if type is 'ensemble_exclusive':
            ls_results, l_results, s_results, bh_results, general_info = self.get_results(ensemble_type=type, epoch_selection_policy=epoch_selection_policy, remove_nets=remove_nets, thr_ensemble_exclusive=thr, stop_loss=stop_loss, penalty=penalty)
            type_str = 'Ens Excl'

        if type is 'without_ensemble':
            ls_results, l_results, s_results, bh_results, general_info = self.get_results(ensemble_type=type, epoch_selection_policy=epoch_selection_policy, remove_nets=remove_nets, stop_loss=stop_loss, penalty=penalty)
            type_str = 'Senza Ens'

        # Se non esiste la cartella, la creo
        report_path = self.experiment_original_path + self.experiment_name + '/reports/' + type_str + '/'

        create_folder(report_path)
        # Create an new Excel file and add a worksheet.
        workbook = xlsxwriter.Workbook(report_path + report_name + '.xlsx')

        ###
        cell_classic = workbook.add_format({'border': 1})
        cell_bold = workbook.add_format({'bold': True, 'border': 1})
        cell_header = workbook.add_format({'bold': True, 'font_size': 20})

        ###
        worksheet = workbook.add_worksheet('Report')
        # dimensione delle colonne 17 pollici = 126px
        worksheet.set_column('A:I', 17)

        worksheet.write('A1:E1', 'Report su Esperimento: ' + report_name, cell_header)

        worksheet.write('A3', '', cell_bold)
        worksheet.write('B3', 'Long + Short', cell_bold)
        worksheet.write('C3', 'Long Only', cell_bold)
        worksheet.write('D3', 'Short Only', cell_bold)
        worksheet.write('E3', 'Buy & Hold', cell_bold)

        worksheet.write('A4', 'Return', cell_bold)
        worksheet.write('A5', 'MDD', cell_bold)
        worksheet.write('A6', 'Romad', cell_bold)
        worksheet.write('A7', 'Avg Trade', cell_bold)
        #worksheet.write('A8', 'Time Under Water', cell_bold)

        worksheet.write('A9', 'Giorni Totali', cell_bold)

        worksheet.write('A11', 'Long Totali', cell_bold)
        worksheet.write('A12', 'Long Corrette', cell_bold)
        worksheet.write('A13', 'Precision delle Long', cell_bold)
        worksheet.write('A14', 'Coverage Long', cell_bold)

        worksheet.write('A16', 'Short Totali', cell_bold)
        worksheet.write('A17', 'Short Corrette', cell_bold)
        worksheet.write('A18', 'Precision delle Short', cell_bold)
        worksheet.write('A19', 'Coverage Short', cell_bold)

        worksheet.write('A21', 'Hold Totali', cell_bold)
        worksheet.write('A22', 'Hold Corrette', cell_bold)
        worksheet.write('A23', 'Precision delle Hold', cell_bold)
        worksheet.write('A24', 'Coverage hold', cell_bold)

        worksheet.set_column(22, 23, 20)
        worksheet.write('G3', "Data inizio test", cell_bold)
        worksheet.write('H3', "Data fine test", cell_bold)
        worksheet.write('G4', general_info['dates'][0], cell_classic)
        worksheet.write('H4', general_info['dates'][-1], cell_classic)

        # LONG + SHORT
        if general_info['total_trade'] > 0:
            ls_avg_winning_trade = round(ls_results['return'] / general_info['total_trade'], 3)
        else:
            ls_avg_winning_trade = 0

        worksheet.write('B4', ls_results['return'], cell_classic)
        worksheet.write('B5', ls_results['mdd'], cell_classic)
        worksheet.write('B6', round(ls_results['romad'], 3), cell_classic)
        worksheet.write('B7', ls_avg_winning_trade, cell_classic)
        #worksheet.write('B8', '', cell_classic)

        # LONG
        if general_info['long_info']['count'] > 0:
            l_avg_winning_trade = round(l_results['return'] / general_info['long_info']['count'], 3)
        else:
            l_avg_winning_trade = 0

        worksheet.write('C4', l_results['return'], cell_classic)
        worksheet.write('C5', l_results['mdd'], cell_classic)
        worksheet.write('C6', round(l_results['romad'], 3), cell_classic)
        worksheet.write('C7', l_avg_winning_trade, cell_classic)
        #worksheet.write('C8', '', cell_classic)

        # SHORT
        if general_info['short_info']['count'] > 0:
            s_avg_winning_trade = round(s_results['return'] / general_info['short_info']['count'], 3)
        else:
            s_avg_winning_trade = 0

        worksheet.write('D4', s_results['return'], cell_classic)
        worksheet.write('D5', s_results['mdd'], cell_classic)
        worksheet.write('D6', round(s_results['romad'], 3), cell_classic)
        worksheet.write('D7', s_avg_winning_trade, cell_classic)
        #worksheet.write('D8', '', cell_classic)

        # BUY & HOLD
        worksheet.write('E4', bh_results['return'], cell_classic)
        worksheet.write('E5', bh_results['mdd'], cell_classic)
        worksheet.write('E6', round(bh_results['romad'], 3), cell_classic)
        worksheet.write('E7', '', cell_classic)
        #worksheet.write('E8', '', cell_classic)

        # GENERAL
        total_days = general_info['long_info']['count'] + \
            general_info['short_info']['count'] + \
            general_info['hold_info']['count']
        worksheet.write('B9', total_days, cell_classic)

        worksheet.write(
            'B11', general_info['long_info']['count'], cell_classic)
        worksheet.write(
            'B12', general_info['long_info']['guessed'], cell_classic)
        worksheet.write('B13', round(
            general_info['long_info']['precision'], 3), cell_classic)
        worksheet.write('B14', round(
            100 * general_info['long_info']['count'] / total_days, 3), cell_classic)  # coverage

        worksheet.write(
            'B16', general_info['short_info']['count'], cell_classic)
        worksheet.write(
            'B17', general_info['short_info']['guessed'], cell_classic)
        worksheet.write('B18', round(
            general_info['short_info']['precision'], 3), cell_classic)
        worksheet.write('B19', round(
            100 * general_info['short_info']['count'] / total_days, 3), cell_classic)  # coverage

        worksheet.write(
            'B21', general_info['hold_info']['count'], cell_classic)
        worksheet.write('B22', "-", cell_classic)
        worksheet.write('B23', "-", cell_classic)
        worksheet.write('B24', round(
            100 * general_info['hold_info']['count'] / total_days, 3), cell_classic)  # coverage

        # PESI CUSTOM LOSS
        worksheet.write('D10:E10', "Pesi Loss", cell_header)
        worksheet.write('D11', "Predizione", cell_bold)
        worksheet.write('E11', "Label", cell_bold)
        worksheet.write('F11', "Errore", cell_bold)

        worksheet.write('D12', "L", cell_classic)
        worksheet.write('D13', "L", cell_classic)
        worksheet.write('D14', "L", cell_classic)
        worksheet.write('D15', "H", cell_classic)
        worksheet.write('D16', "H", cell_classic)
        worksheet.write('D17', "H", cell_classic)
        worksheet.write('D18', "S", cell_classic)
        worksheet.write('D19', "S", cell_classic)
        worksheet.write('D20', "S", cell_classic)

        worksheet.write('E12', "L", cell_classic)
        worksheet.write('E13', "H", cell_classic)
        worksheet.write('E14', "S", cell_classic)
        worksheet.write('E15', "L", cell_classic)
        worksheet.write('E16', "H", cell_classic)
        worksheet.write('E17', "S", cell_classic)
        worksheet.write('E18', "L", cell_classic)
        worksheet.write('E19', "H", cell_classic)
        worksheet.write('E20', "S", cell_classic)

        worksheet.write(
            'F12', self.iperparameters['loss_weight'][2][2], cell_classic)
        worksheet.write(
            'F13', self.iperparameters['loss_weight'][1][2], cell_classic)
        worksheet.write(
            'F14', self.iperparameters['loss_weight'][0][2], cell_classic)
        worksheet.write(
            'F15', self.iperparameters['loss_weight'][2][1], cell_classic)
        worksheet.write(
            'F16', self.iperparameters['loss_weight'][1][1], cell_classic)
        worksheet.write(
            'F17', self.iperparameters['loss_weight'][0][1], cell_classic)
        worksheet.write(
            'F18', self.iperparameters['loss_weight'][2][0], cell_classic)
        worksheet.write(
            'F19', self.iperparameters['loss_weight'][1][0], cell_classic)
        worksheet.write(
            'F20', self.iperparameters['loss_weight'][0][0], cell_classic)

        # IPERPARAMETRI ZONE
        worksheet.write('B26:C26', "Date Training Set", cell_header)
        worksheet.write('D26:E26', "Date Validation Set", cell_header)
        worksheet.write('F26:G26', "Date Test Set", cell_header)

        worksheet.write('A27', "Walk n°", cell_bold)
        worksheet.write('B27', "Inizio", cell_bold)
        worksheet.write('C27', "Fine", cell_bold)
        worksheet.write('D27', "Inizio", cell_bold)
        worksheet.write('E27', "Fine", cell_bold)
        worksheet.write('F27', "Inizio", cell_bold)
        worksheet.write('G27', "Fine", cell_bold)

        # Salvo tutte le date degli walk
        for i, value in enumerate(self.iperparameters['training_set']):
            worksheet.write('A' + str(28 + i), 1 + i, cell_bold)
            worksheet.write(
                'B' + str(28 + i), self.iperparameters['training_set'][i][0], cell_classic)
            worksheet.write(
                'C' + str(28 + i), self.iperparameters['training_set'][i][1], cell_classic)
            worksheet.write(
                'D' + str(28 + i), self.iperparameters['validation_set'][i][0], cell_classic)
            worksheet.write(
                'E' + str(28 + i), self.iperparameters['validation_set'][i][1], cell_classic)
            worksheet.write('F' + str(28 + i),
                            self.iperparameters['test_set'][i][0], cell_classic)
            worksheet.write('G' + str(28 + i),
                            self.iperparameters['test_set'][i][1], cell_classic)

        worksheet.write('H10:I10', "Configurazione rete", cell_header)
        worksheet.write('H11', "Numero epoche", cell_bold)
        worksheet.write('H12', "Numero reti", cell_bold)
        worksheet.write('H13', "Learning Rate", cell_bold)
        worksheet.write('H14', "Batch Size", cell_bold)
        worksheet.write('H15', "Loss Function", cell_bold)
        worksheet.write('H16', "Intorno ensemble", cell_bold)
        worksheet.write('H17', "Mercato", cell_bold)

        worksheet.write('I11', self.iperparameters['epochs'], cell_classic)
        worksheet.write(
            'I12', self.iperparameters['number_of_nets'], cell_classic)
        worksheet.write('I13', self.iperparameters['init_lr'], cell_classic)
        worksheet.write('I14', self.iperparameters['bs'], cell_classic)
        worksheet.write(
            'I15', self.iperparameters['loss_function'], cell_classic)
        worksheet.write(
            'I16', self.iperparameters['validation_thr'], cell_classic)
        worksheet.write(
            'I17', self.iperparameters['predictions_dataset'], cell_classic)

        #################################################################################################
        #################################################################################################
        #################################################################################################

        first_value_of_bh = bh_results['equity_line'][0]
        bh_results['equity_line'] = [
            x - first_value_of_bh for x in bh_results['equity_line']]

        # Equity long + short
        worksheet = workbook.add_worksheet('CurvaEquityLongShort')
        worksheet.write_column('AA1', general_info['dates'])
        worksheet.write_column('AB1', ls_results['equity_line'])
        worksheet.write_column('AC1', bh_results['equity_line'])

        chart = workbook.add_chart({'type': 'line'})
        chart.add_series({'name': 'Equity Long + Short',
                          'categories': '=CurvaEquityLongShort!$AA$1:$AA$' + str(len(general_info['dates'])),
                          'values': '=CurvaEquityLongShort!$AB$1:$AB$' + str(len(ls_results['equity_line']))
                          })

        chart.add_series({'name': 'Buy & Hold',
                          'categories': '=CurvaEquityLongShort!$AA$1:$AA$' + str(len(general_info['dates'])),
                          'values': '=CurvaEquityLongShort!$AC$1:$AC$' + str(len(bh_results['equity_line']))
                          })

        chart.set_title({'name': 'Curva Equity Long + Short'})
        chart.set_x_axis({'name': 'Giorni'})
        chart.set_y_axis({'name': 'Return ($)'})
        chart.set_style(2)
        chart.set_size({'width': 1200, 'height': 500})
        worksheet.insert_chart('A1', chart)

        # Equity long  only
        worksheet = workbook.add_worksheet('CurvaEquityLongOnly')
        worksheet.write_column('AA1', general_info['dates'])
        worksheet.write_column('AB1', l_results['equity_line'])
        worksheet.write_column('AC1', bh_results['equity_line'])

        chart = workbook.add_chart({'type': 'line'})
        chart.add_series({'name': 'Equity Long Only',
                          'categories': '=CurvaEquityLongOnly!$AA$1:$AA$' + str(len(general_info['dates'])),
                          'values': '=CurvaEquityLongOnly!$AB$1:$AB$' + str(len(l_results['equity_line']))
                          })
        chart.add_series({'name': 'Buy & Hold',
                          'categories': '=CurvaEquityLongOnly!$AA$1:$AA$' + str(len(general_info['dates'])),
                          'values': '=CurvaEquityLongOnly!$AC$1:$AC$' + str(len(bh_results['equity_line']))
                          })

        chart.set_title({'name': 'Equity Long Only'})
        chart.set_x_axis({'name': 'Giorni'})
        chart.set_y_axis({'name': 'Return ($)'})
        chart.set_style(2)
        chart.set_size({'width': 1200, 'height': 500})
        worksheet.insert_chart('A1', chart)

        # Equity short  only
        worksheet = workbook.add_worksheet('CurvaEquityShortgOnly')
        worksheet.write_column('AA1', general_info['dates'])
        worksheet.write_column('AB1', s_results['equity_line'])
        worksheet.write_column('AC1', bh_results['equity_line'])

        chart = workbook.add_chart({'type': 'line'})
        chart.add_series({'name': 'Equity Short Only',
                          'categories': '=CurvaEquityShortgOnly!$AA$1:$AA$' + str(len(general_info['dates'])),
                          'values': '=CurvaEquityShortgOnly!$AB$1:$AB$' + str(len(s_results['equity_line']))
                          })
        chart.add_series({'name': 'Buy & Hold',
                          'categories': '=CurvaEquityShortgOnly!$AA$1:$AA$' + str(len(general_info['dates'])),
                          'values': '=CurvaEquityShortgOnly!$AC$1:$AC$' + str(len(bh_results['equity_line']))
                          })

        chart.set_title({'name': 'Curva Equity Short Only'})
        chart.set_x_axis({'name': 'Giorni'})
        chart.set_y_axis({'name': 'Return ($)'})
        chart.set_style(2)
        chart.set_size({'width': 1200, 'height': 500})
        worksheet.insert_chart('A1', chart)

        workbook.close()

    '''
    '
    '''
    def get_report_excel_swipe(self, report_name, type, thrs_ensemble_magg=[], thrs_ensemble_exclusive=[], thrs_ensemble_elimination=[], remove_nets=False, 
    epoch_selection_policy='long_short', stop_loss=1000, penalty=32, insample=[], subfolder_is=''):
        print("Generating Report Excel Swipe " + subfolder_is + " for:", type, "| Epoch selection policy:", epoch_selection_policy, "| Stop Loss:", stop_loss, "| Penalty:", penalty)
        thr_global = []

        if type is 'ensemble_magg':
            thr_global = thrs_ensemble_magg
        
        if type is 'ensemble_exclusive':
            thr_global = thrs_ensemble_exclusive
        
        if type is 'ensemble_elimination':
            thr_global = thrs_ensemble_elimination

        ls_results = [dict() for x in range(len(thr_global))] 
        l_results = [dict() for x in range(len(thr_global))] 
        s_results = [dict() for x in range(len(thr_global))] 
        bh_results = [dict() for x in range(len(thr_global))] 
        general_info = [dict() for x in range(len(thr_global))] 

        if type is 'ensemble_magg':
            for i, thr in enumerate(thr_global):
                ls_results[i], l_results[i], s_results[i], bh_results[i], general_info[i] = self.get_results(ensemble_type=type, epoch_selection_policy=epoch_selection_policy, 
                    thr_ensemble_magg=thr, remove_nets=remove_nets, stop_loss=stop_loss, penalty=penalty, insample=insample)

        if type is 'ensemble_exclusive':
            for i, thr in enumerate(thr_global):
                ls_results[i], l_results[i], s_results[i], bh_results[i], general_info[i] = self.get_results(ensemble_type=type, epoch_selection_policy=epoch_selection_policy, 
                    thr_ensemble_exclusive=thr, remove_nets=remove_nets, stop_loss=stop_loss, penalty=penalty, insample=insample)

        # Se non esiste la cartella, la creo
        report_path = self.experiment_original_path + self.experiment_name + '/reports/' + subfolder_is + '/'
        if not os.path.isdir(report_path):
            os.makedirs(report_path)

        # Create an new Excel file and add a worksheet.
        workbook = xlsxwriter.Workbook(report_path + report_name + '.xlsx')

        ###
        cell_classic = workbook.add_format({'border': 1})
        cell_bold = workbook.add_format({'bold': True, 'border': 1})
        cell_header = workbook.add_format({'bold': True, 'font_size': 20})

        ###
        worksheet = workbook.add_worksheet('Report')

        # INSERISCO LE DATE DI INIZIO E FINE TRADE. NON PRENDE I DATA RANGE, MA LE DATE EFFETTIVE
        # CIOE' LA PRIMA E L'ULTIMA DATA IN CUI ENTRA
        worksheet.set_column(22, 23, 20)
        worksheet.write('W2', "Inizio test set", cell_bold)
        worksheet.write('X2', "Fine test set", cell_bold)
        worksheet.write('W3', general_info[0]['dates'][0], cell_classic)
        worksheet.write('X3', general_info[0]['dates'][-1], cell_classic)

        worksheet.write_column('BA2', thr_global) # categorie

        worksheet.write('BB1', "Return Long + Short")
        worksheet.write('BC1', "MDD Long + Short")
        worksheet.write('BD1', "Romad Long + Short")
        
        worksheet.write('BE1', "Return Long Only")
        worksheet.write('BF1', "MDD Long Only")
        worksheet.write('BG1', "Romad Long Only")

        worksheet.write('BH1', "Return Short Only")
        worksheet.write('BJ1', "MDD Short Only")
        worksheet.write('BK1', "Romad Short Only")
        
        worksheet.write('BL1', "Precision Long ")
        worksheet.write('BM1', "Coverage Long")
        worksheet.write('BN1', "Precision Short")
        worksheet.write('BO1', "Coverage Short")
        worksheet.write('BP1', "Coverage Hold")
        
        
        # BH
        worksheet.write('BQ1', "BH Return")
        worksheet.write('BR1', "BH MDD")
        worksheet.write('BS1', "BH Romad")

        # Random  Precision / Percentuali di labeling nel dataset
        worksheet.write('BT1', "Random Precision Long")
        worksheet.write('BU1', "Random Precision Short")
        worksheet.write('BV1', "Random Precision Hold")
        
        worksheet.write('BZ1', "Count Long")
        worksheet.write('CA1', "Count Short")
        worksheet.write('CB1', "Count Hold")
        
        for i in range(0, len(thr_global)):
            worksheet.write('BB' + str(i + 2), ls_results[i]['return'])
            worksheet.write('BC' + str(i + 2), ls_results[i]['mdd'])
            worksheet.write('BD' + str(i + 2), ls_results[i]['romad'])

            worksheet.write('BE' + str(i + 2), l_results[i]['return'])
            worksheet.write('BF' + str(i + 2), l_results[i]['mdd'])
            worksheet.write('BG' + str(i + 2), l_results[i]['romad'])

            worksheet.write('BH' + str(i + 2), s_results[i]['return'])
            worksheet.write('BJ' + str(i + 2), s_results[i]['mdd'])
            worksheet.write('BK' + str(i + 2), s_results[i]['romad'])

            worksheet.write('BL' + str(i + 2), general_info[i]['long_info']['precision'])
            worksheet.write('BM' + str(i + 2), general_info[i]['long_info']['coverage'])
            
            worksheet.write('BN' + str(i + 2), general_info[i]['short_info']['precision'])
            worksheet.write('BO' + str(i + 2), general_info[i]['short_info']['coverage'])
            worksheet.write('BP' + str(i + 2), general_info[i]['hold_info']['coverage'])

            worksheet.write('BQ' + str(i + 2), bh_results[i]['return'])
            worksheet.write('BR' + str(i + 2), bh_results[i]['mdd'])
            worksheet.write('BS' + str(i + 2), bh_results[i]['romad'])

            worksheet.write('BT' + str(i + 2), general_info[i]['long_info']['random_perc'])
            worksheet.write('BU' + str(i + 2), general_info[i]['short_info']['random_perc'])
            worksheet.write('BV' + str(i + 2), general_info[i]['hold_info']['random_perc'])

            worksheet.write('BZ' + str(i + 2), general_info[i]['long_info']['count'])
            worksheet.write('CA' + str(i + 2), general_info[i]['short_info']['count'])
            #worksheet.write('CB' + str(i + 2), general_info[i]['hold_info']['count'])
        #worksheet.write_column('AC1', bh_results['equity_line'])

        # RETURN 
        chart = workbook.add_chart({'type': 'line'})
        chart.add_series({'name': 'Return Long Short',
                          'categories': '=Report!$BA$2:$BA$' + str(len(thr_global) + 1),
                          'values': '=Report!$BB$2:$BB$' + str(len(thr_global) + 1),
                          'line':   {'color': 'blue'}
                          })
        chart.add_series({'name': 'Return Long Only',
                          'categories': '=Report!$BA$2:$BA$' + str(len(thr_global) + 1),
                          'values': '=Report!$BE$2:$BE$' + str(len(thr_global) + 1),
                          'line':   {'color': 'green'}
                          })
        chart.add_series({'name': 'Return Short Only',
                          'categories': '=Report!$BA$2:$BA$' + str(len(thr_global) + 1),
                          'values': '=Report!$BH$2:$BH$' + str(len(thr_global) + 1),
                          'line':   {'color': 'red'}
                          })
        chart.add_series({'name': 'Return Buy Hold',
                          'categories': '=Report!$BA$2:$BA$' + str(len(thr_global) + 1),
                          'values': '=Report!$BQ$2:$BQ$' + str(len(thr_global) + 1),
                          'line':   {'color': 'orange', 'dash_type': 'dash'}
                          })
        chart.set_style(2)
        chart.set_size({'width': 635, 'height': 300})
        worksheet.insert_chart('A1', chart)

        # MDD
        chart = workbook.add_chart({'type': 'line'})
        chart.add_series({'name': 'MDD Long Short',
                          'categories': '=Report!$BA$2:$BA$' + str(len(thr_global) + 1),
                          'values': '=Report!$BC$2:$BC$' + str(len(thr_global) + 1),
                          'line':   {'color': 'blue'}
                          })
        chart.add_series({'name': 'MDD Long Only',
                          'categories': '=Report!$BA$2:$BA$' + str(len(thr_global) + 1),
                          'values': '=Report!$BF$2:$BF$' + str(len(thr_global) + 1),
                          'line':   {'color': 'green'}
                          })
        chart.add_series({'name': 'MDD Short Only',
                          'categories': '=Report!$BA$2:$BA$' + str(len(thr_global) + 1),
                          'values': '=Report!$BJ$2:$BJ$' + str(len(thr_global) + 1),
                          'line':   {'color': 'red'}
                          })   
        chart.add_series({'name': 'MDD Buy Hold',
                          'categories': '=Report!$BA$2:$BA$' + str(len(thr_global) + 1),
                          'values': '=Report!$BR$2:$BR$' + str(len(thr_global) + 1),
                          'line':   {'color': 'orange', 'dash_type': 'dash'}
                          })              
        chart.set_style(2)
        chart.set_size({'width': 635, 'height': 300})
        worksheet.insert_chart('A17', chart)

        # ROMAD
        chart = workbook.add_chart({'type': 'line'})
        chart.add_series({'name': 'Romad Long Short',
                          'categories': '=Report!$BA$2:$BA$' + str(len(thr_global) + 1),
                          'values': '=Report!$BD$2:$BD$' + str(len(thr_global) + 1),
                          'line':   {'color': 'blue'}
                          })
        chart.add_series({'name': 'Romad Long Only',
                          'categories': '=Report!$BA$2:$BA$' + str(len(thr_global) + 1),
                          'values': '=Report!$BG$2:$BG$' + str(len(thr_global) + 1),
                          'line':   {'color': 'green'}
                          })
        chart.add_series({'name': 'Romad Short Only',
                          'categories': '=Report!$BA$2:$BA$' + str(len(thr_global) + 1),
                          'values': '=Report!$BK$2:$BK$' + str(len(thr_global) + 1),
                          'line':   {'color': 'red'}
                          })
        chart.add_series({'name': 'Romad Buy Hold',
                          'categories': '=Report!$BA$2:$BA$' + str(len(thr_global) + 1),
                          'values': '=Report!$BS$2:$BS$' + str(len(thr_global) + 1),
                          'line':   {'color': 'orange', 'dash_type': 'dash'}
                          })
        chart.set_style(2)
        chart.set_size({'width': 635, 'height': 300})
        worksheet.insert_chart('A33', chart)
        
        # PRECISION
        chart = workbook.add_chart({'type': 'line'})
        chart.add_series({'name': 'Precision Long',
                          'categories': '=Report!$BA$2:$BA$' + str(len(thr_global) + 1),
                          'values': '=Report!$BL$2:$BL$' + str(len(thr_global) + 1),
                          'line':   {'color': 'green'}
                          })
        chart.add_series({'name': 'Precision Short',
                          'categories': '=Report!$BA$2:$BA$' + str(len(thr_global) + 1),
                          'values': '=Report!$BN$2:$BN$' + str(len(thr_global) + 1),
                          'line':   {'color': 'red'}
                          })
        chart.add_series({'name': 'Random Long',
                          'categories': '=Report!$BA$2:$BA$' + str(len(thr_global) + 1),
                          'values': '=Report!$BT$2:$BT$' + str(len(thr_global) + 1),
                          'line':   {'color': 'green', 'dash_type': 'dash'}
                          })
        chart.add_series({'name': 'Random Short',
                          'categories': '=Report!$BA$2:$BA$' + str(len(thr_global) + 1),
                          'values': '=Report!$BU$2:$BU$' + str(len(thr_global) + 1),
                          'line':   {'color': 'red', 'dash_type': 'dash'}
                          })
        chart.set_style(2)
        chart.set_size({'width': 635, 'height': 300})
        worksheet.insert_chart('L1', chart)

        # COVERAGE
        chart = workbook.add_chart({'type': 'line'})
        chart.add_series({'name': 'Operazioni Long',
                          'categories': '=Report!$BA$2:$BA$' + str(len(thr_global) + 1),
                          'values': '=Report!$BM$2:$BM$' + str(len(thr_global) + 1),
                          'line':   {'color': 'green'}
                          })
        chart.add_series({'name': 'Operazioni Short',
                          'categories': '=Report!$BA$2:$BA$' + str(len(thr_global) + 1),
                          'values': '=Report!$BO$2:$BO$' + str(len(thr_global) + 1),
                          'line':   {'color': 'red'}
                          })
        chart.add_series({'name': 'Operazioni Hold',
                          'categories': '=Report!$BA$2:$BA$' + str(len(thr_global) + 1),
                          'values': '=Report!$BP$2:$BP$' + str(len(thr_global) + 1),
                          'line':   {'color': 'gray'}
                          })
        chart.set_style(2)
        chart.set_size({'width': 635, 'height': 300})
        worksheet.insert_chart('L17', chart)

        workbook.close()
    
    '''
    █▀▀ █░░█ █▀▀ ▀▀█▀▀ █▀▀█ █▀▄▀█   █▀▄▀█ █▀▀ ▀▀█▀▀ █▀▀█ ░▀░ █▀▀ █▀▀
    █░░ █░░█ ▀▀█ ░░█░░ █░░█ █░▀░█   █░▀░█ █▀▀ ░░█░░ █▄▄▀ ▀█▀ █░░ ▀▀█
    ▀▀▀ ░▀▀▀ ▀▀▀ ░░▀░░ ▀▀▀▀ ▀░░░▀   ▀░░░▀ ▀▀▀ ░░▀░░ ▀░▀▀ ▀▀▀ ▀▀▀ ▀▀▀
    '''

    '''
    '
    '''
    def generate_custom_metrics(self, type='validation', penalty=32, stop_loss=1000):

        for index_walk in range(0, len(self.walks_list)):
            walk_str = 'walk_' + str(index_walk)
            if type is 'validation':
                date_list, epochs_list = self.get_date_epochs_walk(path=self.original_predictions_validation_folder, walk=walk_str)
            if type is 'test': 
                date_list, epochs_list = self.get_date_epochs_walk(path=self.original_predictions_test_folder, walk=walk_str)

            
            avg_ls_returns = []
            avg_lh_returns = []
            avg_sh_returns = []
            avg_bh_returns = [] # BH

            # MDDS
            avg_ls_mdds = []
            avg_lh_mdds = []
            avg_sh_mdds = []
            avg_bh_mdds = [] # BH

            # ROMADS
            avg_ls_romads = []
            avg_lh_romads = []
            avg_sh_romads = []
            avg_bh_romads = []

            avg_longs_precisions = []
            avg_shorts_precisions = []
            
            avg_label_longs_coverage = []
            avg_label_shorts_coverage = []

            # NUOVO TEST AVG POC
            avg_longs_poc = []
            avg_shorts_poc = []

            # RETURNS
            all_ls_returns = np.zeros(shape=(len(self.nets_list), len(epochs_list)))
            all_lh_returns = np.zeros(shape=(len(self.nets_list), len(epochs_list)))
            all_sh_returns = np.zeros(shape=(len(self.nets_list), len(epochs_list)))
            all_bh_return = np.zeros(shape=(len(self.nets_list), len(epochs_list))) #BH

            # ROMADS
            all_ls_romads = np.zeros(shape=(len(self.nets_list), len(epochs_list)))
            all_lh_romads = np.zeros(shape=(len(self.nets_list), len(epochs_list)))
            all_sh_romads = np.zeros(shape=(len(self.nets_list), len(epochs_list)))
            all_bh_romads = np.zeros(shape=(len(self.nets_list), len(epochs_list))) #BH

            # MDDS
            all_ls_mdds = np.zeros(shape=(len(self.nets_list), len(epochs_list)))
            all_lh_mdds = np.zeros(shape=(len(self.nets_list), len(epochs_list)))
            all_sh_mdds = np.zeros(shape=(len(self.nets_list), len(epochs_list)))
            all_bh_mdds = np.zeros(shape=(len(self.nets_list), len(epochs_list))) # BH

            # PRECISIONI E LINEA RETTA DEL BILANCIAMENTO DELLE CLASSI
            all_longs_precisions = np.zeros(shape=(len(self.nets_list), len(epochs_list)))
            all_shorts_precisions = np.zeros(shape=(len(self.nets_list), len(epochs_list)))
            all_labels_longs_coverage = np.zeros(shape=(len(self.nets_list), len(epochs_list)))
            all_labels_shorts_coverage = np.zeros(shape=(len(self.nets_list), len(epochs_list)))

            # % di operazioni fatte 
            all_long_operations = np.zeros(shape=(len(self.nets_list), len(epochs_list)))
            all_short_operations = np.zeros(shape=(len(self.nets_list), len(epochs_list)))
            all_hold_operations = np.zeros(shape=(len(self.nets_list), len(epochs_list)))

            # Precision over coverage
            all_longs_poc = np.zeros(shape=(len(self.nets_list), len(epochs_list)))
            all_shorts_poc = np.zeros(shape=(len(self.nets_list), len(epochs_list)))

            for index_net in range(0, len(self.nets_list)):
                net = 'net_' + str(index_net) + '.csv'
                # leggo le predizioni fatte con l'esnemble
                df = pd.read_csv(self.experiment_original_path + self.iperparameters['experiment_name'] + '/predictions/predictions_during_training/' + type + '/walk_' + str(index_walk) + '/' + net)

                # mergio con le label, così ho un subset del df con le date che mi servono e la predizione 
                df_merge_with_label = df_date_merger(df=df, columns=['date_time', 'delta_next_day', 'close', 'open', 'high', 'low'], dataset=self.iperparameters['predictions_dataset'], thr_hold=self.iperparameters['hold_labeling'])

                # RETURNS 
                ls_returns = []
                lh_returns = []
                sh_returns = []
                bh_returns = [] # BH

                # ROMADS
                ls_romads = []
                lh_romads = []
                sh_romads = []
                bh_romads = [] # BH

                # MDDS
                ls_mdds = []
                lh_mdds = []
                sh_mdds = []
                bh_mdds = [] # BH

                # PRECISIONI E LINEA RETTA DEL BILANCIAMENTO DELLE CLASSI
                longs_precisions = []
                shorts_precisions = []
                longs_label_coverage = []
                shorts_label_coverage = []

                # % DI OPERAZIONI FATTE
                long_operations = []
                short_operations = []
                hold_operations = []

                # POC
                longs_poc = []
                shorts_poc = []
                
                label_coverage = Measures.get_delta_coverage(delta=df_merge_with_label['delta_next_day'].tolist())

                bh_equity_line, bh_global_return, bh_mdd, bh_romad, bh_i, bh_j  = Measures.get_return_mdd_romad_bh(close=df_merge_with_label['close'].tolist(), multiplier=self.iperparameters['return_multiplier'])

                dates_debug = df_merge_with_label['date_time'].tolist()

                #print("Type set:", type, "| Return BH per le date:", dates_debug[0], "-", dates_debug[-1], "|", bh_global_return)
                #input()
                # calcolo il return per un epoca
                for epoch in range(1, len(epochs_list) + 1): 
                    df_epoch_rename = df_merge_with_label.copy()
                    df_epoch_rename = df_epoch_rename.rename(columns={'epoch_' + str(epoch): 'decision'})

                    ls_equity_line, ls_global_return, ls_mdd, ls_romad, ls_i, ls_j  = Measures.get_equity_return_mdd_romad(df=df_epoch_rename.copy(), multiplier=self.iperparameters['return_multiplier'],
                        type='long_short', penalty=penalty, stop_loss=stop_loss, delta_to_use='delta_next_day')
                    
                    lh_equity_line, lh_global_return, lh_mdd, lh_romad, lh_i, lh_j  = Measures.get_equity_return_mdd_romad(df=df_epoch_rename.copy(), multiplier=self.iperparameters['return_multiplier'], 
                        type='long_only', penalty=penalty, stop_loss=stop_loss, delta_to_use='delta_next_day')
                    
                    sh_equity_line, sh_global_return, sh_mdd, sh_romad, sh_i, sh_j  = Measures.get_equity_return_mdd_romad(df=df_epoch_rename.copy(), multiplier=self.iperparameters['return_multiplier'], 
                        type='short_only', penalty=penalty, stop_loss=stop_loss, delta_to_use='delta_next_day')
                    
                    long, short, hold, general = Measures.get_precision_count_coverage(df=df_epoch_rename.copy(), multiplier=self.iperparameters['return_multiplier'], stop_loss=0, penalty=0, delta_to_use='delta_next_day')
                    long_poc, short_poc = Measures.get_precision_over_coverage(df=df_epoch_rename.copy(), multiplier=self.iperparameters['return_multiplier'], stop_loss=0, penalty=0, delta_to_use='delta_next_day')

                    # RETURNS 
                    ls_returns.append(ls_global_return)
                    lh_returns.append(lh_global_return)
                    sh_returns.append(sh_global_return)
                    bh_returns.append(bh_global_return) # BH

                    # ROMADS
                    ls_romads.append(ls_romad)
                    lh_romads.append(lh_romad)
                    sh_romads.append(sh_romad)
                    bh_romads.append(bh_romad) # BH

                    # MDDS
                    ls_mdds.append(ls_mdd)
                    lh_mdds.append(lh_mdd)
                    sh_mdds.append(sh_mdd)
                    bh_mdds.append(bh_mdd) # BH

                    # PRECISIONI E LINEA RETTA DEL BILANCIAMENTO DELLE CLASSI
                    longs_precisions.append(long['precision'])
                    shorts_precisions.append(short['precision'])
                    longs_label_coverage.append(label_coverage['long'])
                    shorts_label_coverage.append(label_coverage['short'])

                    # % di operazioni fatte
                    long_operations.append(long['coverage'])
                    short_operations.append(short['coverage'])
                    hold_operations.append(hold['coverage'])

                    # POC
                    longs_poc.append(long_poc)
                    shorts_poc.append(short_poc)

                net_json = {
                    "ls_returns": ls_returns,
                    "lh_returns": lh_returns,
                    "sh_returns": sh_returns,
                    "bh_returns": bh_returns,

                    "ls_romads": ls_romads,
                    "lh_romads": lh_romads,
                    "sh_romads": sh_romads,
                    "bh_romads": bh_romads,

                    "ls_mdds": ls_mdds,
                    "lh_mdds": lh_mdds,
                    "sh_mdds": sh_mdds,
                    "bh_mdds": bh_mdds,

                    "longs_precisions": longs_precisions,
                    "shorts_precisions": shorts_precisions,
                    "longs_label_coverage": longs_label_coverage,
                    "shorts_label_coverage": shorts_label_coverage,

                    "long_operations": long_operations,
                    "short_operations": short_operations,
                    "hold_operations": hold_operations,

                    "longs_poc": longs_poc,
                    "shorts_poc": shorts_poc           
                }

                output_path = self.experiment_original_path + self.iperparameters['experiment_name'] + '/calculated_metrics/' + type + '/walk_' + str(index_walk) + '/' 

                create_folder(output_path)

                with open(output_path + 'net_' + str(index_net) + '.json', 'w') as json_file:
                    json.dump(net_json, json_file, indent=4)
                    #json.dump(net_json, json_file)
                
                # PLOT SINGOLA RETE
                do_plot(metrics=net_json, walk=index_walk, epochs=len(epochs_list), main_path=self.experiment_original_path, experiment_name=self.iperparameters['experiment_name'], net=index_net, type=type)
                
                print(self.iperparameters['experiment_name'] + ' | ' + type + " - Salvate le metriche per walk n° ", index_walk, " rete: ", net)

                # RETURNS
                all_ls_returns[index_net] = ls_returns
                all_lh_returns[index_net] = lh_returns
                all_sh_returns[index_net] = sh_returns
                all_bh_return[index_net] = bh_returns # BH

                # ROMADS
                all_ls_romads[index_net] = ls_romads
                all_lh_romads[index_net] = lh_romads
                all_sh_romads[index_net] = sh_romads
                all_bh_romads[index_net] = bh_romads # BH

                # MDDS
                all_ls_mdds[index_net] = ls_mdds
                all_lh_mdds[index_net] = lh_mdds
                all_sh_mdds[index_net] = sh_mdds
                all_bh_mdds[index_net] = bh_mdds #BH

                # PRECISIONI E LINEA RETTA DEL BILANCIAMENTO DELLE CLASSI
                all_longs_precisions[index_net] = longs_precisions
                all_shorts_precisions[index_net] = shorts_precisions
                all_labels_longs_coverage[index_net] = longs_label_coverage
                all_labels_shorts_coverage[index_net] = shorts_label_coverage

                # % di operazioni fatte
                all_long_operations[index_net] = long_operations
                all_short_operations[index_net] = short_operations
                all_hold_operations[index_net] = hold_operations

                all_longs_poc[index_net] = longs_poc
                all_shorts_poc[index_net] = shorts_poc

            # RETURNS
            avg_ls_returns = np.around(np.average(all_ls_returns, axis=0), decimals=3)
            avg_lh_returns = np.around(np.average(all_lh_returns, axis=0), decimals=3)
            avg_sh_returns = np.around(np.average(all_sh_returns, axis=0), decimals=3)
            avg_bh_returns = np.average(all_bh_return, axis=0) # BH

            # MDDS
            avg_ls_mdds = np.around(np.average(all_ls_mdds, axis=0), decimals=3)
            avg_lh_mdds = np.around(np.average(all_lh_mdds, axis=0), decimals=3)
            avg_sh_mdds = np.around(np.average(all_sh_mdds, axis=0), decimals=3)
            avg_bh_mdds = np.average(all_bh_mdds, axis=0) # BH

            # ROMADS
            avg_ls_romads = np.divide(avg_ls_returns, avg_ls_mdds, out=np.zeros_like(avg_ls_returns), where=avg_ls_mdds!=0)
            avg_lh_romads = np.divide(avg_lh_returns, avg_lh_mdds, out=np.zeros_like(avg_ls_returns), where=avg_ls_mdds!=0)
            avg_sh_romads = np.divide(avg_sh_returns, avg_sh_mdds, out=np.zeros_like(avg_ls_returns), where=avg_ls_mdds!=0)
            avg_bh_romads = np.divide(avg_bh_returns, avg_bh_mdds)

            # rimuovo i nan dai romads
            avg_ls_romads = np.around(np.nan_to_num(avg_ls_romads), decimals=3)
            avg_lh_romads = np.around(np.nan_to_num(avg_lh_romads), decimals=3)
            avg_sh_romads = np.around(np.nan_to_num(avg_sh_romads), decimals=3)
            avg_sh_romads[~np.isfinite(avg_sh_romads)] = 0

            avg_longs_precisions = np.around(np.average(all_longs_precisions, axis=0), decimals=3)
            avg_shorts_precisions = np.around(np.average(all_shorts_precisions, axis=0), decimals=3)

            avg_label_longs_coverage = np.around(np.average(all_labels_longs_coverage, axis=0), decimals=3)
            avg_label_shorts_coverage = np.around(np.average(all_labels_shorts_coverage, axis=0), decimals=3)

            # NUOVO TEST AVG POC
            avg_longs_poc = np.around(np.divide(avg_longs_precisions, avg_label_longs_coverage), decimals=3)
            avg_shorts_poc = np.around(np.divide(avg_shorts_precisions, avg_label_shorts_coverage), decimals=3)

            avg_longs_poc = (avg_longs_poc - 1 ) * 100
            
            for avg_id, avg in enumerate(avg_longs_poc):
                if avg_longs_poc[avg_id] < -30:
                    avg_longs_poc[avg_id] = -30
                if avg_longs_poc[avg_id] > 30:
                    avg_longs_poc[avg_id] = 30

            avg_shorts_poc = (avg_shorts_poc - 1 ) * 100

            for avg_id, avg in enumerate(avg_shorts_poc):
                if avg_shorts_poc[avg_id] < -30:
                    avg_shorts_poc[avg_id] = -30
                if avg_shorts_poc[avg_id] > 30:
                    avg_shorts_poc[avg_id] = 30

            avg_long_operations = np.average(all_long_operations, axis=0)
            avg_short_operations= np.average(all_short_operations, axis=0)
            avg_hold_operations = np.average(all_hold_operations, axis=0)

            avg_json = {
                "ls_returns": avg_ls_returns.tolist(),
                "lh_returns": avg_lh_returns.tolist(),
                "sh_returns": avg_sh_returns.tolist(),
                "bh_returns": avg_bh_returns.tolist(), # BH

                "ls_romads": avg_ls_romads.tolist(),
                "lh_romads": avg_lh_romads.tolist(),
                "sh_romads": avg_sh_romads.tolist(),
                "bh_romads": avg_bh_romads.tolist(), # BH

                "ls_mdds": avg_ls_mdds.tolist(),
                "lh_mdds": avg_lh_mdds.tolist(),
                "sh_mdds": avg_sh_mdds.tolist(),
                "bh_mdds": avg_bh_mdds.tolist(), # BH

                "longs_precisions": avg_longs_precisions.tolist(),
                "shorts_precisions": avg_shorts_precisions.tolist(),
                "longs_label_coverage": avg_label_longs_coverage.tolist(),
                "shorts_label_coverage": avg_label_shorts_coverage.tolist(),

                "long_operations": avg_long_operations.tolist(),
                "short_operations": avg_short_operations.tolist(),
                "hold_operations": avg_hold_operations.tolist(),

                "longs_poc": avg_longs_poc.tolist(),
                "shorts_poc": avg_shorts_poc.tolist(),
            }

            avg_output_path = self.experiment_original_path + self.iperparameters['experiment_name'] + '/calculated_metrics/' + type + '/average/' 
            
            create_folder(avg_output_path)

            with open(avg_output_path + 'walk_' + str(index_walk) + '.json', 'w') as json_file:
                json.dump(avg_json, json_file, indent=4)
                #json.dump(net_json, json_file)
            
            print(self.iperparameters['experiment_name'] + ' | ' + type + " - Salvate le metriche AVG per walk n° ", index_walk)

            do_plot(metrics=avg_json, walk=index_walk, epochs=len(epochs_list), main_path=self.experiment_original_path, experiment_name=self.iperparameters['experiment_name'], net='average', type=type)

    '''
    '
    '''
    def generate_json_loss(self):
        for index_walk in range(0, len(self.walks_list)):
            avg_training_loss = []
            avg_validation_loss = []
            avg_test_loss = []

            walk_str = 'walk_' + str(index_walk)
            date_list, epochs_list = self.get_date_epochs_walk(path=self.original_predictions_validation_folder, walk=walk_str)

            # RETURNS
            all_training_loss = np.zeros(shape=(len(self.nets_list), len(epochs_list)))
            all_validation_loss = np.zeros(shape=(len(self.nets_list), len(epochs_list)))
            all_test_loss = np.zeros(shape=(len(self.nets_list), len(epochs_list)))

            for index_net in range(0, len(self.nets_list)):
                net = 'net_' + str(index_net) + '.csv'
                df = pd.read_csv(self.experiment_original_path + self.iperparameters['experiment_name'] + '/predictions/loss_during_training/walk_' + str(index_walk) + '/' + net)

                json_log = {
                    'training_loss': df['training_loss'].tolist(),
                    'validation_loss': df['validation_loss'].tolist(),
                    'test_loss': df['test_loss'].tolist()
                }

                output_path = self.experiment_original_path + self.iperparameters['experiment_name'] + '/calculated_metrics/loss/walk_' + str(index_walk) + '/' 

                create_folder(output_path)

                with open(output_path + 'net_' + str(index_net) + '.json', 'w') as json_file:
                    json.dump(json_log, json_file, indent=4)

                all_training_loss[index_net] = df['training_loss'].tolist()
                all_validation_loss[index_net] = df['validation_loss'].tolist()
                all_test_loss[index_net] = df['test_loss'].tolist()
            
            avg_training_loss = np.around(np.average(all_training_loss, axis=0), decimals=3)
            avg_validation_loss = np.around(np.average(all_validation_loss, axis=0), decimals=3)
            avg_test_loss = np.around(np.average(all_test_loss, axis=0), decimals=3)

            avg_json_log = {
                    'training_loss': avg_training_loss.tolist(),
                    'validation_loss': avg_validation_loss.tolist(),
                    'test_loss': avg_test_loss.tolist()
                }

            output_path = self.experiment_original_path + self.iperparameters['experiment_name'] + '/calculated_metrics/loss/average/' 

            create_folder(output_path)

            with open(output_path + 'walk_' + str(index_walk) + '.json', 'w') as json_file:
                json.dump(avg_json_log, json_file, indent=4)
    
    '''
    '
    '''
    def calculate_custom_metrics(self, penalty=32, stop_loss=1000):
        
        t1 = threading.Thread(target=self.generate_custom_metrics, args=(['validation', penalty, stop_loss]))
        t2 = threading.Thread(target=self.generate_custom_metrics, args=(['test', penalty, stop_loss]))
        t3 = threading.Thread(target=self.generate_json_loss)

        t1.start()
        t2.start()
        t3.start()

        t1.join()
        t2.join()
        t3.join()
