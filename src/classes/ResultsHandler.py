import os
import numpy as np
import pandas as pd
from datetime import timedelta
from os import listdir
from os.path import isfile, join
from classes.Market import Market
from classes.Measures import Measures
from classes.Utils import natural_keys, df_date_merger, do_plot, revert_probabilities, df_date_merger_binary
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
from operator import add

# per togliere la notazione esponenziale di numpy
np.set_printoptions(suppress=True)



class ResultsHandler:

    platform = platform.platform()

    experiment_name = ''

    #experiment_original_path = 'C:/Users/Utente/Documents/GitHub/PhD-Market-Nets/experiments/Esperimenti Vacanze/'
    #experiment_original_path = 'C:/Users/Utente/Documents/GitHub/PhD-Market-Nets/experiments/'  # locale
    
    if platform == 'Linux-4.15.0-45-generic-x86_64-with-Ubuntu-16.04-xenial': 
        experiment_original_path = '/media/unica/HDD 9TB Raid0 - 1/experiments/'
    else: 
        experiment_original_path = '../experiments/'  # locale
        experiment_original_path = 'D:/PhD-Market-Nets/experiments/'  # locale
        #experiment_original_path = '/var/www/html/experiments/' # server hawkeye


    experiment_base_path = ''
    prediction_base_folder = ''
    original_predictions_validation_folder = ''
    original_predictions_test_folder = ''
    ensemble_base_folder = ''

    final_decision_folder = ''
    valid_final_decision_folder = ''

    final_decision_folder_alg4 = ''
    final_decision_per_walk = ''

    selection_folder = ''

    dataset = ''

    walks_list = []
    nets_list = []

    def __init__(self, experiment_name, experiment_path=""):

        # sovrascrivo la basepath dell'esperimento qualora lo passassi come parametro
        if experiment_path != "":
            self.experiment_original_path = experiment_path

        self.experiment_base_path = self.experiment_original_path + experiment_name + '/'
        self.experiment_name = experiment_name
        self.prediction_base_folder = self.experiment_original_path + experiment_name + '/predictions/'

        self.original_predictions_validation_folder = self.experiment_original_path + experiment_name + '/predictions/predictions_during_training/validation/'
        self.original_predictions_test_folder = self.experiment_original_path + experiment_name + '/predictions/predictions_during_training/test/'
        self.ensemble_base_folder = self.experiment_original_path + experiment_name + '/predictions/predictions_post_ensemble/'
        
        self.final_decision_folder = self.experiment_original_path + experiment_name + '/predictions/final_decisions/'
        self.valid_final_decision_folder = self.experiment_original_path + experiment_name + '/predictions/valid_final_decisions/'
        self.final_decision_per_walk = self.experiment_original_path + experiment_name + '/predictions/final_decisions_per_walk/'
        self.final_decision_folder_alg4 = self.experiment_original_path + experiment_name + '/predictions/final_decision_alg4/'

    
        self.selection_folder = self.experiment_original_path + experiment_name + '/selection/'

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
        # DA FIXARE
        #if os.path.isfile(self.original_predictions_validation_folder):
        #    print("sono dentro")
        self.walks_list = os.listdir(self.original_predictions_validation_folder)
        self.walks_list.sort(key=natural_keys)

        #    if os.path.isfile(self.original_predictions_validation_folder + self.walks_list[0]):
        self.nets_list = os.listdir(self.original_predictions_validation_folder + self.walks_list[0])
        self.nets_list.sort(key=natural_keys)

        #self.walks_list.sort(key=natural_keys)
        #self.nets_list.sort(key=natural_keys)

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
    ' short, hold, long
    '''
    def generate_triple(self, df):
        m = pd.DataFrame()
        m['ensemble'] = df.eq(0).sum(1).astype(str) + ';' + df.eq(1).sum(1).astype(str) + ';' + df.eq(2).sum(1).astype(str) 
        
        return m

    '''
    '
    '''
    def ensemble_magg_classic(self, df, thr):
        short = (df.eq(0).sum(1) / df.shape[1]).gt(thr)
        hold = (df.eq(1).sum(1) / df.shape[1]).gt(thr)
        long = (df.eq(2).sum(1) / df.shape[1]).gt(thr)
        
        long_1 = df.eq(2).sum(1) > df.eq(0).sum(1)
        long_2 = df.eq(2).sum(1) > df.eq(1).sum(1)

        short_1 = df.eq(0).sum(1) > df.eq(1).sum(1)
        short_2 = df.eq(0).sum(1) > df.eq(2).sum(1)

        df = pd.DataFrame(np.select([long & long_1 & long_2, short & short_1 & short_2], [2, 0], default=1), index=df.index, columns=['decision'])
        return df

    '''
    ' Calcola l'ensemble exclusive (long) sulla colonna con le triple 
    '''
    def ensemble_exclusive_classic(self, df, thr):
        jolly_number = int((len(self.nets_list) * (thr * 100) / 100))

        long = df.eq(2).sum(1) > jolly_number
        
        df = pd.DataFrame(np.select([long], [2], default=1), index=df.index, columns=['decision'])

        return df

    '''
    '
    '''
    def ensemble_exclusive_short_classic(self, df, thr):
        jolly_number = int((len(self.nets_list) * (thr * 100) / 100))

        short = df.eq(0).sum(1) > jolly_number

        df = pd.DataFrame(np.select([short], [0], default=1), index=df.index, columns=['decision'])

        return df

    '''
    ' Calcola l'ensemable sulle colonne (reti) con una % di agreement
    ' Inserendo nel calcolo della percentuale anche il numero di hold
    ' Il calcolo ora viene fatto sul csv delle triple, quindi 
    ' ogni colonna contiene il numero totale di operazioni per tutte le 
    ' reti per quell'epoca
    '''
    def ensemble_magg(self, df, thr):
        for column in df:   
            n_short = (df[column].str.split(';').str[0]).astype(int)
            n_hold = (df[column].str.split(';').str[1]).astype(int)
            n_long = (df[column].str.split(';').str[2]).astype(int)

            short = (n_short / len(self.nets_list)).gt(thr)
            hold = (n_hold / len(self.nets_list)).gt(thr)
            long = (n_long / len(self.nets_list)).gt(thr)
            
            long_1 = n_long > n_short
            long_2 = n_long > n_hold

            short_1 = n_short> n_hold
            short_2 = n_short > n_long

            # DECOMMENTARE QUESTA RIGA PER ESCLUDERE LE HOLD DAL CALCOLO
            #m = pd.DataFrame(np.select([n_short, n_long], [0, 2], 1), index=df.index, columns=['ensemble'])
            df[column] = np.select([long & long_1 & long_2, short & short_1 & short_2], [2, 0], 1)

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
    ' Calcola l'ensemble exclusive (long) sulla colonna con le triple 
    '''
    def ensemble_exclusive(self, df, thr):
        jolly_number = int((len(self.nets_list) * (thr * 100) / 100))

        for column in df:          
            df[column]  = df[column].apply(lambda x: 2 if int(x.split(';')[2]) > jolly_number else 1)

        return df

    '''
    ' Calcola l'ensemble exclusive (short) sulla colonna con le triple 
    '''
    def ensemble_exclusive_short(self, df, thr):
        jolly_number = int((len(self.nets_list) * (thr * 100) / 100))

        for column in df:          
            df[column]  = df[column].apply(lambda x: 0 if int(x.split(';')[0]) > jolly_number else 1)

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

        m = pd.DataFrame(np.select([m1 & m2, m1 & m3], [2, 0], default=1), index=df.index, columns=['ensemble'])

        return m

    '''
    ' Calcola gli ensemble a maggioranza
    ' @deprecated
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

                    if remove_nets==True:
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

                if remove_nets==True:
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
                create_folder(output_path + 'ensemble_exclusive_short/' + str(thr) + '/')

                #long
                df_ensemble_exclusive = self.ensemble_exclusive(df=df_walk.copy(), thr=thr)
                df_ensemble_exclusive['date_time'] = date_time
                df_ensemble_exclusive.to_csv(output_path + 'ensemble_exclusive/' + str(thr) + '/' + walk + '.csv', header=True, index=False)

                #long
                df_ensemble_exclusive_short = self.ensemble_exclusive_short(df=df_walk.copy(), thr=thr)
                df_ensemble_exclusive_short['date_time'] = date_time
                df_ensemble_exclusive_short.to_csv(output_path + 'ensemble_exclusive_short/' + str(thr) + '/' + walk + '.csv', header=True, index=False)

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
    ' prob_mode = none | long_short_thr
    '''
    def generate_triple_csv(self, remove_nets=False, prob_mode='none', prob_thr=0.98):
        
        t1 = threading.Thread(target=self.calculate_triple_csv, args=(['validation', prob_mode, prob_thr]))
        t2 = threading.Thread(target=self.calculate_triple_csv, args=(['test', prob_mode, prob_thr]))        
        
        t1.start()
        t2.start()

        t1.join()
        t2.join()
    
    '''
    ' Nuovo ensemble che genera un csv di triple per ogni epoca / giorno
    '''
    def calculate_triple_csv(self, type, prob_mode='none', prob_thr=0.98):
        # A seconda del set creo input e output path
        if type == 'validation':
            input_path = self.original_predictions_validation_folder
            output_path = self.prediction_base_folder + 'triple_csv/validation/'

        if type == 'test':
            input_path = self.original_predictions_test_folder
            output_path = self.prediction_base_folder + 'triple_csv/test/'

        if not os.path.isdir(output_path):
            os.makedirs(output_path)

        print(self.walks_list)
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

                    if self.iperparameters['use_probabilities'] == False:
                        df_net[net] = df_predizioni[epoch]
                    
                    if self.iperparameters['use_probabilities'] == True:
                        df_net[net] = df_predizioni[epoch].apply(lambda x: revert_probabilities(x = x, mode=prob_mode, thr=prob_thr))

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
        
        generic_thr_for_print = 0

        remove_nets_str = 'con-rimozione-reti/'

        if remove_nets == False:
            remove_nets_str = 'senza-rimozione-reti/'

        if type=='ensemble_magg':
            print("Generating final decision | type:", type, "| thr:", thr_ensemble_magg, "| Epoch selection policy:", epoch_selection_policy)
            valid_output_path = self.valid_final_decision_folder + 'ensemble_magg/' + remove_nets_str + epoch_selection_policy + '/'
            test_output_path = self.final_decision_folder + 'ensemble_magg/' + remove_nets_str + epoch_selection_policy + '/'
            valid_output_path_per_walk = self.final_decision_per_walk + 'validation/ensemble_magg/' + remove_nets_str + epoch_selection_policy + '/'
            test_output_path_per_walk = self.final_decision_per_walk + 'test/ensemble_magg/' + remove_nets_str + epoch_selection_policy + '/'

            validation_input_path = self.ensemble_base_folder + remove_nets_str + 'validation/ensemble_magg/' + str(thr_ensemble_magg) + '/'
            test_input_path = self.ensemble_base_folder  + remove_nets_str + 'test/ensemble_magg/' + str(thr_ensemble_magg) + '/'
            generic_thr_for_print = thr_ensemble_magg

        if type=='ensemble_el_long_only':
            print("Generating final decision | type:", type, "| thr:", thr_ensemble_elimination, "| Epoch selection policy:", epoch_selection_policy)
            valid_output_path = self.valid_final_decision_folder + 'ensemble_el_longonly/' + remove_nets_str + epoch_selection_policy + '/'
            test_output_path = self.final_decision_folder + 'ensemble_el_longonly/' + remove_nets_str + epoch_selection_policy + '/'
            valid_output_path_per_walk = self.final_decision_per_walk + 'validation/ensemble_el_longonly/' + remove_nets_str + epoch_selection_policy + '/'
            test_output_path_per_walk = self.final_decision_per_walk + 'test/ensemble_el_longonly/' + remove_nets_str + epoch_selection_policy + '/'

            validation_input_path = self.ensemble_base_folder + remove_nets_str + 'validation/ensemble_el/' #+ str(thr_ensemble_elimination) + '/'
            test_input_path = self.ensemble_base_folder + remove_nets_str + 'test/ensemble_el/' #+ str(thr_ensemble_elimination) + '/'
            generic_thr_for_print = thr_ensemble_elimination

        if type == 'ensemble_exclusive': 
            print("Generating final decision | type:", type, "| thr:", thr_ensemble_exclusive, "| Epoch selection policy:", epoch_selection_policy)
            valid_output_path = self.valid_final_decision_folder + 'ensemble_exclusive/' + remove_nets_str + epoch_selection_policy + '/'
            test_output_path = self.final_decision_folder + 'ensemble_exclusive/' + remove_nets_str + epoch_selection_policy + '/'

            valid_output_path_per_walk = self.final_decision_per_walk + 'validation/ensemble_exclusive/' + remove_nets_str + epoch_selection_policy + '/'
            test_output_path_per_walk = self.final_decision_per_walk + 'test/ensemble_exclusive/' + remove_nets_str + epoch_selection_policy + '/'

            validation_input_path = self.ensemble_base_folder + remove_nets_str + 'validation/ensemble_exclusive/' + str(thr_ensemble_exclusive) + '/'
            test_input_path = self.ensemble_base_folder + remove_nets_str + 'test/ensemble_exclusive/' + str(thr_ensemble_exclusive) + '/'
            generic_thr_for_print = thr_ensemble_exclusive

        if type == 'ensemble_exclusive_short': 
            print("Generating final decision | type:", type, "| thr:", thr_ensemble_exclusive, "| Epoch selection policy:", epoch_selection_policy)
            valid_output_path = self.valid_final_decision_folder + 'ensemble_exclusive_short/' + remove_nets_str + epoch_selection_policy + '/'
            test_output_path = self.final_decision_folder + 'ensemble_exclusive_short/' + remove_nets_str + epoch_selection_policy + '/'

            valid_output_path_per_walk = self.final_decision_per_walk + 'validation/ensemble_exclusive_short/' + remove_nets_str + epoch_selection_policy + '/'
            test_output_path_per_walk = self.final_decision_per_walk + 'test/ensemble_exclusive_short/' + remove_nets_str + epoch_selection_policy + '/'

            validation_input_path = self.ensemble_base_folder + remove_nets_str + 'validation/ensemble_exclusive_short/' + str(thr_ensemble_exclusive) + '/'
            test_input_path = self.ensemble_base_folder + remove_nets_str + 'test/ensemble_exclusive_short/' + str(thr_ensemble_exclusive) + '/'
            generic_thr_for_print = thr_ensemble_exclusive

        # creo la path finale     
        if not os.path.isdir(valid_output_path):
            os.makedirs(valid_output_path)
  
        if not os.path.isdir(test_output_path):
            os.makedirs(test_output_path)

        df_global_valid = pd.DataFrame(columns=['date_time', 'close', 'delta_current_day', 'delta_next_day', 'label'])
        df_global_test = pd.DataFrame(columns=['date_time', 'close', 'delta_current_day', 'delta_next_day', 'label'])

        dataset = Market(dataset=self.dataset)
        dataset_label = dataset.get_label(freq='1d', columns=['open', 'close', 'delta_current_day', 'delta_next_day', 'high', 'low'], thr=self.iperparameters['hold_labeling'])
        dataset_label = dataset_label.reset_index()
        dataset_label['date_time'] = dataset_label['date_time'].astype(str)

        for index_walk, walk in enumerate(self.walks_list):
            full_valid_output_path_per_walk = valid_output_path_per_walk + 'walk_' + str(index_walk) + '/' 
            full_test_output_path_per_walk = test_output_path_per_walk + 'walk_' + str(index_walk) + '/' 

            # creo le cartelle per i file di decisione walk per walk  
            if not os.path.isdir(full_valid_output_path_per_walk):
                os.makedirs(full_valid_output_path_per_walk)
  
            if not os.path.isdir(full_test_output_path_per_walk):
                os.makedirs(full_test_output_path_per_walk)

            df_ensemble_val = pd.read_csv(validation_input_path + walk + '.csv')
            df_merge_with_label = pd.merge(df_ensemble_val, dataset_label, how="inner")
            
            val_idx, romad, return_value, mdd_value = self.get_max_idx_from_validation(df_merge_with_label=df_merge_with_label, validation_thr=validation_thr, metric='romad', epoch_selection_policy=epoch_selection_policy, stop_loss=stop_loss, penalty=penalty)
            
            #print(romad)
            #input()
            #DEBUG IS1 E IS2
            #val_idx = 400 # DEBUG
            #df_calcoli_debug = df_merge_with_label[['epoch_' + str(val_idx), 'date_time', 'delta_next_day', 'low', 'open', 'close']]
            #df_calcoli_debug = df_calcoli_debug.rename(columns={'epoch_' + str(val_idx) : 'decision'})
            #vlong, vshort, vhold, vgeneral = Measures.get_precision_count_coverage(df=df_calcoli_debug, multiplier=50, delta_to_use='delta_next_day', stop_loss=1000, penalty=25)

            # Open a file with access mode 'a'
            with open(self.experiment_base_path + "selection_epoch_log.txt", "a") as file_object:
                string = ("Ensemble: " +  type + " con soglia: " + str(generic_thr_for_print) + " | Walk n° " + str(index_walk) + " | Politica selezione epoca: " +\
                     epoch_selection_policy + " - Metrica di selezione: " + validation_metric + " | ID validation scelto: " + str(val_idx) + " con romad: " + str(romad) + "\n")
                file_object.write(string)

            # concatenazione valid
            df_valid = df_merge_with_label[['epoch_' + str(val_idx), 'open', 'close', 'delta_current_day', 'delta_next_day', 'date_time', 'high', 'low']]
            df_valid = df_valid.rename(columns={"epoch_" + str(val_idx): "decision"})

            # concatenazione test
            df_test = pd.read_csv(test_input_path + walk + '.csv')
            # mergio con le label, così ho un subset del df con le date che mi servono e la predizione
            df_test = pd.merge(df_test, dataset_label, how="inner")
            #df_merge_with_label = df_merge_with_label.set_index('index')
            df_test = df_test[['epoch_' + str(val_idx), 'open', 'close', 'delta_current_day', 'delta_next_day', 'date_time', 'high', 'low']]
            df_test = df_test.rename(columns={"epoch_" + str(val_idx): "decision"})

            # salvo i csv per walk 
            if type == 'ensemble_el_longonly':
                df_valid.to_csv(full_valid_output_path_per_walk + 'decisions_ensemble_el_long_only_' + str(thr_ensemble_elimination) + '.csv', header=True, index=False)
                df_test.to_csv(full_test_output_path_per_walk + 'decisions_ensemble_el_long_only_' + str(thr_ensemble_elimination) + '.csv', header=True, index=False)

            if type == 'ensemble_exclusive': 
                df_valid.to_csv(full_valid_output_path_per_walk + 'decisions_ensemble_exclusive_' + str(thr_ensemble_exclusive) + '.csv', header=True, index=False)
                df_test.to_csv(full_test_output_path_per_walk + 'decisions_ensemble_exclusive_' + str(thr_ensemble_exclusive) + '.csv', header=True, index=False)

            if type == 'ensemble_exclusive_short': 
                df_valid.to_csv(full_valid_output_path_per_walk + 'decisions_ensemble_exclusive_short_' + str(thr_ensemble_exclusive) + '.csv', header=True, index=False)
                df_test.to_csv(full_test_output_path_per_walk + 'decisions_ensemble_exclusive_short_' + str(thr_ensemble_exclusive) + '.csv', header=True, index=False)

            if type == 'ensemble_magg': 
                df_valid.to_csv(full_valid_output_path_per_walk + 'decisions_ensemble_magg_' + str(thr_ensemble_magg) + '.csv', header=True, index=False)
                df_test.to_csv(full_test_output_path_per_walk + 'decisions_ensemble_magg_' + str(thr_ensemble_magg) + '.csv', header=True, index=False)

            df_global_valid = pd.concat([df_global_valid, df_valid], sort=True)
            df_global_test = pd.concat([df_global_test, df_test], sort=True)

            #tlong, tshort, thold, tgeneral = Measures.get_precision_count_coverage(df=df_test, multiplier=50, delta_to_use='delta_next_day', stop_loss=1000, penalty=25)
            #print("Walk n°", index_walk, " Epoca migliore selezionata: ", val_idx, " | Soglia Ensemble Exclusive:", thr_ensemble_exclusive, "Valid Long Coverage:", vlong['coverage'], " Test Long Coverage:", tlong['coverage'])

       
        df_global_valid = df_global_valid.drop_duplicates(subset='date_time', keep="first")
        df_global_test = df_global_test.drop_duplicates(subset='date_time', keep="first")

        # valid
        df_global_valid['date_time'] = df_global_valid['date_time'].shift(-1)
        #df_global_valid = df_global_valid.drop(columns=['close', 'delta'], axis=1)
        df_global_valid = df_global_valid[['date_time', 'decision']]
        df_global_valid = df_global_valid.drop(df_global_valid.index[0])
        df_global_valid = df_date_merger(df=df_global_valid, columns=['open', 'close', 'high', 'low', 'delta_current_day', 'delta_next_day'], dataset=self.iperparameters['predictions_dataset'], thr_hold=self.iperparameters['hold_labeling'])
        df_global_valid['decision'] = df_global_valid['decision'].astype(int)

        # test
        df_global_test['date_time'] = df_global_test['date_time'].shift(-1)
        #df_global_test = df_global_test.drop(columns=['close', 'delta'], axis=1)
        df_global_test = df_global_test[['date_time', 'decision']]
        df_global_test = df_global_test.drop(df_global_test.index[0])
        df_global_test = df_date_merger(df=df_global_test, columns=['open', 'close', 'high', 'low', 'delta_current_day', 'delta_next_day'], dataset=self.iperparameters['predictions_dataset'], thr_hold=self.iperparameters['hold_labeling'])
        df_global_test['decision'] = df_global_test['decision'].astype(int)

        #ttlong, ttshort, tthold, ttgeneral = Measures.get_precision_count_coverage(df=df_global_test, multiplier=50, delta_to_use='delta_next_day', stop_loss=1000, penalty=25)
        #print("% OPERAZIONI SUL TEST SET TOTALE: ", ttlong['coverage'])

      
        if type == 'ensemble_el_longonly':
            df_global_valid.to_csv(valid_output_path + 'decisions_ensemble_el_long_only_' + str(thr_ensemble_elimination) + '.csv', header=True, index=False)
            df_global_test.to_csv(test_output_path + 'decisions_ensemble_el_long_only_' + str(thr_ensemble_elimination) + '.csv', header=True, index=False)

        if type == 'ensemble_exclusive': 
            df_global_valid.to_csv(valid_output_path + 'decisions_ensemble_exclusive_' + str(thr_ensemble_exclusive) + '.csv', header=True, index=False)
            df_global_test.to_csv(test_output_path + 'decisions_ensemble_exclusive_' + str(thr_ensemble_exclusive) + '.csv', header=True, index=False)

        if type == 'ensemble_exclusive_short': 
            df_global_valid.to_csv(valid_output_path + 'decisions_ensemble_exclusive_short_' + str(thr_ensemble_exclusive) + '.csv', header=True, index=False)
            df_global_test.to_csv(test_output_path + 'decisions_ensemble_exclusive_short_' + str(thr_ensemble_exclusive) + '.csv', header=True, index=False)

        if type == 'ensemble_magg': 
            df_global_valid.to_csv(valid_output_path + 'decisions_ensemble_magg_' + str(thr_ensemble_magg) + '.csv', header=True, index=False)
            df_global_test.to_csv(test_output_path + 'decisions_ensemble_magg_' + str(thr_ensemble_magg) + '.csv', header=True, index=False)

    '''
    '
    '''
    def get_csv_alg4(self, validation_thr=15, validation_metric='romad', epoch_selection_policy='long_short',  stop_loss=1000, penalty=32):

        for index_walk, walk in enumerate(self.walks_list):
            df_walk_valid = pd.DataFrame()
            df_walk_test = pd.DataFrame()
            walk_path = self.prediction_base_folder + 'predictions_after_alg3/'

            for index_net, net in enumerate(self.nets_list): 
                
                input_validation_path = self.original_predictions_validation_folder + walk + '/' + net
                input_test_path = self.original_predictions_test_folder + walk + '/' + net 

                df_valid = pd.read_csv(input_validation_path)
                df_test = pd.read_csv(input_test_path)

                #rimuovo la colonna unnamed: 0
                df_valid = df_valid.drop(df_valid.columns[0], axis=1)
                df_valid = df_date_merger(df=df_valid, columns=['delta_next_day', 'delta_current_day', 'high', 'low', 'open', 'close'], thr_hold=self.iperparameters['hold_labeling'], dataset=self.iperparameters['predictions_dataset'])
                
                val_idx, romad, return_value, mdd_value = self.get_max_idx_from_validation(df_merge_with_label=df_valid, validation_thr=validation_thr, metric=validation_metric, epoch_selection_policy=epoch_selection_policy, stop_loss=stop_loss, penalty=penalty)

                if index_net == 0:
                    df_walk_valid['date_time'] = df_valid['date_time'].tolist()
                    df_walk_test['date_time'] = df_test['date_time'].tolist()

                df_walk_valid['net_' + str(index_net) + '_epoch_' + str(val_idx)] = df_valid['epoch_' + str(val_idx)]
                df_walk_test['net_' + str(index_net) + '_epoch_' + str(val_idx)] = df_test['epoch_' + str(val_idx)]

                #print("Walk n", index_walk, "- Rete n", index_net, "- Id selezionato:", val_idx)
            
            output_validation_path = walk_path + 'validation/' + epoch_selection_policy + '/'
            output_test_path = walk_path + 'test/' + epoch_selection_policy + '/'

            if not os.path.isdir(output_validation_path):
                create_folder(output_validation_path)

            if not os.path.isdir(output_test_path):
                create_folder(output_test_path)

            df_walk_valid.to_csv(output_validation_path + walk + '.csv', header=True, index=False)
            df_walk_test.to_csv(output_test_path + walk + '.csv', header=True, index=False)

       
    '''
    ' 
    '''
    def get_final_decision_alg4(self, ensemble_type='ensemble_magg', thrs=[], remove_nets=False, epoch_selection_policy='long_only'):
        
        if remove_nets == False:
            remove_nets_str = 'senza-rimozione-reti/'

        test_input_path = self.prediction_base_folder + 'predictions_after_alg3/test/' + epoch_selection_policy + '/'

        

        for thr in thrs:
            print("[ALG4]Calcolo file decisioni finali per", ensemble_type, "soglia:", thr)
            df_global = pd.DataFrame()

            
            test_output_path = self.final_decision_folder_alg4 + ensemble_type + '/' + remove_nets_str + '/' + epoch_selection_policy +  '/'


            if not os.path.isdir(test_output_path):
                create_folder(test_output_path)

            for index_walk, walk in enumerate(self.walks_list):
                df_tmp = pd.read_csv (test_input_path + walk + '.csv')
                dates_list = df_tmp['date_time'].tolist()
                df_tmp = df_tmp.drop(columns=['date_time']) 

                if ensemble_type == 'ensemble_magg': 
                    df_ensemble = self.ensemble_magg_classic(df=df_tmp, thr=thr)

                if ensemble_type == 'ensemble_exclusive': 
                    df_ensemble = self.ensemble_exclusive_classic(df=df_tmp, thr=thr)

                if ensemble_type == 'ensemble_exclusive_short': 
                    df_ensemble = self.ensemble_exclusive_short_classic(df=df_tmp, thr=thr)

                df_ensemble['date_time'] = dates_list
                df_global = pd.concat([df_global, df_ensemble])

            df_global['date_time'] = df_global['date_time'].shift(-1)
            #df_global_test = df_global_test.drop(columns=['close', 'delta'], axis=1)
            df_global = df_global[['date_time', 'decision']]
            df_global = df_date_merger(df=df_global, columns=['open', 'close', 'high', 'low', 'delta_current_day', 'delta_next_day'], dataset=self.iperparameters['predictions_dataset'], thr_hold=self.iperparameters['hold_labeling'])
            df_global['decision'] = df_global['decision'].astype(int)
            
            filename = ''
            if ensemble_type == 'ensemble_el_longonly':
                filename = 'decisions_ensemble_el_long_only_' + str(thr) + '.csv'

            if ensemble_type == 'ensemble_exclusive': 
                filename = 'decisions_ensemble_exclusive_' + str(thr) + '.csv'

            if ensemble_type == 'ensemble_exclusive_short': 
                filename = 'decisions_ensemble_exclusive_short_' + str(thr) + '.csv'

            if ensemble_type == 'ensemble_magg': 
                filename = 'decisions_ensemble_magg_' + str(thr) + '.csv'

            df_global.to_csv(test_output_path + filename, header=True, index=False)




        '''
        if type == 'ensemble_magg':
            print("Generating final decision | type:", type, "| thr:", thr_ensemble_magg, "| Epoch selection policy:", epoch_selection_policy)


        if type == 'ensemble_el_long_only':
            print("Generating final decision | type:", type, "| thr:", thr_ensemble_elimination, "| Epoch selection policy:", epoch_selection_policy)

        if type == 'ensemble_exclusive': 
            print("Generating final decision | type:", type, "| thr:", thr_ensemble_exclusive, "| Epoch selection policy:", epoch_selection_policy)


        if type == 'ensemble_exclusive_short': 
            print("Generating final decision | type:", type, "| thr:", thr_ensemble_exclusive, "| Epoch selection policy:", epoch_selection_policy)


        # creo la path finale     
        if not os.path.isdir(valid_output_path):
            os.makedirs(valid_output_path)
  
        if not os.path.isdir(test_output_path):
            os.makedirs(test_output_path)
        '''

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

        if metric == 'return':
            mean_values = np.full(number_of_epochs, -100.0)
            array_values = return_epochs
        if metric == 'romad':
            mean_values = np.full(number_of_epochs, -1000000.0)
            array_values = romad_epochs
        if metric == 'mdd':
            mean_values = np.full(number_of_epochs, 1000000.0)
            array_values = mdd_epochs

        # RIMUOVO L'INTORNO HARD CODE
        #mean_values = array_values

        # COMMENTATO PER RIMUOVERE L'INTORNO HARD CODE
        for index_for, value in enumerate(array_values):
            # entro solo tra i valori compresi dalla soglia, se soglia = 15, parto dal 15° sino a i-15
            if index_for >= validation_thr and index_for <= (number_of_epochs - validation_thr):
                mean_values[index_for] = statistics.mean(array_values[index_for-validation_thr:index_for+validation_thr])
                #print("mean_values id", index_for, "valore: ", mean_values[index_for])

        # seleziono l'epoca con il return migliore a cui aggiungo l'intorno per selezionarlo int est
        if metric == 'mdd':
            epoch_index = np.argmin(mean_values) + 1
        else:
            epoch_index = np.argmax(mean_values) + 1
        
        
        #print(romad_epochs[:200])
        #print(mean_values[:200])

        print("Epoca selezionata:", epoch_index)
        print("Valore medio romad dell'intorno selezionato:", mean_values[epoch_index])
        print("Romad effettivo di quell'epoca:", romad_epochs[epoch_index])

        ''' COMMENTATO PER RIMUOVERE L'INTORNO HARD CODE'''
        sub_sample = array_values[epoch_index-validation_thr:epoch_index+validation_thr]
        #print(sub_sample)
        #index_alg = (np.abs(sub_sample-mean_values[epoch_index])).argmin() # ALG2
        index_alg = sub_sample.argmax() # ALG3
        index_original_array = epoch_index + index_alg - validation_thr
        
        #index_original_array = epoch_index

        #print("[ALG2] Epoca selezionata per vicinanza: ", index, "con romad:", sub_sample[index])
        #print("[ALG3] Epoca selezionata per romad maggiore: ", index_alg, "con romad:", sub_sample[index_alg])
        print("[ALG3] Epoca selezionata per romad maggiore: ", index_original_array, "con romad:", romad_epochs[index_original_array])
        print("")
        #input()
        
        return index_original_array, romad_epochs[index_original_array], return_epochs[index_original_array], mdd_epochs[index_original_array]
        #return epoch_index, romad_epochs[epoch_index], return_epochs[epoch_index], mdd_epochs[epoch_index]

    '''
    ' Leggo il file con le decisioni ultime
    ' Quindi calcolo tutte le metriche per quel CSV (unico per tutti gli walk)
    ' Utilizzo il delta_current_day poiché il vettore delle label è già 
    ' allineato al giorno corrente (giorno in cui deve venir fatta l'operazione)
    '''
    def get_results(self, ensemble_type='ensemble_magg', epoch_selection_policy='long_short', thr_ensemble_magg=0.35, thr_ensemble_exclusive=0.3, 
        thr_ensemble_elimination=0.3, remove_nets=False, penalty=32, stop_loss=1000, subsample=[], decision_folder='test', input_df=pd.DataFrame()):
        input_folder = ''

        if decision_folder == 'validation': 
            input_folder = self.valid_final_decision_folder

        if decision_folder == 'test_alg4': 
            input_folder = self.final_decision_folder_alg4
        else: 
            input_folder = self.final_decision_folder


        remove_nets_str = 'con-rimozione-reti/'

        if remove_nets == False:
            remove_nets_str = 'senza-rimozione-reti/'

        #ensemble magg
        if ensemble_type == 'ensemble_magg':
            df = pd.read_csv(input_folder + 'ensemble_magg/' + remove_nets_str + epoch_selection_policy + '/decisions_ensemble_magg_' + str(thr_ensemble_magg) + '.csv')

        if ensemble_type == 'ensemble_el_long_only':
            df = pd.read_csv(input_folder + 'ensemble_el_longonly/' + remove_nets_str + epoch_selection_policy + '/decisions_ensemble_el_long_only_' + str(thr_ensemble_elimination) + '.csv')

        if ensemble_type == 'ensemble_exclusive':
            df = pd.read_csv(input_folder + 'ensemble_exclusive/' + remove_nets_str + epoch_selection_policy + '/decisions_ensemble_exclusive_' + str(thr_ensemble_exclusive) + '.csv')
        
        if ensemble_type == 'ensemble_exclusive_short':
            df = pd.read_csv(input_folder + 'ensemble_exclusive_short/' + remove_nets_str + epoch_selection_policy + '/decisions_ensemble_exclusive_short_' + str(thr_ensemble_exclusive) + '.csv')
        
        # without ensemble
        if ensemble_type == 'without_ensemble':
            df = pd.read_csv(input_folder + 'without_ensemble/decisions_without_ensemble.csv')

        if ensemble_type == 'from_df':
            df = input_df 

        close = df['close'].tolist()
        dates = df['date_time'].tolist()

        
        # BLOCCO DI CODICE IS1 - IS2
        #if subsample != []:
        #    df = Market.get_df_by_data_range(df=df.copy(), start_date=subsample[0], end_date=subsample[1])
        #    close = df['close'].tolist()
        #    dates = df['date_time'].tolist()
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
            #    input("Salvato " + ensemble_type + " - " + str(thr_ensemble_exclusive))
        
        # calcolo il b&h
        bh_equity_line, bh_global_return, bh_mdd, bh_romad, i, j = Measures.get_return_mdd_romad_bh(close=close, multiplier=self.iperparameters['return_multiplier'])
        bh_intraday_equity_line, bh_intraday_global_return, bh_intraday_mdd, bh_intraday_romad, bh_intraday_i, bh_intraday_j = Measures.get_equity_return_mdd_romad(df=df.copy(), multiplier=self.iperparameters['return_multiplier'], type='bh_long', penalty=penalty, stop_loss=stop_loss, delta_to_use='delta_current_day')

        # calcolo tutte le info per long + short, long + hold e short + hold
        long_info, short_info, hold_info, general_info = Measures.get_precision_count_coverage(df=df.copy(), penalty=penalty, stop_loss=stop_loss, multiplier=self.iperparameters['return_multiplier'], delta_to_use='delta_current_day')
        
        coverage_per_months = Measures.get_coverage_per_month(df=df.copy())
        
        #testtttt = Measures.get_avg_coverage_per_month(predictions_folder=self.original_predictions_test_folder, walks_list=self.walks_list, nets_list=self.nets_list, epochs=self.iperparameters['epochs'])

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

        bh_intraday_results = {
            "return": bh_intraday_global_return,
            "mdd": bh_intraday_mdd,
            "romad": bh_intraday_romad,
            "equity_line": bh_intraday_equity_line
        }


        general_info = {
            "long_info": long_info,
            "short_info": short_info,
            "hold_info": hold_info,
            "total_trade": general_info['total_trade'],
            "total_guessed_trade": general_info['total_guessed_trade'],
            "total_operation": general_info['total_operation'],
            "dates": dates,
            "coverage_per_months": coverage_per_months
        }

        return ls_results, l_results, s_results, bh_results, bh_intraday_results, general_info

    '''
    '
    '''
    def get_results_vix(self, thr, penalty=25, stop_loss=1000):
        input_folder = ''

        df = pd.read_csv(self.final_decision_folder + 'final_decision_' + str(thr) + '.csv')
        
        if 'close' not in df.columns:
            df = df_date_merger(df=df.copy(), thr_hold=self.iperparameters['hold_labeling'], dataset=self.iperparameters['predictions_dataset'], columns=['close', 'open', 'delta_current_day', 'delta_next_day', 'high', 'low'])

        # Fix per quando i csv erano generati dal 2003 ad oggi. I csv final_decision ora arrivano già allineati con le date giuste
        #df = Market.get_df_by_data_range(df=df.copy(), start_date=self.iperparameters['test_set'][0][0], end_date=self.iperparameters['test_set'][-1][-1])
        #df = Market.get_df_by_data_range(df=df.copy(), start_date="2011-01-03", end_date="2019-12-31")
        
        close = df['close'].tolist()
        dates = df['date_time'].tolist()
        
        # calcolo il b&h
        bh_equity_line, bh_global_return, bh_mdd, bh_romad, i, j = Measures.get_return_mdd_romad_bh(close=close, multiplier=self.iperparameters['return_multiplier'])
        bh_intraday_equity_line, bh_intraday_global_return, bh_intraday_mdd, bh_intraday_romad, bh_intraday_i, bh_intraday_j = Measures.get_equity_return_mdd_romad(df=df.copy(), multiplier=self.iperparameters['return_multiplier'], type='bh_long', penalty=penalty, stop_loss=stop_loss, delta_to_use='delta_current_day')

        # calcolo tutte le info per long + short, long + hold e short + hold
        long_info, short_info, hold_info, general_info = Measures.get_precision_count_coverage(df=df.copy(), penalty=penalty, stop_loss=stop_loss, multiplier=self.iperparameters['return_multiplier'], delta_to_use='delta_current_day')
        
        coverage_per_months = Measures.get_coverage_per_month(df=df.copy())
        
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

        bh_intraday_results = {
            "return": bh_intraday_global_return,
            "mdd": bh_intraday_mdd,
            "romad": bh_intraday_romad,
            "equity_line": bh_intraday_equity_line
        }


        general_info = {
            "long_info": long_info,
            "short_info": short_info,
            "hold_info": hold_info,
            "total_trade": general_info['total_trade'],
            "total_guessed_trade": general_info['total_guessed_trade'],
            "total_operation": general_info['total_operation'],
            "dates": dates,
            "coverage_per_months": coverage_per_months
        }

        return ls_results, l_results, s_results, bh_results, bh_intraday_results, general_info
   
    '''
    '
    '''
    def get_results_per_walk(self, ensemble_type='ensemble_magg', epoch_selection_policy='long_short', thr_ensemble_magg=0.35, thr_ensemble_exclusive=0.3, 
        thr_ensemble_elimination=0.3, remove_nets=False, penalty=32, stop_loss=1000, subsample=[], decision_folder='test', walk=0, get_coverage_per_month=True):
        input_folder = ''

        input_folder = self.final_decision_per_walk + decision_folder + '/'

        remove_nets_str = 'con-rimozione-reti/'

        if remove_nets == False:
            remove_nets_str = 'senza-rimozione-reti/'

        #ensemble magg
        if ensemble_type == 'ensemble_magg':
            full_path = input_folder + 'ensemble_magg/' + remove_nets_str + epoch_selection_policy + '/' + 'walk_' + str(walk) + '/'
            df = pd.read_csv(full_path + 'decisions_ensemble_magg_' + str(thr_ensemble_magg) + '.csv')

        if ensemble_type == 'ensemble_el_long_only':
            full_path = input_folder + 'ensemble_el_long_only/' + remove_nets_str + epoch_selection_policy + '/' + 'walk_' + str(walk) + '/'
            df = pd.read_csv(full_path + 'decisions_ensemble_el_long_only_' + str(thr_ensemble_elimination) + '.csv')

        if ensemble_type == 'ensemble_exclusive':
            full_path = input_folder + 'ensemble_exclusive/' + remove_nets_str + epoch_selection_policy + '/' + 'walk_' + str(walk) + '/'
            df = pd.read_csv(full_path + 'decisions_ensemble_exclusive_' + str(thr_ensemble_exclusive) + '.csv')
        
        if ensemble_type == 'ensemble_exclusive_short':
            full_path = input_folder + 'ensemble_exclusive_short/' + remove_nets_str + epoch_selection_policy + '/' + 'walk_' + str(walk) + '/'
            df = pd.read_csv(full_path + 'decisions_ensemble_exclusive_short_' + str(thr_ensemble_exclusive) + '.csv')

        close = df['close'].tolist()
        dates = df['date_time'].tolist()

        df = df_date_merger(df=df, thr_hold=self.iperparameters['hold_labeling'], dataset=self.iperparameters['predictions_dataset'])
        
        
        # calcolo il b&h
        bh_equity_line, bh_global_return, bh_mdd, bh_romad, i, j = Measures.get_return_mdd_romad_bh(close=close, multiplier=self.iperparameters['return_multiplier'])
        bh_intraday_equity_line, bh_intraday_global_return, bh_intraday_mdd, bh_intraday_romad, bh_intraday_i, bh_intraday_j = Measures.get_equity_return_mdd_romad(df=df.copy(), multiplier=self.iperparameters['return_multiplier'], type='bh_long', penalty=penalty, stop_loss=stop_loss, delta_to_use='delta_current_day')

        # calcolo tutte le info per long + short, long + hold e short + hold
        long_info, short_info, hold_info, general_info = Measures.get_precision_count_coverage(df=df.copy(), penalty=penalty, stop_loss=stop_loss, multiplier=self.iperparameters['return_multiplier'], delta_to_use='delta_current_day')
        
        if get_coverage_per_month == True:
            coverage_per_months = Measures.get_coverage_per_month(df=df.copy())
        else:
            coverage_per_months = []

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

        bh_intraday_results = {
            "return": bh_intraday_global_return,
            "mdd": bh_intraday_mdd,
            "romad": bh_intraday_romad,
            "equity_line": bh_intraday_equity_line
        }


        general_info = {
            "long_info": long_info,
            "short_info": short_info,
            "hold_info": hold_info,
            "total_trade": general_info['total_trade'],
            "total_guessed_trade": general_info['total_guessed_trade'],
            "total_operation": general_info['total_operation'],
            "total_coverage": general_info['total_coverage'],
            "dates": dates,
            "coverage_per_months": coverage_per_months
        }

        return ls_results, l_results, s_results, bh_results, bh_intraday_results, general_info
    
    
    '''
    ' Leggo tutte le metriche in output da un tipo di ensemble con get_results
    ' e salvo un file excel con tute le informazioni che mi servono
    ' metric = usato per i csv selection di sebastian
    '''
    def get_report_excel(self, report_name, epoch_selection_policy='long_short', thr=0.3, remove_nets=False, type='ensemble_magg', stop_loss=1000, penalty=32, subsample=[], 
        subfolder_is='', metric="valid_romad", second_metric="", inpud_df=pd.DataFrame(), decision_folder='test'):
        print("Generating Report Excel for:", type, "| Epoch selection policy:", epoch_selection_policy, "| Stop Loss:", stop_loss, "| Penalty:", penalty)
        type_str = ''

        if type == 'ensemble_magg':
            ls_results, l_results, s_results, bh_results, bh_intraday_results, general_info = self.get_results(ensemble_type=type, epoch_selection_policy=epoch_selection_policy, remove_nets=remove_nets, 
                thr_ensemble_magg=thr, stop_loss=stop_loss, penalty=penalty, subsample=subsample, decision_folder=decision_folder)
            type_str = 'Ens Magg'

        if type == 'ensemble_elimination':
            ls_results, l_results, s_results, bh_results, bh_intraday_results, general_info = self.get_results(ensemble_type=type, epoch_selection_policy=epoch_selection_policy, remove_nets=remove_nets, 
                thr_ensemble_elimination=thr, stop_loss=stop_loss, penalty=penalty, subsample=subsample, decision_folder=decision_folder)
            type_str = 'Ens El'

        if type == 'ensemble_exclusive':
            ls_results, l_results, s_results, bh_results, bh_intraday_results, general_info = self.get_results(ensemble_type=type, epoch_selection_policy=epoch_selection_policy, remove_nets=remove_nets,
                thr_ensemble_exclusive=thr, stop_loss=stop_loss, penalty=penalty, subsample=subsample, decision_folder=decision_folder)
            type_str = 'Ens Excl'

        if type == 'ensemble_exclusive_short':
            ls_results, l_results, s_results, bh_results, bh_intraday_results, general_info = self.get_results(ensemble_type=type, epoch_selection_policy=epoch_selection_policy, remove_nets=remove_nets,
                thr_ensemble_exclusive=thr, stop_loss=stop_loss, penalty=penalty, subsample=subsample, decision_folder=decision_folder)
            type_str = 'Ens Excl Short'

        if type == 'without_ensemble':
            ls_results, l_results, s_results, bh_results, bh_intraday_results, general_info = self.get_results(ensemble_type=type, epoch_selection_policy=epoch_selection_policy, remove_nets=remove_nets, stop_loss=stop_loss, penalty=penalty, decision_folder=decision_folder)
            type_str = 'Senza Ens'

        if type == 'selection':
            type_str = 'Selection'
            ls_results, l_results, s_results, bh_results, bh_intraday_results, general_info = self.get_selection_results(second_metric=second_metric, operation=epoch_selection_policy, metric=metric, top_number=10, penalty=penalty, stop_loss=stop_loss, walk=0, decision_folder=decision_folder)

        if type == 'from_df':
            type_str = 'From DF'
            ls_results, l_results, s_results, bh_results, bh_intraday_results, general_info = self.get_results(ensemble_type=type, epoch_selection_policy=epoch_selection_policy, remove_nets=remove_nets,
                    stop_loss=stop_loss, penalty=penalty, input_df=inpud_df, decision_folder=decision_folder)

        # Se non esiste la cartella, la creo
        if subfolder_is == '':
            report_path = self.experiment_original_path + self.experiment_name + '/reports/' + type_str + '/'
        else: 
            report_path = self.experiment_original_path + self.experiment_name + '/reports/' + subfolder_is + '/'

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
        worksheet.write('F3', 'Buy & Hold Intraday', cell_bold)

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
        worksheet.write('H3', "Data inizio test", cell_bold)
        worksheet.write('I3', "Data fine test", cell_bold)
        worksheet.write('H4', general_info['dates'][0], cell_classic)
        worksheet.write('I4', general_info['dates'][-1], cell_classic)

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
        if general_info['short_info']['count'] > 0: s_avg_winning_trade = round(s_results['return'] / general_info['short_info']['count'], 3)
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

        # BUY & HOLD INTRADAY
        worksheet.write('F4', bh_intraday_results['return'], cell_classic)
        worksheet.write('F5', bh_intraday_results['mdd'], cell_classic)
        worksheet.write('F6', round(bh_intraday_results['romad'], 3), cell_classic)
        worksheet.write('F7', '', cell_classic)
        #worksheet.write('E8', '', cell_classic)

        # GENERAL
        total_days = general_info['long_info']['count'] + general_info['short_info']['count'] + general_info['hold_info']['count']
        worksheet.write('B9', total_days, cell_classic)

        worksheet.write('B11', general_info['long_info']['count'], cell_classic)
        worksheet.write('B12', general_info['long_info']['guessed'], cell_classic)
        worksheet.write('B13', round(general_info['long_info']['precision'], 3), cell_classic)
        worksheet.write('B14', round(100 * general_info['long_info']['count'] / total_days, 3), cell_classic)  # coverage

        worksheet.write('B16', general_info['short_info']['count'], cell_classic)
        worksheet.write('B17', general_info['short_info']['guessed'], cell_classic)
        worksheet.write('B18', round(general_info['short_info']['precision'], 3), cell_classic)
        worksheet.write('B19', round(100 * general_info['short_info']['count'] / total_days, 3), cell_classic)  # coverage

        worksheet.write('B21', general_info['hold_info']['count'], cell_classic)
        worksheet.write('B22', "-", cell_classic)
        worksheet.write('B23', "-", cell_classic)
        worksheet.write('B24', round(100 * general_info['hold_info']['count'] / total_days, 3), cell_classic)  # coverage

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

        worksheet.write('F12', self.iperparameters['loss_weight'][2][2], cell_classic)
        worksheet.write('F13', self.iperparameters['loss_weight'][1][2], cell_classic)
        worksheet.write('F14', self.iperparameters['loss_weight'][0][2], cell_classic)
        worksheet.write('F15', self.iperparameters['loss_weight'][2][1], cell_classic)
        worksheet.write('F16', self.iperparameters['loss_weight'][1][1], cell_classic)
        worksheet.write('F17', self.iperparameters['loss_weight'][0][1], cell_classic)
        worksheet.write('F18', self.iperparameters['loss_weight'][2][0], cell_classic)
        worksheet.write('F19', self.iperparameters['loss_weight'][1][0], cell_classic)
        worksheet.write('F20', self.iperparameters['loss_weight'][0][0], cell_classic)

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
            worksheet.write('B' + str(28 + i), self.iperparameters['training_set'][i][0], cell_classic)
            worksheet.write('C' + str(28 + i), self.iperparameters['training_set'][i][1], cell_classic)
            worksheet.write('D' + str(28 + i), self.iperparameters['validation_set'][i][0], cell_classic)
            worksheet.write('E' + str(28 + i), self.iperparameters['validation_set'][i][1], cell_classic)
            worksheet.write('F' + str(28 + i), self.iperparameters['test_set'][i][0], cell_classic)
            worksheet.write('G' + str(28 + i), self.iperparameters['test_set'][i][1], cell_classic)

        worksheet.write('H10:I10', "Configurazione rete", cell_header)
        worksheet.write('H11', "Numero epoche", cell_bold)
        worksheet.write('H12', "Numero reti", cell_bold)
        worksheet.write('H13', "Learning Rate", cell_bold)
        worksheet.write('H14', "Batch Size", cell_bold)
        worksheet.write('H15', "Loss Function", cell_bold)
        worksheet.write('H16', "Intorno ensemble", cell_bold)
        worksheet.write('H17', "Mercato", cell_bold)

        worksheet.write('I11', self.iperparameters['epochs'], cell_classic)
        worksheet.write('I12', self.iperparameters['number_of_nets'], cell_classic)
        worksheet.write('I13', self.iperparameters['init_lr'], cell_classic)
        worksheet.write('I14', self.iperparameters['bs'], cell_classic)
        worksheet.write('I15', self.iperparameters['loss_function'], cell_classic)
        worksheet.write('I16', self.iperparameters['validation_thr'], cell_classic)
        worksheet.write('I17', self.iperparameters['predictions_dataset'], cell_classic)

        #################################################################################################
        #################################################################################################
        #################################################################################################

        first_value_of_bh = bh_results['equity_line'][0]
        bh_results['equity_line'] = [x - first_value_of_bh for x in bh_results['equity_line']]

        # Equity long + short
        worksheet = workbook.add_worksheet('CurvaEquityLongShort')
        worksheet.write_column('AA1', general_info['dates'])
        worksheet.write_column('AB1', ls_results['equity_line'])
        worksheet.write_column('AC1', bh_results['equity_line']) # BH
        worksheet.write_column('AD1', bh_intraday_results['equity_line']) # BH intraday

        worksheet.write_column('AE1', general_info['coverage_per_months']['months_start']) # Date coverage mensili
        worksheet.write_column('AF1', general_info['coverage_per_months']['coverage_long_short']) # Coverage mensili l+s
        worksheet.write_column('AG1', general_info['coverage_per_months']['coverage_hold']) # Coverage mensili hold
        
        
        # Equity
        chart = workbook.add_chart({'type': 'line'})
        chart.add_series({'name': 'Equity Long + Short',
                          'categories': '=CurvaEquityLongShort!$AA$1:$AA$' + str(len(general_info['dates'])),
                          'values': '=CurvaEquityLongShort!$AB$1:$AB$' + str(len(ls_results['equity_line']))
                          })

        chart.add_series({'name': 'Buy & Hold',
                          'categories': '=CurvaEquityLongShort!$AA$1:$AA$' + str(len(general_info['dates'])),
                          'values': '=CurvaEquityLongShort!$AC$1:$AC$' + str(len(bh_results['equity_line']))
                          })
        chart.add_series({'name': 'Buy & Hold Intraday',
                          'categories': '=CurvaEquityLongShort!$AA$1:$AA$' + str(len(general_info['dates'])),
                          'values': '=CurvaEquityLongShort!$AD$1:$AD$' + str(len(bh_results['equity_line']))
                          })

        chart.set_title({'name': 'Coverage Long + Short'})
        chart.set_x_axis({'name': 'Mesi'})
        chart.set_y_axis({'name': 'Coverage (%)'})
        chart.set_style(2)
        chart.set_size({'width': 1200, 'height': 500})
        worksheet.insert_chart('A1', chart)

        #chart cvg
        chart = workbook.add_chart({'type': 'line'})
        chart.add_series({'name': 'Coverage Long + Short',
                          'categories': '=CurvaEquityLongShort!$AE$1:$AE$' + str(len( general_info['coverage_per_months']['months_start'])),
                          'values': '=CurvaEquityLongShort!$AF$1:$AF$' + str(len(general_info['coverage_per_months']['coverage_long_short']))
                          })

        chart.add_series({'name': 'Coverage Hold',
                          'categories': '=CurvaEquityLongShort!$AE$1:$AE$' + str(len(general_info['dates'])),
                          'values': '=CurvaEquityLongShort!$AG$1:$AG$' + str(len(general_info['coverage_per_months']['coverage_hold']))
                          })

        chart.set_title({'name': 'Coverage Long + Short'})
        chart.set_x_axis({'name': 'Mesi'})
        chart.set_y_axis({'name': 'Coverage (%)'})
        chart.set_style(2)
        chart.set_size({'width': 1200, 'height': 500})
        worksheet.insert_chart('A27', chart)

        ##### Equity long  only
        worksheet = workbook.add_worksheet('CurvaEquityLongOnly')
        worksheet.write_column('AA1', general_info['dates'])
        worksheet.write_column('AB1', l_results['equity_line'])
        worksheet.write_column('AC1', bh_results['equity_line'])
        worksheet.write_column('AD1', bh_intraday_results['equity_line']) # BH intraday
        
        worksheet.write_column('AE1', general_info['coverage_per_months']['months_start']) # Date coverage mensili
        worksheet.write_column('AF1', general_info['coverage_per_months']['coverage_long']) # Coverage mensili l+s
        worksheet.write_column('AG1', general_info['coverage_per_months']['coverage_hold']) # Coverage mensili hold

        chart = workbook.add_chart({'type': 'line'})
        chart.add_series({'name': 'Equity Long Only',
                          'categories': '=CurvaEquityLongOnly!$AA$1:$AA$' + str(len(general_info['dates'])),
                          'values': '=CurvaEquityLongOnly!$AB$1:$AB$' + str(len(l_results['equity_line']))
                          })
        chart.add_series({'name': 'Buy & Hold',
                          'categories': '=CurvaEquityLongOnly!$AA$1:$AA$' + str(len(general_info['dates'])),
                          'values': '=CurvaEquityLongOnly!$AC$1:$AC$' + str(len(bh_results['equity_line']))
                          })

        chart.add_series({'name': 'Buy & Hold Intraday',
                          'categories': '=CurvaEquityLongOnly!$AA$1:$AA$' + str(len(general_info['dates'])),
                          'values': '=CurvaEquityLongOnly!$AD$1:$AD$' + str(len(bh_results['equity_line']))
                          })

        chart.set_title({'name': 'Equity Long Only'})
        chart.set_x_axis({'name': 'Giorni'})
        chart.set_y_axis({'name': 'Return ($)'})
        chart.set_style(2)
        chart.set_size({'width': 1200, 'height': 500})
        worksheet.insert_chart('A1', chart)

        #chart cvg
        chart = workbook.add_chart({'type': 'line'})
        chart.add_series({'name': 'Coverage Long Only',
                          'categories': '=CurvaEquityLongOnly!$AE$1:$AE$' + str(len( general_info['coverage_per_months']['months_start'])),
                          'values': '=CurvaEquityLongOnly!$AF$1:$AF$' + str(len(general_info['coverage_per_months']['coverage_long']))
                          })

        chart.add_series({'name': 'Coverage Hold',
                          'categories': '=CurvaEquityLongOnly!$AE$1:$AE$' + str(len(general_info['dates'])),
                          'values': '=CurvaEquityLongOnly!$AG$1:$AG$' + str(len(general_info['coverage_per_months']['coverage_hold']))
                          })

        chart.set_title({'name': 'Coverage Long Only'})
        chart.set_x_axis({'name': 'Mesi'})
        chart.set_y_axis({'name': 'Coverage (%)'})
        chart.set_style(2)
        chart.set_size({'width': 1200, 'height': 500})
        worksheet.insert_chart('A27', chart)

        #### Equity short  only
        worksheet = workbook.add_worksheet('CurvaEquityShortgOnly')
        worksheet.write_column('AA1', general_info['dates'])
        worksheet.write_column('AB1', s_results['equity_line'])
        worksheet.write_column('AC1', bh_results['equity_line'])
        worksheet.write_column('AD1', bh_intraday_results['equity_line']) # BH intraday

        worksheet.write_column('AE1', general_info['coverage_per_months']['months_start']) # Date coverage mensili
        worksheet.write_column('AF1', general_info['coverage_per_months']['coverage_short']) # Coverage mensili l+s
        worksheet.write_column('AG1', general_info['coverage_per_months']['coverage_hold']) # Coverage mensili hold

        chart = workbook.add_chart({'type': 'line'})
        chart.add_series({'name': 'Equity Short Only',
                          'categories': '=CurvaEquityShortgOnly!$AA$1:$AA$' + str(len(general_info['dates'])),
                          'values': '=CurvaEquityShortgOnly!$AB$1:$AB$' + str(len(s_results['equity_line']))
                          })
        chart.add_series({'name': 'Buy & Hold',
                          'categories': '=CurvaEquityShortgOnly!$AA$1:$AA$' + str(len(general_info['dates'])),
                          'values': '=CurvaEquityShortgOnly!$AC$1:$AC$' + str(len(bh_results['equity_line']))
                          })
        chart.add_series({'name': 'Buy & Hold Intraday',
                          'categories': '=CurvaEquityShortgOnly!$AA$1:$AA$' + str(len(general_info['dates'])),
                          'values': '=CurvaEquityShortgOnly!$AD$1:$AD$' + str(len(bh_results['equity_line']))
                          })

        chart.set_title({'name': 'Curva Equity Short Only'})
        chart.set_x_axis({'name': 'Giorni'})
        chart.set_y_axis({'name': 'Return ($)'})
        chart.set_style(2)
        chart.set_size({'width': 1200, 'height': 500})
        worksheet.insert_chart('A1', chart)

        #chart cvg
        chart = workbook.add_chart({'type': 'line'})
        chart.add_series({'name': 'Coverage Short Only',
                          'categories': '=CurvaEquityShortgOnly!$AE$1:$AE$' + str(len( general_info['coverage_per_months']['months_start'])),
                          'values': '=CurvaEquityShortgOnly!$AF$1:$AF$' + str(len(general_info['coverage_per_months']['coverage_short']))
                          })

        chart.add_series({'name': 'Coverage Hold',
                          'categories': '=CurvaEquityShortgOnly!$AE$1:$AE$' + str(len(general_info['dates'])),
                          'values': '=CurvaEquityShortgOnly!$AG$1:$AG$' + str(len(general_info['coverage_per_months']['coverage_hold']))
                          })

        chart.set_title({'name': 'Coverage Short Only'})
        chart.set_x_axis({'name': 'Mesi'})
        chart.set_y_axis({'name': 'Coverage (%)'})
        chart.set_style(2)
        chart.set_size({'width': 1200, 'height': 500})
        worksheet.insert_chart('A27', chart)

        workbook.close()

    '''
    '
    '''
    def get_report_excel_swipe(self, report_name, type, thrs_ensemble_magg=[], thrs_ensemble_exclusive=[], thrs_ensemble_elimination=[], remove_nets=False, 
        epoch_selection_policy='long_short', stop_loss=1000, penalty=32, subsample=[], subfolder_is='',  decision_folder='test'):
        print("Generating Report Excel Swipe " + subfolder_is + " for:", type, "| Epoch selection policy:", epoch_selection_policy, "| Stop Loss:", stop_loss, "| Penalty:", penalty)
        thr_global = []

        if type == 'ensemble_magg':
            thr_global = thrs_ensemble_magg
        
        if type == 'ensemble_exclusive':
            thr_global = thrs_ensemble_exclusive
        
        if type == 'ensemble_exclusive_short':
            thr_global = thrs_ensemble_exclusive

        if type == 'ensemble_elimination':
            thr_global = thrs_ensemble_elimination

        ls_results = [dict() for x in range(len(thr_global))] 
        l_results = [dict() for x in range(len(thr_global))] 
        s_results = [dict() for x in range(len(thr_global))] 
        bh_results = [dict() for x in range(len(thr_global))] 
        bh_intraday_results = [dict() for x in range(len(thr_global))] 
        general_info = [dict() for x in range(len(thr_global))] 

        if type == 'ensemble_magg':
            for i, thr in enumerate(thr_global):
                ls_results[i], l_results[i], s_results[i], bh_results[i], bh_intraday_results[i], general_info[i] = self.get_results(ensemble_type=type, epoch_selection_policy=epoch_selection_policy, 
                    thr_ensemble_magg=thr, remove_nets=remove_nets, stop_loss=stop_loss, penalty=penalty, subsample=subsample, decision_folder=decision_folder)

        if type == 'ensemble_exclusive':
            for i, thr in enumerate(thr_global):
                ls_results[i], l_results[i], s_results[i], bh_results[i], bh_intraday_results[i], general_info[i] = self.get_results(ensemble_type=type, epoch_selection_policy=epoch_selection_policy, 
                    thr_ensemble_exclusive=thr, remove_nets=remove_nets, stop_loss=stop_loss, penalty=penalty, subsample=subsample, decision_folder=decision_folder)

        if type == 'ensemble_exclusive_short':
            for i, thr in enumerate(thr_global):
                ls_results[i], l_results[i], s_results[i], bh_results[i], bh_intraday_results[i], general_info[i] = self.get_results(ensemble_type=type, epoch_selection_policy=epoch_selection_policy, 
                    thr_ensemble_exclusive=thr, remove_nets=remove_nets, stop_loss=stop_loss, penalty=penalty, subsample=subsample, decision_folder=decision_folder)

        # Se non esiste la cartella, la creo
        report_path = self.experiment_original_path + self.experiment_name + '/reports/' + subfolder_is + '/'
        if not os.path.isdir(report_path):
            create_folder(report_path)

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
        
        worksheet.write('BZ1', "Count Long")
        worksheet.write('CA1', "Count Short")
        worksheet.write('CB1', "Count Hold")
        
        worksheet.write('CC1', "BH Intraday Return")
        worksheet.write('CD1', "BH Intraday MDD")
        worksheet.write('CE1', "BH Intradey Romad")

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

            # BUY HOLD
            worksheet.write('BQ' + str(i + 2), bh_results[i]['return'])
            worksheet.write('BR' + str(i + 2), bh_results[i]['mdd'])
            worksheet.write('BS' + str(i + 2), bh_results[i]['romad'])

            worksheet.write('BT' + str(i + 2), general_info[i]['long_info']['random_perc'])
            worksheet.write('BU' + str(i + 2), general_info[i]['short_info']['random_perc'])
            worksheet.write('BV' + str(i + 2), general_info[i]['hold_info']['random_perc'])

            worksheet.write('BZ' + str(i + 2), general_info[i]['long_info']['count'])
            worksheet.write('CA' + str(i + 2), general_info[i]['short_info']['count'])
            #worksheet.write('CB' + str(i + 2), general_info[i]['hold_info']['count'])

            # BUY HOLD ITNRADAY
            worksheet.write('CC' + str(i + 2), bh_intraday_results[i]['return'])
            worksheet.write('CD' + str(i + 2), bh_intraday_results[i]['mdd'])
            worksheet.write('CE' + str(i + 2), bh_intraday_results[i]['romad'])
            
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
        chart.add_series({'name': 'Return Buy Hold Intraday',
                          'categories': '=Report!$BA$2:$BA$' + str(len(thr_global) + 1),
                          'values': '=Report!$CC$2:$CC$' + str(len(thr_global) + 1),
                          'line':   {'color': 'purple', 'dash_type': 'dash'}
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
        chart.add_series({'name': 'MDD Buy Hold Intraday',
                          'categories': '=Report!$BA$2:$BA$' + str(len(thr_global) + 1),
                          'values': '=Report!$CD$2:$CD$' + str(len(thr_global) + 1),
                          'line':   {'color': 'purple', 'dash_type': 'dash'}
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
        chart.add_series({'name': 'Romad Buy Hold Intraday',
                          'categories': '=Report!$BA$2:$BA$' + str(len(thr_global) + 1),
                          'values': '=Report!$CE$2:$CE$' + str(len(thr_global) + 1),
                          'line':   {'color': 'purple', 'dash_type': 'dash'}
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
    '
    '''
    def get_report_excel_swipe_per_walk(self, report_name, type, thrs_ensemble_magg=[], thrs_ensemble_exclusive=[], thrs_ensemble_elimination=[], remove_nets=False, 
        epoch_selection_policy='long_short', stop_loss=1000, penalty=32, subsample=[], subfolder='',  decision_folder='test', walk=0):
        print("Generating Report Excel Swipe " + subfolder + " for:", type, "| Epoch selection policy:", epoch_selection_policy, "| Stop Loss:", stop_loss, "| Penalty:", penalty)
        thr_global = []

        if type == 'ensemble_magg':
            thr_global = thrs_ensemble_magg
        
        if type == 'ensemble_exclusive':
            thr_global = thrs_ensemble_exclusive
        
        if type == 'ensemble_exclusive_short':
            thr_global = thrs_ensemble_exclusive

        if type == 'ensemble_elimination':
            thr_global = thrs_ensemble_elimination

        ls_results = [dict() for x in range(len(thr_global))] 
        l_results = [dict() for x in range(len(thr_global))] 
        s_results = [dict() for x in range(len(thr_global))] 
        bh_results = [dict() for x in range(len(thr_global))] 
        bh_intraday_results = [dict() for x in range(len(thr_global))] 
        general_info = [dict() for x in range(len(thr_global))] 

        if type == 'ensemble_magg':
            for i, thr in enumerate(thr_global):
                ls_results[i], l_results[i], s_results[i], bh_results[i], bh_intraday_results[i], general_info[i] = self.get_results_per_walk(ensemble_type=type, epoch_selection_policy=epoch_selection_policy, 
                    thr_ensemble_magg=thr, remove_nets=remove_nets, stop_loss=stop_loss, penalty=penalty, subsample=subsample, decision_folder=decision_folder, walk=walk)

        if type == 'ensemble_exclusive':
            for i, thr in enumerate(thr_global):
                ls_results[i], l_results[i], s_results[i], bh_results[i], bh_intraday_results[i], general_info[i] = self.get_results_per_walk(ensemble_type=type, epoch_selection_policy=epoch_selection_policy, 
                    thr_ensemble_exclusive=thr, remove_nets=remove_nets, stop_loss=stop_loss, penalty=penalty, subsample=subsample, decision_folder=decision_folder, walk=walk)

        if type == 'ensemble_exclusive_short':
            for i, thr in enumerate(thr_global):
                ls_results[i], l_results[i], s_results[i], bh_results[i], bh_intraday_results[i], general_info[i] = self.get_results_per_walk(ensemble_type=type, epoch_selection_policy=epoch_selection_policy, 
                    thr_ensemble_exclusive=thr, remove_nets=remove_nets, stop_loss=stop_loss, penalty=penalty, subsample=subsample, decision_folder=decision_folder, walk=walk)

        # Se non esiste la cartella, la creo
        report_path = self.experiment_original_path + self.experiment_name + '/reports/' + subfolder + '/'
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
        
        worksheet.write('CC1', "BH Intraday Return")
        worksheet.write('CD1', "BH Intraday MDD")
        worksheet.write('CE1', "BH Intradey Romad")

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

            # BUY HOLD
            worksheet.write('BQ' + str(i + 2), bh_results[i]['return'])
            worksheet.write('BR' + str(i + 2), bh_results[i]['mdd'])
            worksheet.write('BS' + str(i + 2), bh_results[i]['romad'])

            worksheet.write('BT' + str(i + 2), general_info[i]['long_info']['random_perc'])
            worksheet.write('BU' + str(i + 2), general_info[i]['short_info']['random_perc'])
            worksheet.write('BV' + str(i + 2), general_info[i]['hold_info']['random_perc'])

            worksheet.write('BZ' + str(i + 2), general_info[i]['long_info']['count'])
            worksheet.write('CA' + str(i + 2), general_info[i]['short_info']['count'])
            #worksheet.write('CB' + str(i + 2), general_info[i]['hold_info']['count'])

            # BUY HOLD ITNRADAY
            worksheet.write('CC' + str(i + 2), bh_intraday_results[i]['return'])
            worksheet.write('CD' + str(i + 2), bh_intraday_results[i]['mdd'])
            worksheet.write('CE' + str(i + 2), bh_intraday_results[i]['romad'])

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
        chart.add_series({'name': 'Return Buy Hold Intraday',
                          'categories': '=Report!$BA$2:$BA$' + str(len(thr_global) + 1),
                          'values': '=Report!$CC$2:$CC$' + str(len(thr_global) + 1),
                          'line':   {'color': 'purple', 'dash_type': 'dash'}
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
        chart.add_series({'name': 'MDD Buy Hold Intraday',
                          'categories': '=Report!$BA$2:$BA$' + str(len(thr_global) + 1),
                          'values': '=Report!$CD$2:$CD$' + str(len(thr_global) + 1),
                          'line':   {'color': 'purple', 'dash_type': 'dash'}
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
        chart.add_series({'name': 'Romad Buy Hold Intraday',
                          'categories': '=Report!$BA$2:$BA$' + str(len(thr_global) + 1),
                          'values': '=Report!$CE$2:$CE$' + str(len(thr_global) + 1),
                          'line':   {'color': 'purple', 'dash_type': 'dash'}
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
    '
    '''
    def get_thr_per_walk_by_cvg(self, type, thrs_ensemble_magg=[], thrs_ensemble_exclusive=[], thrs_ensemble_elimination=[], remove_nets=False, 
        epoch_selection_policy='long_short', stop_loss=1000, penalty=32, target_cvg=0.5):
        
        thr_global = []

        if type == 'ensemble_magg':
            thr_global = thrs_ensemble_magg
        
        if type == 'ensemble_exclusive':
            thr_global = thrs_ensemble_exclusive
        
        if type == 'ensemble_exclusive_short':
            thr_global = thrs_ensemble_exclusive

        if type == 'ensemble_elimination':
            thr_global = thrs_ensemble_elimination

        ls_results = [dict() for x in range(len(thr_global))] 
        l_results = [dict() for x in range(len(thr_global))] 
        s_results = [dict() for x in range(len(thr_global))] 
        bh_results = [dict() for x in range(len(thr_global))] 
        bh_intraday_results = [dict() for x in range(len(thr_global))] 
        general_info = [dict() for x in range(len(thr_global))] 

        

        selected_thr_per_walk = []

        for id_walk, walk in enumerate(self.walks_list):
            coverage_per_thr = []
            print("Calcolo la soglia di ensemble ideale per la coverage target:", target_cvg, "per il walk n°", id_walk)
            if type == 'ensemble_magg':
                for i, thr in enumerate(thr_global):
                    ls_results[i], l_results[i], s_results[i], bh_results[i], bh_intraday_results[i], general_info[i] = self.get_results_per_walk(ensemble_type=type, epoch_selection_policy=epoch_selection_policy, 
                        thr_ensemble_magg=thr, remove_nets=remove_nets, stop_loss=stop_loss, penalty=penalty, decision_folder='validation', walk=id_walk, get_coverage_per_month=False)
                    coverage_per_thr.append(general_info[i]['total_coverage'])

            if type == 'ensemble_exclusive':
                for i, thr in enumerate(thr_global):
                    ls_results[i], l_results[i], s_results[i], bh_results[i], bh_intraday_results[i], general_info[i] = self.get_results_per_walk(ensemble_type=type, epoch_selection_policy=epoch_selection_policy, 
                        thr_ensemble_exclusive=thr, remove_nets=remove_nets, stop_loss=stop_loss, penalty=penalty, decision_folder='validation', walk=id_walk, get_coverage_per_month=False)
                    coverage_per_thr.append(general_info[i]['long_info']['coverage'])

            if type == 'ensemble_exclusive_short':
                for i, thr in enumerate(thr_global):
                    ls_results[i], l_results[i], s_results[i], bh_results[i], bh_intraday_results[i], general_info[i] = self.get_results_per_walk(ensemble_type=type, epoch_selection_policy=epoch_selection_policy, 
                        thr_ensemble_exclusive=thr, remove_nets=remove_nets, stop_loss=stop_loss, penalty=penalty, decision_folder='validation', walk=id_walk, get_coverage_per_month=False)
                    coverage_per_thr.append(general_info[i]['short_info']['coverage'])
            
            selected_id = min(range(len(coverage_per_thr)), key=lambda i: abs(coverage_per_thr[i] - target_cvg))

            print("Selezionato:", selected_id, "per il walk n°", id_walk, "con soglia:", thr_global[selected_id])
            selected_thr_per_walk.append(selected_id )
            print(selected_thr_per_walk)
        
        print(selected_thr_per_walk)
    
    '''
    '
    '''
    def get_report_with_cvg_sel(self, report_name, type, epoch_selection_policy, thrs=[], selected_thrs=[], remove_nets=False):
        remove_nets_str = 'con-rimozione-reti/'

        df = pd.DataFrame()

        if remove_nets == False:
            remove_nets_str = 'senza-rimozione-reti/'

        for id_walk, walk in enumerate(self.walks_list):
            thr = selected_thrs[id_walk]
            print("Per il walk n°", id_walk, " prendo la soglia", str(thrs[thr]))
            test_output_path_per_walk = self.final_decision_per_walk + 'test/' + type + '/' + remove_nets_str + epoch_selection_policy + '/walk_' + str(id_walk) + '/'

            df_test = pd.read_csv(test_output_path_per_walk + 'decisions_' + type + '_' + str(thrs[thr]) + '.csv')

            df = pd.concat([df, df_test])

        df = df_date_merger(df=df, thr_hold=self.iperparameters['hold_labeling'], dataset=self.iperparameters['predictions_dataset'])

        self.get_report_excel(report_name=report_name, type="from_df", epoch_selection_policy=epoch_selection_policy, remove_nets=False,  stop_loss=1000, penalty=32,  inpud_df=df)
    
    '''
    █▀▀ █░░█ █▀▀ ▀▀█▀▀ █▀▀█ █▀▄▀█   █▀▄▀█ █▀▀ ▀▀█▀▀ █▀▀█ ░▀░ █▀▀ █▀▀
    █░░ █░░█ ▀▀█ ░░█░░ █░░█ █░▀░█   █░▀░█ █▀▀ ░░█░░ █▄▄▀ ▀█▀ █░░ ▀▀█
    ▀▀▀ ░▀▀▀ ▀▀▀ ░░▀░░ ▀▀▀▀ ▀░░░▀   ▀░░░▀ ▀▀▀ ░░▀░░ ▀░▀▀ ▀▀▀ ▀▀▀ ▀▀▀
    '''

    '''
    ' 1
    '''
    def generate_single_net_json(self, index_walk, index_net, type='validation', penalty=32, stop_loss=1000):

        walk_str = 'walk_' + str(index_walk)

        if type == 'validation':
            date_list, epochs_list = self.get_date_epochs_walk(path=self.original_predictions_validation_folder, walk=walk_str)
        if type == 'test': 
            date_list, epochs_list = self.get_date_epochs_walk(path=self.original_predictions_test_folder, walk=walk_str)

        net = 'net_' + str(index_net) + '.csv'
        # leggo le predizioni fatte con l'esnemble
        df = pd.read_csv(self.experiment_original_path + self.iperparameters['experiment_name'] + '/predictions/predictions_during_training/' + type + '/walk_' + str(index_walk) + '/' + net)

        # mergio con le label, così ho un subset del df con le date che mi servono e la predizione 
        df_merge_with_label = df_date_merger(df=df.copy(), columns=['date_time', 'delta_next_day', 'delta_current_day', 'close', 'open', 'high', 'low'], dataset=self.iperparameters['predictions_dataset'], thr_hold=self.iperparameters['hold_labeling'])
        #df_merge_with_label = df_date_merger_binary(df=df.copy(), columns=['date_time', 'delta_next_day', 'delta_current_day', 'close', 'open', 'high', 'low'], dataset=self.iperparameters['predictions_dataset'], thr_hold=self.iperparameters['thr_binary_labeling'])
        df_merge_with_label_BINARY = df_date_merger_binary(df=df.copy(), columns=['date_time', 'delta_next_day', 'delta_current_day', 'close', 'open', 'high', 'low'], dataset=self.iperparameters['predictions_dataset'], thr=self.iperparameters['thr_binary_labeling'])

        #df_merge_with_label['date_time'] = df_merge_with_label['date_time'].shift(-1)
        df_merge_with_label = df_merge_with_label.drop(df.index[0])
        df_merge_with_label = df_merge_with_label.drop_duplicates(subset='date_time', keep="first")

        # RETURNS 
        ls_returns = []
        lh_returns = []
        sh_returns = []
        bh_returns = [] # BH
        bh_2_returns = [] # BH Intraday

        # ROMADS
        ls_romads = []
        lh_romads = []
        sh_romads = []
        bh_romads = [] # BH
        bh_2_romads = [] # BH Intraday

        # MDDS
        ls_mdds = []
        lh_mdds = []
        sh_mdds = []
        bh_mdds = [] # BH
        bh_2_mdds = [] # BH Intraday

        # SORTINO
        ls_sortinos = []
        lh_sortinos = []
        sh_sortinos = []
        bh_sortinos = [] # BH
        bh_2_sortinos = [] # BH Intraday

        # SHARPE
        ls_sharpes = []
        lh_sharpes = []
        sh_sharpes = [] # BH
        bh_sharpes = [] # BH 
        bh_2_sharpes = [] # BH Intraday

        # PRECISIONI delta
        longs_precisions = []
        shorts_precisions = []
        holds_precisions = []
        longs_label_coverage = []
        shorts_label_coverage = []

        # PRECISION LABEL
        longs_precisions_ol = []
        shorts_precisions_ol = []
        holds_precisions_ol = []
        longs_label_coverage_ol = []
        shorts_label_coverage_ol = []
        holds_label_coverage_ol = []

        # % DI OPERAZIONI FATTE
        long_operations = []
        short_operations = []
        hold_operations = []

        # Precision over delta
        longs_poc = []
        shorts_poc = []
        holds_poc = []

        # Precision over label
        longs_pol = []
        shorts_pol = []
        holds_pol = []

        #accuracy
        accuracy = []

        label_coverage = Measures.get_delta_coverage(delta=df_merge_with_label['delta_current_day'].tolist())

        # BH
        bh_equity_line, bh_global_return, bh_mdd, bh_romad, bh_i, bh_j  = Measures.get_return_mdd_romad_bh(close=df_merge_with_label['close'].tolist(), multiplier=self.iperparameters['return_multiplier'])

        #dates_debug = df_merge_with_label['date_time'].tolist()
        #print("Type set:", type, "| Return BH per le date:", dates_debug[0], "-", dates_debug[-1], "|", bh_global_return)
        #input()
        # calcolo il return per un epoca
        for epoch in range(1, len(epochs_list) + 1): 
            df_epoch_rename = df_merge_with_label
            df_epoch_rename = df_epoch_rename.rename(columns={'epoch_' + str(epoch): 'decision'})

            df_epoch_rename_BINARY = df_merge_with_label_BINARY
            df_epoch_rename_BINARY = df_merge_with_label_BINARY.rename(columns={'epoch_' + str(epoch): 'decision'})
            #bh intraday
            bh_2_equity_line, bh_2_global_return, bh_2_mdd, bh_2_romad, bh_2_i, bh_2_j  = Measures.get_equity_return_mdd_romad(df=df_epoch_rename.copy(), multiplier=self.iperparameters['return_multiplier'], type='bh_long', penalty=penalty, stop_loss=stop_loss, delta_to_use='delta_current_day')

            ls_equity_line, ls_global_return, ls_mdd, ls_romad, ls_i, ls_j  = Measures.get_equity_return_mdd_romad(df=df_epoch_rename.copy(), multiplier=self.iperparameters['return_multiplier'], type='long_short', penalty=penalty, stop_loss=stop_loss, delta_to_use='delta_next_day')
            lh_equity_line, lh_global_return, lh_mdd, lh_romad, lh_i, lh_j  = Measures.get_equity_return_mdd_romad(df=df_epoch_rename.copy(), multiplier=self.iperparameters['return_multiplier'], type='long_only', penalty=penalty, stop_loss=stop_loss, delta_to_use='delta_next_day')
            sh_equity_line, sh_global_return, sh_mdd, sh_romad, sh_i, sh_j  = Measures.get_equity_return_mdd_romad(df=df_epoch_rename.copy(), multiplier=self.iperparameters['return_multiplier'], type='short_only', penalty=penalty, stop_loss=stop_loss, delta_to_use='delta_next_day')
            
            # Precision over delta
            long, short, hold, general = Measures.get_precision_count_coverage(df=df_epoch_rename.copy(), multiplier=self.iperparameters['return_multiplier'], stop_loss=0, penalty=0, delta_to_use='delta_next_day')

            # Precision over label
            long_ol, short_ol, hold_ol = Measures.get_precision_label(df=df_epoch_rename_BINARY.copy(), label_to_use='label_next_day')
            
            long_poc, short_poc, hold_poc = Measures.get_precision_over_coverage(df=df_epoch_rename.copy(), multiplier=self.iperparameters['return_multiplier'], stop_loss=0, penalty=0, delta_to_use='delta_next_day')
            long_pol, short_pol, hold_pol = Measures.get_precision_over_label(df=df_epoch_rename_BINARY.copy(), label_to_use='label_next_day')

            # SORTINO
            ls_sortinos.append(Measures.get_sortino_ratio(df=df_epoch_rename.copy(), multiplier=self.iperparameters['return_multiplier'], type='long_short', penalty=penalty, stop_loss=stop_loss, delta_to_use='delta_next_day', risk_free=0)[0])
            lh_sortinos.append(Measures.get_sortino_ratio(df=df_epoch_rename.copy(), multiplier=self.iperparameters['return_multiplier'], type='long_only', penalty=penalty, stop_loss=stop_loss, delta_to_use='delta_next_day', risk_free=0)[0])
            sh_sortinos.append(Measures.get_sortino_ratio(df=df_epoch_rename.copy(), multiplier=self.iperparameters['return_multiplier'], type='short_only', penalty=penalty, stop_loss=stop_loss, delta_to_use='delta_next_day', risk_free=0)[0])
            bh_sortinos.append(Measures.get_sortino_ratio(df=df_epoch_rename.copy(), multiplier=self.iperparameters['return_multiplier'], type='bh_long', penalty=0, stop_loss=0, delta_to_use='delta_next_day', risk_free=0)[0])

            # SHARPE
            ls_sharpes.append(Measures.get_sharpe_ratio(df=df_epoch_rename.copy(), multiplier=self.iperparameters['return_multiplier'], type='long_short', penalty=penalty, stop_loss=stop_loss, delta_to_use='delta_next_day', risk_free=0)[0])
            lh_sharpes.append(Measures.get_sharpe_ratio(df=df_epoch_rename.copy(), multiplier=self.iperparameters['return_multiplier'], type='long_only', penalty=penalty, stop_loss=stop_loss, delta_to_use='delta_next_day', risk_free=0)[0])
            sh_sharpes.append(Measures.get_sharpe_ratio(df=df_epoch_rename.copy(), multiplier=self.iperparameters['return_multiplier'], type='short_only', penalty=penalty, stop_loss=stop_loss, delta_to_use='delta_next_day', risk_free=0)[0])
            bh_sharpes.append(Measures.get_sharpe_ratio(df=df_epoch_rename.copy(), multiplier=self.iperparameters['return_multiplier'], type='bh_long', penalty=0, stop_loss=0, delta_to_use='delta_next_day', risk_free=0)[0])

            # RETURNS 
            ls_returns.append(ls_global_return)
            lh_returns.append(lh_global_return)
            sh_returns.append(sh_global_return)
            bh_returns.append(bh_global_return) # BH
            bh_2_returns.append(bh_2_global_return) # BH Intraday

            # ROMADS
            ls_romads.append(ls_romad)
            lh_romads.append(lh_romad)
            sh_romads.append(sh_romad)
            bh_romads.append(bh_romad) # BH
            bh_2_romads.append(bh_2_romad) # BH Intraday

            # MDDS
            ls_mdds.append(ls_mdd)
            lh_mdds.append(lh_mdd)
            sh_mdds.append(sh_mdd)
            bh_mdds.append(bh_mdd) # BH 
            bh_2_mdds.append(bh_2_mdd) # BH Intraday

            # PRECISIONI delta
            longs_precisions.append(long['precision'])
            shorts_precisions.append(short['precision'])
            holds_precisions.append(hold['precision'])
            longs_label_coverage.append(label_coverage['long'])
            shorts_label_coverage.append(label_coverage['short'])

            # PRECISION OVER RANDOM 
            longs_precisions_ol.append(long_ol['precision'])
            shorts_precisions_ol.append(short_ol['precision'])
            holds_precisions_ol.append(hold_ol['precision'])
            longs_label_coverage_ol.append(long_ol['random_perc'])
            shorts_label_coverage_ol.append(short_ol['random_perc'])
            holds_label_coverage_ol.append(hold_ol['random_perc'])

            # % di operazioni fatte
            long_operations.append(long['coverage'])
            short_operations.append(short['coverage'])
            hold_operations.append(hold['coverage'])

            # precision over delta
            longs_poc.append(long_poc)
            shorts_poc.append(short_poc)
            holds_poc.append(hold_poc)

            longs_pol.append(long_pol)
            shorts_pol.append(short_pol)
            holds_pol.append(hold_pol)

            accuracy.append(general['accuracy'])

            #print(' - Epoca ' + str(epoch) + ' / ' + type + ': completa!')

        net_json = {
            "ls_returns": ls_returns,
            "lh_returns": lh_returns,
            "sh_returns": sh_returns,
            "bh_returns": bh_returns,
            "bh_2_returns": bh_2_returns,

            "ls_romads": ls_romads,
            "lh_romads": lh_romads,
            "sh_romads": sh_romads,
            "bh_romads": bh_romads,
            "bh_2_romads": bh_2_romads,

            "ls_mdds": ls_mdds,
            "lh_mdds": lh_mdds,
            "sh_mdds": sh_mdds,
            "bh_mdds": bh_mdds,
            "bh_2_mdds": bh_2_mdds,

            "longs_precisions": longs_precisions,
            "shorts_precisions": shorts_precisions,
            "holds_precisions": holds_precisions,
            "longs_label_coverage": longs_label_coverage,
            "shorts_label_coverage": shorts_label_coverage,

            "longs_precisions_ol": longs_precisions_ol,
            "shorts_precisions_ol": shorts_precisions_ol,
            "holds_precisions_ol": holds_precisions_ol,
            "longs_label_coverage_ol": longs_label_coverage_ol,
            "shorts_label_coverage_ol": shorts_label_coverage_ol,
            "holds_label_coverage_ol": holds_label_coverage_ol,

            "long_operations": long_operations,
            "short_operations": short_operations,
            "hold_operations": hold_operations,

            "longs_poc": longs_poc,
            "shorts_poc": shorts_poc,
            "holds_poc": holds_poc,

            "longs_pol": longs_pol,
            "shorts_pol": shorts_pol,
            "holds_pol": holds_pol,

            "accuracy": accuracy,
            
            "ls_sortinos": ls_sortinos,
            "lh_sortinos": lh_sortinos,
            "sh_sortinos": sh_sortinos,
            "bh_sortinos": bh_sortinos, # BH
            
            "ls_sharpes": ls_sharpes,
            "lh_sharpes": lh_sharpes,
            "sh_sharpes": sh_sharpes,
            "bh_sharpes": bh_sharpes # BH           
        }

        output_path = self.experiment_original_path + self.iperparameters['experiment_name'] + '/calculated_metrics/' + type + '/walk_' + str(index_walk) + '/' 
        create_folder(output_path)

        with open(output_path + 'net_' + str(index_net) + '.json', 'w') as json_file:
            json.dump(net_json, json_file, indent=4)

        do_plot(metrics=net_json, walk=str(index_walk), epochs=len(epochs_list), main_path=self.experiment_original_path, experiment_name=self.iperparameters['experiment_name'], net=str(index_net), type=type)

        return net_json

    '''
    ' 2 
    '''
    def generate_avg_json(self, type='validation', id_walk=None):
        bottom = 0
        top = len(self.walks_list)

        if id_walk != None: 
            bottom = id_walk
            top = id_walk + 1

        for index_walk in range(bottom, top):
            
            walk_str = 'walk_' + str(index_walk)
            if type == 'validation':
                date_list, epochs_list = self.get_date_epochs_walk(path=self.original_predictions_validation_folder, walk=walk_str)
            if type=='test': 
                date_list, epochs_list = self.get_date_epochs_walk(path=self.original_predictions_test_folder, walk=walk_str)

            
            avg_ls_returns = []
            avg_lh_returns = []
            avg_sh_returns = []
            avg_bh_returns = [] # BH
            avg_bh_2_returns = [] # BH Intraday

            # MDDS
            avg_ls_mdds = []
            avg_lh_mdds = []
            avg_sh_mdds = []
            avg_bh_mdds = [] # BH
            avg_bh_2_mdds = [] # BH Intraday

            # ROMADS
            avg_ls_romads = []
            avg_lh_romads = []
            avg_sh_romads = []
            avg_bh_romads = []
            avg_bh_2_romads = [] # BH Intraday

            # PRECISION DELTA
            avg_longs_precisions = []
            avg_shorts_precisions = []
            avg_label_longs_coverage = []
            avg_label_shorts_coverage = []

            # PRECISION LABEL 
            avg_longs_precisions_ol = []
            avg_shorts_precisions_ol = []
            avg_holds_precisions_ol = []
            avg_longs_label_coverage_ol = []
            avg_shorts_label_coverage_ol = []
            avg_holds_label_coverage_ol = []

            # precision over delta
            avg_longs_poc = []
            avg_shorts_poc = []
            avg_holds_poc = []

            # precision over label
            avg_longs_pol = []
            avg_shorts_pol = []
            avg_holds_pol = []

            # accuracy
            avg_accuracy = []

            # SORTINO
            avg_ls_sortinos = []
            avg_lh_sortinos = []
            avg_sh_sortinos = []
            avg_bh_sortinos = [] # BH
            
            # SHARPE
            avg_ls_sharpes = []
            avg_lh_sharpes = []
            avg_sh_sharpes = []
            avg_bh_sharpes = [] # BH

            # RETURNS
            all_ls_returns = np.zeros(shape=(len(self.nets_list), len(epochs_list)))
            all_lh_returns = np.zeros(shape=(len(self.nets_list), len(epochs_list)))
            all_sh_returns = np.zeros(shape=(len(self.nets_list), len(epochs_list)))
            all_bh_return = np.zeros(shape=(len(self.nets_list), len(epochs_list))) #BH
            all_bh_2_return = np.zeros(shape=(len(self.nets_list), len(epochs_list))) #BH

            # ROMADS
            all_ls_romads = np.zeros(shape=(len(self.nets_list), len(epochs_list)))
            all_lh_romads = np.zeros(shape=(len(self.nets_list), len(epochs_list)))
            all_sh_romads = np.zeros(shape=(len(self.nets_list), len(epochs_list)))
            all_bh_romads = np.zeros(shape=(len(self.nets_list), len(epochs_list))) #BH
            all_bh_2_romads = np.zeros(shape=(len(self.nets_list), len(epochs_list))) #BH

            # MDDS
            all_ls_mdds = np.zeros(shape=(len(self.nets_list), len(epochs_list)))
            all_lh_mdds = np.zeros(shape=(len(self.nets_list), len(epochs_list)))
            all_sh_mdds = np.zeros(shape=(len(self.nets_list), len(epochs_list)))
            all_bh_mdds = np.zeros(shape=(len(self.nets_list), len(epochs_list))) # BH
            all_bh_2_mdds = np.zeros(shape=(len(self.nets_list), len(epochs_list))) # BH

            # Precision delta
            all_longs_precisions = np.zeros(shape=(len(self.nets_list), len(epochs_list)))
            all_shorts_precisions = np.zeros(shape=(len(self.nets_list), len(epochs_list)))
            all_holds_precisions = np.zeros(shape=(len(self.nets_list), len(epochs_list)))
            all_labels_longs_coverage = np.zeros(shape=(len(self.nets_list), len(epochs_list)))
            all_labels_shorts_coverage = np.zeros(shape=(len(self.nets_list), len(epochs_list)))

            # precision label
            all_longs_precisions_ol = np.zeros(shape=(len(self.nets_list), len(epochs_list)))
            all_shorts_precisions_ol = np.zeros(shape=(len(self.nets_list), len(epochs_list)))
            all_holds_precisions_ol = np.zeros(shape=(len(self.nets_list), len(epochs_list)))
            all_labels_longs_coverage_ol = np.zeros(shape=(len(self.nets_list), len(epochs_list)))
            all_labels_shorts_coverage_ol = np.zeros(shape=(len(self.nets_list), len(epochs_list)))
            all_labels_holds_coverage_ol = np.zeros(shape=(len(self.nets_list), len(epochs_list)))

            # % di operazioni fatte 
            all_long_operations = np.zeros(shape=(len(self.nets_list), len(epochs_list)))
            all_short_operations = np.zeros(shape=(len(self.nets_list), len(epochs_list)))
            all_hold_operations = np.zeros(shape=(len(self.nets_list), len(epochs_list)))

            # Precision over coverage
            all_longs_poc = np.zeros(shape=(len(self.nets_list), len(epochs_list)))
            all_shorts_poc = np.zeros(shape=(len(self.nets_list), len(epochs_list)))
            all_holds_poc = np.zeros(shape=(len(self.nets_list), len(epochs_list)))
            
            # precision over label
            all_longs_pol = np.zeros(shape=(len(self.nets_list), len(epochs_list)))
            all_shorts_pol = np.zeros(shape=(len(self.nets_list), len(epochs_list)))
            all_holds_pol = np.zeros(shape=(len(self.nets_list), len(epochs_list)))

            #accuracy
            all_accuracy = np.zeros(shape=(len(self.nets_list), len(epochs_list)))

             # SORTINO
            all_ls_sortinos = np.zeros(shape=(len(self.nets_list), len(epochs_list)))
            all_lh_sortinos = np.zeros(shape=(len(self.nets_list), len(epochs_list)))
            all_sh_sortinos = np.zeros(shape=(len(self.nets_list), len(epochs_list)))
            all_bh_sortinos = np.zeros(shape=(len(self.nets_list), len(epochs_list)))
            
            # SHARPE
            all_ls_sharpes = np.zeros(shape=(len(self.nets_list), len(epochs_list)))
            all_lh_sharpes = np.zeros(shape=(len(self.nets_list), len(epochs_list)))
            all_sh_sharpes = np.zeros(shape=(len(self.nets_list), len(epochs_list)))
            all_bh_sharpes = np.zeros(shape=(len(self.nets_list), len(epochs_list)))

            input_path = self.experiment_original_path + self.iperparameters['experiment_name'] + '/calculated_metrics/' + type + '/walk_' + str(index_walk) + '/' 

            for index_net in range(0, len(self.nets_list)):
                net_json = {}
                with open(input_path + 'net_' + str(index_net) + '.json') as json_file:
                    net_json = json.load(json_file)

                # RETURNS
                all_ls_returns[index_net] = net_json['ls_returns']
                all_lh_returns[index_net] = net_json['lh_returns']
                all_sh_returns[index_net] = net_json['sh_returns']
                all_bh_return[index_net] = net_json['bh_returns'] # BH
                all_bh_2_return[index_net] = net_json['bh_2_returns'] # BH

                # ROMADS
                all_ls_romads[index_net] = net_json['ls_romads']
                all_lh_romads[index_net] = net_json['lh_romads']
                all_sh_romads[index_net] = net_json['sh_romads']
                all_bh_romads[index_net] = net_json['bh_romads'] # BH
                all_bh_2_romads[index_net] = net_json['bh_2_romads'] # BH

                # MDDS
                all_ls_mdds[index_net] = net_json['ls_mdds']
                all_lh_mdds[index_net] = net_json['lh_mdds']
                all_sh_mdds[index_net] = net_json['sh_mdds']
                all_bh_mdds[index_net] = net_json['bh_mdds'] #BH
                all_bh_2_mdds[index_net] = net_json['bh_2_mdds'] #BH

                # precision delta
                all_longs_precisions[index_net] = net_json['longs_precisions']
                all_shorts_precisions[index_net] = net_json['shorts_precisions']
                all_holds_precisions[index_net] = net_json['holds_precisions']
                all_labels_longs_coverage[index_net] = net_json['longs_label_coverage']
                all_labels_shorts_coverage[index_net] = net_json['shorts_label_coverage']

                # precision label
                all_longs_precisions_ol[index_net] = net_json['longs_precisions_ol']
                all_shorts_precisions_ol[index_net] = net_json['shorts_precisions_ol']
                all_holds_precisions_ol[index_net] = net_json['holds_precisions_ol']
                all_labels_longs_coverage_ol[index_net] = net_json['longs_label_coverage_ol']
                all_labels_shorts_coverage_ol[index_net] = net_json['shorts_label_coverage_ol']
                all_labels_holds_coverage_ol[index_net] = net_json['holds_label_coverage_ol']

                # % di operazioni fatte
                all_long_operations[index_net] = net_json['long_operations']
                all_short_operations[index_net] = net_json['short_operations']
                all_hold_operations[index_net] = net_json['hold_operations']

                all_longs_poc[index_net] = net_json['longs_poc']
                all_shorts_poc[index_net] = net_json['shorts_poc']
                all_holds_poc[index_net] = net_json['holds_poc']

                all_longs_pol[index_net] = net_json['longs_pol']
                all_shorts_pol[index_net] = net_json['shorts_pol']
                all_holds_pol[index_net] = net_json['holds_pol']

                # SORTINO
                all_ls_sortinos[index_net] = net_json['ls_sortinos']
                all_lh_sortinos[index_net] = net_json['lh_sortinos']
                all_sh_sortinos[index_net] = net_json['sh_sortinos']
                all_bh_sortinos[index_net] = net_json['bh_sortinos']
                
                # SHARPE
                all_ls_sharpes[index_net] = net_json['ls_sharpes']
                all_lh_sharpes[index_net] = net_json['lh_sharpes']
                all_sh_sharpes[index_net] = net_json['sh_sharpes']
                all_bh_sharpes[index_net] = net_json['bh_sharpes']
                #accuracy
                all_accuracy[index_net] = net_json['accuracy']

            # RETURNS
            avg_ls_returns = np.around(np.average(all_ls_returns, axis=0), decimals=3)
            avg_lh_returns = np.around(np.average(all_lh_returns, axis=0), decimals=3)
            avg_sh_returns = np.around(np.average(all_sh_returns, axis=0), decimals=3)
            avg_bh_returns = np.average(all_bh_return, axis=0) # BH
            avg_bh_2_returns = np.average(all_bh_2_return, axis=0) # BH

            # MDDS
            avg_ls_mdds = np.around(np.average(all_ls_mdds, axis=0), decimals=3)
            avg_lh_mdds = np.around(np.average(all_lh_mdds, axis=0), decimals=3)
            avg_sh_mdds = np.around(np.average(all_sh_mdds, axis=0), decimals=3)
            avg_bh_mdds = np.average(all_bh_mdds, axis=0) # BH
            avg_bh_2_mdds = np.average(all_bh_2_mdds, axis=0) # BH

            # ROMADS
            avg_ls_romads = np.divide(avg_ls_returns, avg_ls_mdds, out=np.zeros_like(avg_ls_returns), where=avg_ls_mdds!=0)
            avg_lh_romads = np.divide(avg_lh_returns, avg_lh_mdds, out=np.zeros_like(avg_lh_returns), where=avg_lh_mdds!=0)
            avg_sh_romads = np.divide(avg_sh_returns, avg_sh_mdds, out=np.zeros_like(avg_sh_returns), where=avg_sh_mdds!=0)
            avg_bh_romads = np.divide(avg_bh_returns, avg_bh_mdds, out=np.zeros_like(avg_bh_returns), where=avg_bh_mdds!=0)
            avg_bh_2_romads = np.divide(avg_bh_2_returns, avg_bh_2_mdds, out=np.zeros_like(avg_bh_2_returns), where=avg_bh_2_mdds!=0)

            # rimuovo i nan dai romads
            avg_ls_romads = np.around(np.nan_to_num(avg_ls_romads), decimals=3)
            avg_lh_romads = np.around(np.nan_to_num(avg_lh_romads), decimals=3)
            avg_sh_romads = np.around(np.nan_to_num(avg_sh_romads), decimals=3)
            avg_sh_romads[~np.isfinite(avg_sh_romads)] = 0

            # PRECISION OVER DELTA
            avg_longs_precisions = np.around(np.average(all_longs_precisions, axis=0), decimals=3)
            avg_shorts_precisions = np.around(np.average(all_shorts_precisions, axis=0), decimals=3)
            avg_holds_precisions = np.around(np.average(all_holds_precisions, axis=0), decimals=3)
            avg_label_longs_coverage = np.around(np.average(all_labels_longs_coverage, axis=0), decimals=3)
            avg_label_shorts_coverage = np.around(np.average(all_labels_shorts_coverage, axis=0), decimals=3)

            # PRECISION LABEL
            avg_longs_precisions_ol = np.around(np.average(all_longs_precisions_ol, axis=0), decimals=3)
            avg_shorts_precisions_ol = np.around(np.average(all_shorts_precisions_ol, axis=0), decimals=3)
            avg_holds_precisions_ol = np.around(np.average(all_holds_precisions_ol, axis=0), decimals=3)
            
            avg_longs_label_coverage_ol = np.around(np.average(all_labels_longs_coverage_ol, axis=0), decimals=3)
            avg_shorts_label_coverage_ol = np.around(np.average(all_labels_shorts_coverage_ol, axis=0), decimals=3)
            avg_holds_label_coverage_ol = np.around(np.average(all_labels_holds_coverage_ol, axis=0), decimals=3)

            # precision over delta
            avg_longs_poc = np.around(np.divide(avg_longs_precisions, avg_label_longs_coverage), decimals=3)
            avg_shorts_poc = np.around(np.divide(avg_shorts_precisions, avg_label_shorts_coverage), decimals=3)
            avg_holds_poc = np.around(np.divide(avg_holds_precisions, avg_label_shorts_coverage), decimals=3)
            avg_longs_poc = (avg_longs_poc - 1 ) * 100
            avg_shorts_poc = (avg_shorts_poc - 1 ) * 100
            avg_holds_poc = (avg_holds_poc - 1 ) * 100

            # precision over label
            avg_longs_pol = np.around(np.divide(avg_longs_precisions_ol, avg_longs_label_coverage_ol), decimals=3) if np.count_nonzero(avg_longs_label_coverage_ol) > 0 else np.zeros(shape=(len(epochs_list)))
            avg_shorts_pol = np.around(np.divide(avg_shorts_precisions_ol, avg_shorts_label_coverage_ol), decimals=3) if np.count_nonzero(avg_shorts_label_coverage_ol) > 0 else np.zeros(shape=(len(epochs_list)))
            avg_holds_pol = np.around(np.divide(avg_holds_precisions_ol, avg_holds_label_coverage_ol), decimals=3) if np.count_nonzero(avg_holds_label_coverage_ol) > 0 else np.zeros(shape=(len(epochs_list)))
            avg_longs_pol = (avg_longs_pol - 1 ) * 100
            avg_shorts_pol = (avg_shorts_pol - 1 ) * 100
            avg_holds_pol = (avg_holds_pol - 1 ) * 100

            #accuracy 
            avg_accuracy = np.around(np.average(all_accuracy, axis=0), decimals=3) 

            avg_long_operations = np.average(all_long_operations, axis=0)
            avg_short_operations= np.average(all_short_operations, axis=0)
            avg_hold_operations = np.average(all_hold_operations, axis=0)

            # SORTINO
            avg_ls_sortinos = np.around(np.average(all_ls_sortinos, axis=0), decimals=3)
            avg_lh_sortinos = np.around(np.average(all_lh_sortinos, axis=0), decimals=3)
            avg_sh_sortinos = np.around(np.average(all_sh_sortinos, axis=0), decimals=3)
            avg_bh_sortinos = np.around(np.average(all_bh_sortinos, axis=0), decimals=3)
            
            # SHARPE
            avg_ls_sharpes = np.around(np.average(all_ls_sharpes, axis=0), decimals=3)
            avg_lh_sharpes = np.around(np.average(all_lh_sharpes, axis=0), decimals=3)
            avg_sh_sharpes = np.around(np.average(all_sh_sharpes, axis=0), decimals=3)
            avg_bh_sharpes = np.around(np.average(all_bh_sharpes, axis=0), decimals=3)

            avg_json = {
                "ls_returns": avg_ls_returns.tolist(),
                "lh_returns": avg_lh_returns.tolist(),
                "sh_returns": avg_sh_returns.tolist(),
                "bh_returns": avg_bh_returns.tolist(), # BH
                "bh_2_returns": avg_bh_2_returns.tolist(), # BH

                "ls_romads": avg_ls_romads.tolist(),
                "lh_romads": avg_lh_romads.tolist(),
                "sh_romads": avg_sh_romads.tolist(),
                "bh_romads": avg_bh_romads.tolist(), # BH
                "bh_2_romads": avg_bh_2_romads.tolist(), # BH

                "ls_mdds": avg_ls_mdds.tolist(),
                "lh_mdds": avg_lh_mdds.tolist(),
                "sh_mdds": avg_sh_mdds.tolist(),
                "bh_mdds": avg_bh_mdds.tolist(), # BH
                "bh_2_mdds": avg_bh_2_mdds.tolist(), # BH

                # precision over delta
                "longs_precisions": avg_longs_precisions.tolist(),
                "shorts_precisions": avg_shorts_precisions.tolist(),
                "holds_precisions": avg_holds_precisions.tolist(),
                "longs_label_coverage": avg_label_longs_coverage.tolist(),
                "shorts_label_coverage": avg_label_shorts_coverage.tolist(),

                # precision over label
                "longs_precisions_ol": avg_longs_precisions_ol.tolist(),
                "shorts_precisions_ol": avg_shorts_precisions_ol.tolist(),
                "holds_precisions_ol": avg_holds_precisions_ol.tolist(),
                "longs_label_coverage_ol": avg_longs_label_coverage_ol.tolist(),
                "shorts_label_coverage_ol": avg_shorts_label_coverage_ol.tolist(),
                "holds_label_coverage_ol": avg_holds_label_coverage_ol.tolist(),

                "long_operations": avg_long_operations.tolist(),
                "short_operations": avg_short_operations.tolist(),
                "hold_operations": avg_hold_operations.tolist(),

                "longs_poc": avg_longs_poc.tolist(),
                "shorts_poc": avg_shorts_poc.tolist(),
                "holds_poc": avg_holds_poc.tolist(),

                "longs_pol": avg_longs_pol.tolist(),
                "shorts_pol": avg_shorts_pol.tolist(),
                "holds_pol": avg_holds_pol.tolist(),

                "accuracy": avg_accuracy.tolist(),

                "ls_sortinos": avg_ls_sortinos.tolist(),
                "lh_sortinos": avg_lh_sortinos.tolist(),
                "sh_sortinos": avg_sh_sortinos.tolist(),
                "bh_sortinos": avg_bh_sortinos.tolist(), # BH
                
                "ls_sharpes": avg_ls_sharpes.tolist(),
                "lh_sharpes": avg_lh_sharpes.tolist(),
                "sh_sharpes": avg_sh_sharpes.tolist(),
                "bh_sharpes": avg_bh_sharpes.tolist() # BH  
            }

            avg_output_path = self.experiment_original_path + self.iperparameters['experiment_name'] + '/calculated_metrics/' + type + '/average/' 
            
            create_folder(avg_output_path)

            with open(avg_output_path + 'walk_' + str(index_walk) + '.json', 'w') as json_file:
                json.dump(avg_json, json_file, indent=4)
                #json.dump(net_json, json_file)
            
            print(self.iperparameters['experiment_name'], '|', type, "| Salvate le metriche AVG per walk n° ", index_walk)

            do_plot(metrics=avg_json, walk=index_walk, epochs=len(epochs_list), main_path=self.experiment_original_path, experiment_name=self.iperparameters['experiment_name'], net='average', type=type) 

            #return avg_json
    
    '''
    ' 3
    '''
    def generate_single_net_json_all_walk(self, index_net, type='validation', penalty=32, stop_loss=1000):

        net = 'net_' + str(index_net) + '.csv'
        
        df = pd.DataFrame()

        for index_walk in range(0, len(self.walks_list)):
            # leggo le predizioni fatte con l'esnemble
            df_walk = pd.read_csv(self.experiment_original_path + self.iperparameters['experiment_name'] + '/predictions/predictions_during_training/' + type + '/walk_' + str(index_walk) + '/' + net)
            df = pd.concat([df, df_walk], axis=0)

        # sistemo le date in linea con il delta per far tornare i risultati uguali ai report excel. 
        
        # mergio con le label, così ho un subset del df con le date che mi servono e la predizione 
        df_merge_with_label = df_date_merger(df=df.copy(), columns=['date_time', 'delta_next_day', 'delta_current_day', 'close', 'open', 'high', 'low'], dataset=self.iperparameters['predictions_dataset'], thr_hold=self.iperparameters['hold_labeling'])
        df_merge_with_label_BINARY = df_date_merger_binary(df=df.copy(), columns=['date_time', 'delta_next_day', 'delta_current_day', 'close', 'open', 'high', 'low'], dataset=self.iperparameters['predictions_dataset'], thr=self.iperparameters['thr_binary_labeling'])

        #df_merge_with_label['date_time'] = df_merge_with_label['date_time'].shift(-1)
        df_merge_with_label = df_merge_with_label.drop(df.index[0])
        df_merge_with_label = df_merge_with_label.drop_duplicates(subset='date_time', keep="first")

        # RETURNS 
        ls_returns = []
        lh_returns = []
        sh_returns = []
        bh_returns = [] # BH
        bh_2_returns = [] # BH

        # ROMADS
        ls_romads = []
        lh_romads = []
        sh_romads = []
        bh_romads = [] # BH
        bh_2_romads = [] # BH

        # MDDS
        ls_mdds = []
        lh_mdds = []
        sh_mdds = []
        bh_mdds = [] # BH
        bh_2_mdds = [] # BH

        # SORTINO
        ls_sortinos = []
        lh_sortinos = []
        sh_sortinos = []
        bh_sortinos = [] # BH

        # SHARPE
        ls_sharpes = []
        lh_sharpes = []
        sh_sharpes = []
        bh_sharpes = [] # BH

        # PRECISIONI E LINEA RETTA DEL BILANCIAMENTO DELLE CLASSI
        longs_precisions = []
        shorts_precisions = []
        holds_precisions = []
        longs_label_coverage = []
        shorts_label_coverage = []

        # PRECISION OVER LABEL
        longs_precisions_ol = []
        shorts_precisions_ol = []
        holds_precisions_ol = []
        longs_label_coverage_ol = []
        shorts_label_coverage_ol = []
        holds_label_coverage_ol = []

        # % DI OPERAZIONI FATTE
        long_operations = []
        short_operations = []
        hold_operations = []

        # POC Delta
        longs_poc = []
        shorts_poc = []
        holds_poc = []
        
        # poc label
        longs_pol = []
        shorts_pol = []
        holds_pol = []

        #accuracy
        accuracy = []

        label_coverage = Measures.get_delta_coverage(delta=df_merge_with_label['delta_current_day'].tolist())

        bh_equity_line, bh_global_return, bh_mdd, bh_romad, bh_i, bh_j  = Measures.get_return_mdd_romad_bh(close=df_merge_with_label['close'].tolist(), multiplier=self.iperparameters['return_multiplier'])

        #if type == 'test':
        #    print(type, ": BH Classico:", bh_global_return)
        #    input()

        dates_debug = df_merge_with_label['date_time'].tolist()

        # calcolo il return per un epoca
        for epoch in range(1, self.iperparameters['epochs'] + 1): 
            df_epoch_rename = df_merge_with_label
            df_epoch_rename = df_epoch_rename.rename(columns={'epoch_' + str(epoch): 'decision'})

            df_epoch_rename_BINARY = df_merge_with_label_BINARY
            df_epoch_rename_BINARY = df_merge_with_label_BINARY.rename(columns={'epoch_' + str(epoch): 'decision'})

            bh_2_equity_line, bh_2_global_return, bh_2_mdd, bh_2_romad, bh_2_i, bh_2_j  = Measures.get_equity_return_mdd_romad(df=df_epoch_rename.copy(), multiplier=self.iperparameters['return_multiplier'], type='bh_long', penalty=penalty, stop_loss=stop_loss, delta_to_use='delta_current_day')
            #if type == 'test':
            #    print(type, ": BH Intradray:", bh_2_global_return)
            #    input()

            ls_equity_line, ls_global_return, ls_mdd, ls_romad, ls_i, ls_j  = Measures.get_equity_return_mdd_romad(df=df_epoch_rename.copy(), multiplier=self.iperparameters['return_multiplier'], type='long_short', penalty=penalty, stop_loss=stop_loss, delta_to_use='delta_next_day')
            lh_equity_line, lh_global_return, lh_mdd, lh_romad, lh_i, lh_j  = Measures.get_equity_return_mdd_romad(df=df_epoch_rename.copy(), multiplier=self.iperparameters['return_multiplier'], type='long_only', penalty=penalty, stop_loss=stop_loss, delta_to_use='delta_next_day')
            sh_equity_line, sh_global_return, sh_mdd, sh_romad, sh_i, sh_j  = Measures.get_equity_return_mdd_romad(df=df_epoch_rename.copy(), multiplier=self.iperparameters['return_multiplier'], type='short_only', penalty=penalty, stop_loss=stop_loss, delta_to_use='delta_next_day')
            
            # Precision delta
            long, short, hold, general = Measures.get_precision_count_coverage(df=df_epoch_rename.copy(), multiplier=self.iperparameters['return_multiplier'], stop_loss=0, penalty=0, delta_to_use='delta_next_day')

            # Precision label
            long_ol, short_ol, hold_ol = Measures.get_precision_label(df=df_epoch_rename_BINARY.copy(), label_to_use='label_next_day')

            long_poc, short_poc, hold_poc = Measures.get_precision_over_coverage(df=df_epoch_rename.copy(), multiplier=self.iperparameters['return_multiplier'], stop_loss=0, penalty=0, delta_to_use='delta_next_day')
            long_pol, short_pol, hold_pol = Measures.get_precision_over_label(df=df_epoch_rename_BINARY.copy(), label_to_use='label_next_day')

            # SORTINO
            ls_sortinos.append(Measures.get_sortino_ratio(df=df_epoch_rename.copy(), multiplier=self.iperparameters['return_multiplier'], type='long_short', penalty=penalty, stop_loss=stop_loss, delta_to_use='delta_next_day', risk_free=0)[0])
            lh_sortinos.append(Measures.get_sortino_ratio(df=df_epoch_rename.copy(), multiplier=self.iperparameters['return_multiplier'], type='long_only', penalty=penalty, stop_loss=stop_loss, delta_to_use='delta_next_day', risk_free=0)[0])
            sh_sortinos.append(Measures.get_sortino_ratio(df=df_epoch_rename.copy(), multiplier=self.iperparameters['return_multiplier'], type='short_only', penalty=penalty, stop_loss=stop_loss, delta_to_use='delta_next_day', risk_free=0)[0])
            bh_sortinos.append(Measures.get_sortino_ratio(df=df_epoch_rename.copy(), multiplier=self.iperparameters['return_multiplier'], type='bh_long', penalty=0, stop_loss=0, delta_to_use='delta_next_day', risk_free=0)[0])

            # SHARPE
            ls_sharpes.append(Measures.get_sharpe_ratio(df=df_epoch_rename.copy(), multiplier=self.iperparameters['return_multiplier'], type='long_short', penalty=penalty, stop_loss=stop_loss, delta_to_use='delta_next_day', risk_free=0)[0])
            lh_sharpes.append(Measures.get_sharpe_ratio(df=df_epoch_rename.copy(), multiplier=self.iperparameters['return_multiplier'], type='long_only', penalty=penalty, stop_loss=stop_loss, delta_to_use='delta_next_day', risk_free=0)[0])
            sh_sharpes.append(Measures.get_sharpe_ratio(df=df_epoch_rename.copy(), multiplier=self.iperparameters['return_multiplier'], type='short_only', penalty=penalty, stop_loss=stop_loss, delta_to_use='delta_next_day', risk_free=0)[0])
            bh_sharpes.append(Measures.get_sharpe_ratio(df=df_epoch_rename.copy(), multiplier=self.iperparameters['return_multiplier'], type='bh_long', penalty=0, stop_loss=0, delta_to_use='delta_next_day', risk_free=0)[0])

            # RETURNS 
            ls_returns.append(ls_global_return)
            lh_returns.append(lh_global_return)
            sh_returns.append(sh_global_return)
            bh_returns.append(bh_global_return) # BH
            bh_2_returns.append(bh_2_global_return) # BH Intraday

            # ROMADS
            ls_romads.append(ls_romad)
            lh_romads.append(lh_romad)
            sh_romads.append(sh_romad)
            bh_romads.append(bh_romad) # BH
            bh_2_romads.append(bh_2_romad) # BH Intraday

            # MDDS
            ls_mdds.append(ls_mdd)
            lh_mdds.append(lh_mdd)
            sh_mdds.append(sh_mdd)
            bh_mdds.append(bh_mdd) # BH 
            bh_2_mdds.append(bh_2_mdd) # BH Intraday

            # PRECISIONI E LINEA RETTA DEL BILANCIAMENTO DELLE CLASSI
            longs_precisions.append(long['precision'])
            shorts_precisions.append(short['precision'])
            holds_precisions.append(hold['precision'])
            longs_label_coverage.append(label_coverage['long'])
            shorts_label_coverage.append(label_coverage['short'])

            # PRECISION OVER RANDOM 
            longs_precisions_ol.append(long_ol['precision'])
            shorts_precisions_ol.append(short_ol['precision'])
            holds_precisions_ol.append(hold_ol['precision'])
            longs_label_coverage_ol.append(long_ol['random_perc'])
            shorts_label_coverage_ol.append(short_ol['random_perc'])
            holds_label_coverage_ol.append(hold_ol['random_perc'])

            # % di operazioni fatte
            long_operations.append(long['coverage'])
            short_operations.append(short['coverage'])
            hold_operations.append(hold['coverage'])

            # POC
            longs_poc.append(long_poc)
            shorts_poc.append(short_poc)
            holds_poc.append(hold_poc)

            longs_pol.append(long_pol)
            shorts_pol.append(short_pol)
            holds_pol.append(hold_pol)

            accuracy.append(general['accuracy'])

            #print(' - Epoca ' + str(epoch) + ' / ' + type + ': completa!')

        net_json = {
            "ls_returns": ls_returns,
            "lh_returns": lh_returns,
            "sh_returns": sh_returns,
            "bh_returns": bh_returns,
            "bh_2_returns": bh_2_returns,

            "ls_romads": ls_romads,
            "lh_romads": lh_romads,
            "sh_romads": sh_romads,
            "bh_romads": bh_romads,
            "bh_2_romads": bh_2_romads,

            "ls_mdds": ls_mdds,
            "lh_mdds": lh_mdds,
            "sh_mdds": sh_mdds,
            "bh_mdds": bh_mdds,
            "bh_2_mdds": bh_2_mdds,

            "longs_precisions": longs_precisions,
            "shorts_precisions": shorts_precisions,
            "holds_precisions": holds_precisions,
            "longs_label_coverage": longs_label_coverage,
            "shorts_label_coverage": shorts_label_coverage,

            "longs_precisions_ol": longs_precisions_ol,
            "shorts_precisions_ol": shorts_precisions_ol,
            "holds_precisions_ol": holds_precisions_ol,
            "longs_label_coverage_ol": longs_label_coverage_ol,
            "shorts_label_coverage_ol": shorts_label_coverage_ol,
            "holds_label_coverage_ol": holds_label_coverage_ol,

            "long_operations": long_operations,
            "short_operations": short_operations,
            "hold_operations": hold_operations,

            "longs_poc": longs_poc,
            "shorts_poc": shorts_poc,
            "holds_poc": holds_poc,

            "longs_pol": longs_pol,
            "shorts_pol": shorts_pol,
            "holds_pol": holds_pol,

            "accuracy": accuracy,
            
            "ls_sortinos": ls_sortinos,
            "lh_sortinos": lh_sortinos,
            "sh_sortinos": sh_sortinos,
            "bh_sortinos": bh_sortinos, # BH
            
            "ls_sharpes": ls_sharpes,
            "lh_sharpes": lh_sharpes,
            "sh_sharpes": sh_sharpes,
            "bh_sharpes": bh_sharpes # BH           
        }

        output_path = self.experiment_original_path + self.iperparameters['experiment_name'] + '/calculated_metrics/' + type + '/all_walk/' 
        create_folder(output_path)

        with open(output_path + 'net_' + str(index_net) + '.json', 'w') as json_file:
            json.dump(net_json, json_file, indent=4)

        do_plot(metrics=net_json, walk=str(index_walk), epochs=self.iperparameters['epochs'], main_path=self.experiment_original_path, experiment_name=self.iperparameters['experiment_name'], net=str(index_net), type=type)

        return net_json

    '''
    ' 4
    '''
    def generate_avg_json_all_walk(self, type='validation'):
        avg_ls_returns = []
        avg_lh_returns = []
        avg_sh_returns = []
        avg_bh_returns = [] # BH
        avg_bh_2_returns = [] # BH

        # MDDS
        avg_ls_mdds = []
        avg_lh_mdds = []
        avg_sh_mdds = []
        avg_bh_mdds = [] # BH
        avg_bh_2_mdds = [] # BH

        # ROMADS
        avg_ls_romads = []
        avg_lh_romads = []
        avg_sh_romads = []
        avg_bh_romads = []
        avg_bh_2_romads = []

        # PRECISION OVER DELTA
        avg_longs_precisions = []
        avg_shorts_precisions = []
        avg_label_longs_coverage = []
        avg_label_shorts_coverage = []

        # PRECISION OVER LABEL
        avg_longs_precisions_ol = []
        avg_shorts_precisions_ol = []
        avg_holds_precisions_ol = []
        avg_longs_label_coverage_ol = []
        avg_shorts_label_coverage_ol = []
        avg_holds_label_coverage_ol = []
        
        # po delta
        avg_longs_poc = []
        avg_shorts_poc = []
        avg_holds_poc = []

        # po label
        avg_shorts_pol = []
        avg_longs_pol = []
        avg_holds_pol = []

        # accuracy
        avg_accuracy = []

        # SORTINO
        avg_ls_sortinos = []
        avg_lh_sortinos = []
        avg_sh_sortinos = []
        avg_bh_sortinos = [] # BH
        
        # SHARPE
        avg_ls_sharpes = []
        avg_lh_sharpes = []
        avg_sh_sharpes = []
        avg_bh_sharpes = [] # BH

        # RETURNS
        all_ls_returns = np.zeros(shape=(len(self.nets_list), self.iperparameters['epochs']))
        all_lh_returns = np.zeros(shape=(len(self.nets_list), self.iperparameters['epochs']))
        all_sh_returns = np.zeros(shape=(len(self.nets_list), self.iperparameters['epochs']))
        all_bh_return = np.zeros(shape=(len(self.nets_list), self.iperparameters['epochs'])) #BH
        all_bh_2_return = np.zeros(shape=(len(self.nets_list), self.iperparameters['epochs'])) #BH

        # ROMADS
        all_ls_romads = np.zeros(shape=(len(self.nets_list), self.iperparameters['epochs']))
        all_lh_romads = np.zeros(shape=(len(self.nets_list), self.iperparameters['epochs']))
        all_sh_romads = np.zeros(shape=(len(self.nets_list), self.iperparameters['epochs']))
        all_bh_romads = np.zeros(shape=(len(self.nets_list), self.iperparameters['epochs'])) #BH
        all_bh_2_romads = np.zeros(shape=(len(self.nets_list), self.iperparameters['epochs'])) #BH

        # MDDS
        all_ls_mdds = np.zeros(shape=(len(self.nets_list), self.iperparameters['epochs']))
        all_lh_mdds = np.zeros(shape=(len(self.nets_list), self.iperparameters['epochs']))
        all_sh_mdds = np.zeros(shape=(len(self.nets_list), self.iperparameters['epochs']))
        all_bh_mdds = np.zeros(shape=(len(self.nets_list), self.iperparameters['epochs'])) # BH
        all_bh_2_mdds = np.zeros(shape=(len(self.nets_list), self.iperparameters['epochs'])) # BH

        # precision delta
        all_longs_precisions = np.zeros(shape=(len(self.nets_list), self.iperparameters['epochs']))
        all_shorts_precisions = np.zeros(shape=(len(self.nets_list), self.iperparameters['epochs']))
        all_holds_precisions = np.zeros(shape=(len(self.nets_list), self.iperparameters['epochs']))
        all_labels_longs_coverage = np.zeros(shape=(len(self.nets_list), self.iperparameters['epochs']))
        all_labels_shorts_coverage = np.zeros(shape=(len(self.nets_list), self.iperparameters['epochs']))

        # precision label
        all_longs_precisions_ol = np.zeros(shape=(len(self.nets_list), self.iperparameters['epochs']))
        all_shorts_precisions_ol = np.zeros(shape=(len(self.nets_list), self.iperparameters['epochs']))
        all_holds_precisions_ol = np.zeros(shape=(len(self.nets_list), self.iperparameters['epochs']))
        all_labels_longs_coverage_ol = np.zeros(shape=(len(self.nets_list), self.iperparameters['epochs']))
        all_labels_shorts_coverage_ol = np.zeros(shape=(len(self.nets_list), self.iperparameters['epochs']))
        all_labels_holds_coverage_ol = np.zeros(shape=(len(self.nets_list), self.iperparameters['epochs']))

        # % di operazioni fatte 
        all_long_operations = np.zeros(shape=(len(self.nets_list), self.iperparameters['epochs']))
        all_short_operations = np.zeros(shape=(len(self.nets_list), self.iperparameters['epochs']))
        all_hold_operations = np.zeros(shape=(len(self.nets_list), self.iperparameters['epochs']))

        # Precision over coverage
        all_longs_poc = np.zeros(shape=(len(self.nets_list), self.iperparameters['epochs']))
        all_shorts_poc = np.zeros(shape=(len(self.nets_list), self.iperparameters['epochs']))
        all_holds_poc = np.zeros(shape=(len(self.nets_list), self.iperparameters['epochs']))
        
        # Precision over label
        all_longs_pol = np.zeros(shape=(len(self.nets_list), self.iperparameters['epochs']))
        all_shorts_pol = np.zeros(shape=(len(self.nets_list), self.iperparameters['epochs']))
        all_holds_pol = np.zeros(shape=(len(self.nets_list), self.iperparameters['epochs']))

        #accuracy
        all_accuracy = np.zeros(shape=(len(self.nets_list), self.iperparameters['epochs']))

        # SORTINO
        all_ls_sortinos = np.zeros(shape=(len(self.nets_list), self.iperparameters['epochs']))
        all_lh_sortinos = np.zeros(shape=(len(self.nets_list), self.iperparameters['epochs']))
        all_sh_sortinos = np.zeros(shape=(len(self.nets_list), self.iperparameters['epochs']))
        all_bh_sortinos = np.zeros(shape=(len(self.nets_list), self.iperparameters['epochs']))
        
        # SHARPE
        all_ls_sharpes = np.zeros(shape=(len(self.nets_list), self.iperparameters['epochs']))
        all_lh_sharpes = np.zeros(shape=(len(self.nets_list), self.iperparameters['epochs']))
        all_sh_sharpes = np.zeros(shape=(len(self.nets_list), self.iperparameters['epochs']))
        all_bh_sharpes = np.zeros(shape=(len(self.nets_list), self.iperparameters['epochs']))

        input_path = self.experiment_original_path + self.iperparameters['experiment_name'] + '/calculated_metrics/' + type + '/all_walk/' 

        for index_net in range(0, len(self.nets_list)):
            net_json = {}
            with open(input_path + 'net_' + str(index_net) + '.json') as json_file:
                net_json = json.load(json_file)
            

            # RETURNS
            all_ls_returns[index_net] = net_json['ls_returns']
            all_lh_returns[index_net] = net_json['lh_returns']
            all_sh_returns[index_net] = net_json['sh_returns']
            all_bh_return[index_net] = net_json['bh_returns'] # BH
            all_bh_2_return[index_net] = net_json['bh_2_returns'] # BH

            # ROMADS
            all_ls_romads[index_net] = net_json['ls_romads']
            all_lh_romads[index_net] = net_json['lh_romads']
            all_sh_romads[index_net] = net_json['sh_romads']
            all_bh_romads[index_net] = net_json['bh_romads'] # BH
            all_bh_2_romads[index_net] = net_json['bh_2_romads'] # BH

            # MDDS
            all_ls_mdds[index_net] = net_json['ls_mdds']
            all_lh_mdds[index_net] = net_json['lh_mdds']
            all_sh_mdds[index_net] = net_json['sh_mdds']
            all_bh_mdds[index_net] = net_json['bh_mdds'] #BH
            all_bh_2_mdds[index_net] = net_json['bh_2_mdds'] #BH

            # precision delta
            all_longs_precisions[index_net] = net_json['longs_precisions']
            all_shorts_precisions[index_net] = net_json['shorts_precisions']
            all_holds_precisions[index_net] = net_json['holds_precisions']
            all_labels_longs_coverage[index_net] = net_json['longs_label_coverage']
            all_labels_shorts_coverage[index_net] = net_json['shorts_label_coverage']

            # precision label
            all_longs_precisions_ol[index_net] = net_json['longs_precisions_ol']
            all_shorts_precisions_ol[index_net] = net_json['shorts_precisions_ol']
            all_holds_precisions_ol[index_net] = net_json['holds_precisions_ol']
            all_labels_longs_coverage_ol[index_net] = net_json['longs_label_coverage_ol']
            all_labels_shorts_coverage_ol[index_net] = net_json['shorts_label_coverage_ol']
            all_labels_holds_coverage_ol[index_net] = net_json['holds_label_coverage_ol']

            # % di operazioni fatte
            all_long_operations[index_net] = net_json['long_operations']
            all_short_operations[index_net] = net_json['short_operations']
            all_hold_operations[index_net] = net_json['hold_operations']

            # precision over delta
            all_longs_poc[index_net] = net_json['longs_poc']
            all_shorts_poc[index_net] = net_json['shorts_poc']
            all_holds_poc[index_net] = net_json['holds_poc']

            all_longs_pol[index_net] = net_json['longs_pol']
            all_shorts_pol[index_net] = net_json['shorts_pol']
            all_holds_pol[index_net] = net_json['holds_pol']

            # SORTINO
            all_ls_sortinos[index_net] = net_json['ls_sortinos']
            all_lh_sortinos[index_net] = net_json['lh_sortinos']
            all_sh_sortinos[index_net] = net_json['sh_sortinos']
            all_bh_sortinos[index_net] = net_json['bh_sortinos']
            
            # SHARPE
            all_ls_sharpes[index_net] = net_json['ls_sharpes']
            all_lh_sharpes[index_net] = net_json['lh_sharpes']
            all_sh_sharpes[index_net] = net_json['sh_sharpes']
            all_bh_sharpes[index_net] = net_json['bh_sharpes']
            #accuracy
            all_accuracy[index_net] = net_json['accuracy']

        # RETURNS
        avg_ls_returns = np.around(np.average(all_ls_returns, axis=0), decimals=3)
        avg_lh_returns = np.around(np.average(all_lh_returns, axis=0), decimals=3)
        avg_sh_returns = np.around(np.average(all_sh_returns, axis=0), decimals=3)
        avg_bh_returns = np.average(all_bh_return, axis=0) # BH
        avg_bh_2_returns = np.average(all_bh_2_return, axis=0) # BH

        # MDDS
        avg_ls_mdds = np.around(np.average(all_ls_mdds, axis=0), decimals=3)
        avg_lh_mdds = np.around(np.average(all_lh_mdds, axis=0), decimals=3)
        avg_sh_mdds = np.around(np.average(all_sh_mdds, axis=0), decimals=3)
        avg_bh_mdds = np.average(all_bh_mdds, axis=0) # BH
        avg_bh_2_mdds = np.average(all_bh_2_mdds, axis=0) # BH

        # ROMADS
        avg_ls_romads = np.divide(avg_ls_returns, avg_ls_mdds, out=np.zeros_like(avg_ls_returns), where=avg_ls_mdds!=0)
        avg_lh_romads = np.divide(avg_lh_returns, avg_lh_mdds, out=np.zeros_like(avg_lh_returns), where=avg_lh_mdds!=0)
        avg_sh_romads = np.divide(avg_sh_returns, avg_sh_mdds, out=np.zeros_like(avg_sh_returns), where=avg_sh_mdds!=0)
        avg_bh_romads = np.divide(avg_bh_returns, avg_bh_mdds, out=np.zeros_like(avg_bh_returns), where=avg_bh_mdds!=0)
        avg_bh_2_romads = np.divide(avg_bh_2_returns, avg_bh_2_mdds, out=np.zeros_like(avg_bh_2_returns), where=avg_bh_2_mdds!=0)

        # rimuovo i nan dai romads
        avg_ls_romads = np.around(np.nan_to_num(avg_ls_romads), decimals=3)
        avg_lh_romads = np.around(np.nan_to_num(avg_lh_romads), decimals=3)
        avg_sh_romads = np.around(np.nan_to_num(avg_sh_romads), decimals=3)
        avg_sh_romads[~np.isfinite(avg_sh_romads)] = 0

        # Precision delta
        avg_longs_precisions = np.around(np.average(all_longs_precisions, axis=0), decimals=3)
        avg_shorts_precisions = np.around(np.average(all_shorts_precisions, axis=0), decimals=3)
        avg_holds_precisions = np.around(np.average(all_holds_precisions, axis=0), decimals=3)
        avg_label_longs_coverage = np.around(np.average(all_labels_longs_coverage, axis=0), decimals=3)
        avg_label_shorts_coverage = np.around(np.average(all_labels_shorts_coverage, axis=0), decimals=3)

         # PRECISION LABEL
        avg_longs_precisions_ol = np.around(np.average(all_longs_precisions_ol, axis=0), decimals=3)
        avg_shorts_precisions_ol = np.around(np.average(all_shorts_precisions_ol, axis=0), decimals=3)
        avg_holds_precisions_ol = np.around(np.average(all_holds_precisions_ol, axis=0), decimals=3)
        avg_longs_label_coverage_ol = np.around(np.average(all_labels_longs_coverage_ol, axis=0), decimals=3)
        avg_shorts_label_coverage_ol = np.around(np.average(all_labels_shorts_coverage_ol, axis=0), decimals=3)
        avg_holds_label_coverage_ol = np.around(np.average(all_labels_holds_coverage_ol, axis=0), decimals=3)

        # precision over delta
        avg_longs_poc = np.around(np.divide(avg_longs_precisions, avg_label_longs_coverage), decimals=3)
        avg_shorts_poc = np.around(np.divide(avg_shorts_precisions, avg_label_shorts_coverage), decimals=3)
        avg_holds_poc = np.around(np.divide(avg_holds_precisions, avg_label_shorts_coverage), decimals=3)
        avg_longs_poc = (avg_longs_poc - 1 ) * 100
        avg_shorts_poc = (avg_shorts_poc - 1 ) * 100
        avg_holds_poc = (avg_holds_poc - 1 ) * 100
        
        # precision over label
        avg_longs_pol = np.around(np.divide(avg_longs_precisions_ol, avg_longs_label_coverage_ol), decimals=3) if np.count_nonzero(avg_longs_label_coverage_ol) > 0 else np.zeros(shape=(self.iperparameters['epochs']))
        avg_shorts_pol = np.around(np.divide(avg_shorts_precisions_ol, avg_shorts_label_coverage_ol), decimals=3) if np.count_nonzero(avg_shorts_label_coverage_ol) > 0 else np.zeros(shape=(self.iperparameters['epochs']))
        avg_holds_pol = np.around(np.divide(avg_holds_precisions_ol, avg_holds_label_coverage_ol), decimals=3) if np.count_nonzero(avg_holds_label_coverage_ol) > 0 else np.zeros(shape=(self.iperparameters['epochs']))
        avg_longs_pol = (avg_longs_pol - 1 ) * 100
        avg_shorts_pol = (avg_shorts_pol - 1 ) * 100
        avg_holds_pol = (avg_holds_pol - 1 ) * 100

        #accuracy 
        avg_accuracy = np.around(np.average(all_accuracy, axis=0), decimals=3) 

        avg_long_operations = np.average(all_long_operations, axis=0)
        avg_short_operations= np.average(all_short_operations, axis=0)
        avg_hold_operations = np.average(all_hold_operations, axis=0)

        # SORTINO
        avg_ls_sortinos = np.around(np.average(all_ls_sortinos, axis=0), decimals=3)
        avg_lh_sortinos = np.around(np.average(all_lh_sortinos, axis=0), decimals=3)
        avg_sh_sortinos = np.around(np.average(all_sh_sortinos, axis=0), decimals=3)
        avg_bh_sortinos = np.around(np.average(all_bh_sortinos, axis=0), decimals=3)
        
        # SHARPE
        avg_ls_sharpes = np.around(np.average(all_ls_sharpes, axis=0), decimals=3)
        avg_lh_sharpes = np.around(np.average(all_lh_sharpes, axis=0), decimals=3)
        avg_sh_sharpes = np.around(np.average(all_sh_sharpes, axis=0), decimals=3)
        avg_bh_sharpes = np.around(np.average(all_bh_sharpes, axis=0), decimals=3)

        avg_json = {
            "ls_returns": avg_ls_returns.tolist(),
            "lh_returns": avg_lh_returns.tolist(),
            "sh_returns": avg_sh_returns.tolist(),
            "bh_returns": avg_bh_returns.tolist(), # BH
            "bh_2_returns": avg_bh_2_returns.tolist(), # BH

            "ls_romads": avg_ls_romads.tolist(),
            "lh_romads": avg_lh_romads.tolist(),
            "sh_romads": avg_sh_romads.tolist(),
            "bh_romads": avg_bh_romads.tolist(), # BH
            "bh_2_romads": avg_bh_2_romads.tolist(), # BH

            "ls_mdds": avg_ls_mdds.tolist(),
            "lh_mdds": avg_lh_mdds.tolist(),
            "sh_mdds": avg_sh_mdds.tolist(),
            "bh_mdds": avg_bh_mdds.tolist(), # BH
            "bh_2_mdds": avg_bh_2_mdds.tolist(), # BH

            # precision delta
            "longs_precisions": avg_longs_precisions.tolist(),
            "shorts_precisions": avg_shorts_precisions.tolist(),
            "holds_precisions": avg_holds_precisions.tolist(),
            "longs_label_coverage": avg_label_longs_coverage.tolist(),
            "shorts_label_coverage": avg_label_shorts_coverage.tolist(),

            # precision label
            "longs_precisions_ol": avg_longs_precisions_ol.tolist(),
            "shorts_precisions_ol": avg_shorts_precisions_ol.tolist(),
            "holds_precisions_ol": avg_holds_precisions_ol.tolist(),
            "longs_label_coverage_ol": avg_longs_label_coverage_ol.tolist(),
            "shorts_label_coverage_ol": avg_shorts_label_coverage_ol.tolist(),
            "holds_label_coverage_ol": avg_holds_label_coverage_ol.tolist(),

            "long_operations": avg_long_operations.tolist(),
            "short_operations": avg_short_operations.tolist(),
            "hold_operations": avg_hold_operations.tolist(),

            "longs_poc": avg_longs_poc.tolist(),
            "shorts_poc": avg_shorts_poc.tolist(),
            "holds_poc": avg_holds_poc.tolist(),

            "longs_pol": avg_longs_pol.tolist(),
            "shorts_pol": avg_shorts_pol.tolist(),
            "holds_pol": avg_holds_pol.tolist(),

            "accuracy": avg_accuracy.tolist(),

            "ls_sortinos": avg_ls_sortinos.tolist(),
            "lh_sortinos": avg_lh_sortinos.tolist(),
            "sh_sortinos": avg_sh_sortinos.tolist(),
            "bh_sortinos": avg_bh_sortinos.tolist(), # BH
            
            "ls_sharpes": avg_ls_sharpes.tolist(),
            "lh_sharpes": avg_lh_sharpes.tolist(),
            "sh_sharpes": avg_sh_sharpes.tolist(),
            "bh_sharpes": avg_bh_sharpes.tolist() # BH  
        }

        avg_output_path = self.experiment_original_path + self.iperparameters['experiment_name'] + '/calculated_metrics/' + type + '/average/' 
        
        create_folder(avg_output_path)

        with open(avg_output_path + 'all_walk.json', 'w') as json_file:
            json.dump(avg_json, json_file, indent=4)
            #json.dump(net_json, json_file)
        
        print(self.iperparameters['experiment_name'], '|', type, "| Salvate le metriche AVG per tutti gli walk")

        do_plot(metrics=avg_json, walk="all", epochs=self.iperparameters['epochs'], main_path=self.experiment_original_path, experiment_name=self.iperparameters['experiment_name'], net='average', type=type) 

        #return avg_json
    
    
    '''
    ' Genero i file json per la loss
    '''
    def generate_loss_json(self):
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
        
            print(self.iperparameters['experiment_name'], '|', "Salvato file Json Loss per walk n° ", index_walk)

    '''
    ' Genero i file json per l'accuracy
    '''
    def generate_accuracy_json(self):
        for index_walk in range(0, len(self.walks_list)):
            avg_training_accuracy = []
            avg_validation_accuracy = []
            avg_test_accuracy = []

            walk_str = 'walk_' + str(index_walk)
            date_list, epochs_list = self.get_date_epochs_walk(path=self.original_predictions_validation_folder, walk=walk_str)

            # RETURNS
            all_training_accuracy = np.zeros(shape=(len(self.nets_list), len(epochs_list)))
            all_validation_accuracy = np.zeros(shape=(len(self.nets_list), len(epochs_list)))
            all_test_accuracy = np.zeros(shape=(len(self.nets_list), len(epochs_list)))

            for index_net in range(0, len(self.nets_list)):
                net = 'net_' + str(index_net) + '.csv'
                df = pd.read_csv(self.experiment_original_path + self.iperparameters['experiment_name'] + '/predictions/loss_during_training/walk_' + str(index_walk) + '/' + net)

                df['training_accuracy'] = df['training_accuracy'] * 100
                df['validation_accuracy'] = df['validation_accuracy'] * 100
                df['test_accuracy'] = df['test_accuracy'] * 100

                json_log = {
                    'training_accuracy': df['training_accuracy'].tolist(),
                    'validation_accuracy': df['validation_accuracy'].tolist(),
                    'test_accuracy': df['test_accuracy'].tolist()
                }

                output_path = self.experiment_original_path + self.iperparameters['experiment_name'] + '/calculated_metrics/accuracy/walk_' + str(index_walk) + '/' 

                create_folder(output_path)

                with open(output_path + 'net_' + str(index_net) + '.json', 'w') as json_file:
                    json.dump(json_log, json_file, indent=4)

                all_training_accuracy[index_net] = df['training_accuracy'].tolist()
                all_validation_accuracy[index_net] = df['validation_accuracy'].tolist()
                all_test_accuracy[index_net] = df['test_accuracy'].tolist()
            
            avg_training_accuracy = np.around(np.average(all_training_accuracy, axis=0), decimals=3)
            avg_validation_accuracy = np.around(np.average(all_validation_accuracy, axis=0), decimals=3)
            avg_test_accuracy = np.around(np.average(all_test_accuracy, axis=0), decimals=3)

            avg_json_log = {
                    'training_accuracy': avg_training_accuracy.tolist(),
                    'validation_accuracy': avg_validation_accuracy.tolist(),
                    'test_accuracy': avg_test_accuracy.tolist()
                }

            output_path = self.experiment_original_path + self.iperparameters['experiment_name'] + '/calculated_metrics/accuracy/average/' 

            create_folder(output_path)

            with open(output_path + 'walk_' + str(index_walk) + '.json', 'w') as json_file:
                json.dump(avg_json_log, json_file, indent=4)  

            print(self.iperparameters['experiment_name'], '|', "Salvato file Json Accuracy per walk n° ", index_walk)
    
    '''
    ' Genera tutti i json con le metriche
    ' single_net = True li rigenera anche per le reti singole 
    ' 
    '
    '''
    def generate_json(self, single_net=False):
        if single_net == True:
            for index_walk in range(0, len(self.walks_list)):
                for index_net in range(0, len(self.nets_list)):
                    print("Generating single net n°", index_net, "at walk n°", index_walk)

                    #t1 = threading.Thread(target=self.generate_single_net_json, args=([index_walk, index_net, 'validation', 0, 0]))
                    t2 = threading.Thread(target=self.generate_single_net_json, args=([index_walk, index_net, 'test', 0, 0]))
                    #t1.start()
                    t2.start()
                    #t1.join()
                    t2.join()
        
        # avg delle single net
        self.generate_avg_json(type='validation')
        self.generate_avg_json(type='test')
        

        # I CSV di loss e accuracy vegnono creati a runtime durante il training. Non posso generare quei json senza i csv
        #if os.path.isfile(self.experiment_original_path + self.iperparameters['experiment_name'] + '/predictions/loss_during_training/'):
        print("Generating loss and accuracy json")
        self.generate_loss_json()
        self.generate_accuracy_json()
        #self.generate_precision_over_label_json()
        
        '''
        for index_net in range(0, len(self.nets_list)):
            print("Generating all walk json for net", index_net)

            t1 = threading.Thread(target=self.generate_single_net_json_all_walk, args=([index_net, 'validation', 0, 0]))
            t2 = threading.Thread(target=self.generate_single_net_json_all_walk, args=([index_net, 'test', 0, 0]))
            t1.start()
            t2.start()
            t1.join()
            t2.join()
            
            self.generate_single_net_json_all_walk(index_net=index_net, type='validation', penalty=0, stop_loss=0)
            self.generate_single_net_json_all_walk(index_net=index_net, type='test', penalty=0, stop_loss=0)
            
        print("Generating all walk avg all nets")
        self.generate_avg_json_all_walk(type='validation')
        self.generate_avg_json_all_walk(type='test')
        '''


        ''' Tolto di default
        t1 = threading.Thread(target=self.generate_avg_json, args=(['validation']))
        t2 = threading.Thread(target=self.generate_avg_json, args=(['test']))
        t3 = threading.Thread(target=self.generate_loss_json)
        t4 = threading.Thread(target=self.generate_accuracy_json)
        
        t1.start()
        t2.start()
        t3.start()
        t4.start()
        t1.join()
        t2.join()
        t3.join()
        t4.join()
        '''
        

    '''

    ░██╗░░░░░░░██╗███████╗██████╗░░█████╗░██████╗░██████╗░  ░█████╗░██████╗░██╗
    ░██║░░██╗░░██║██╔════╝██╔══██╗██╔══██╗██╔══██╗██╔══██╗  ██╔══██╗██╔══██╗██║
    ░╚██╗████╗██╔╝█████╗░░██████╦╝███████║██████╔╝██████╔╝  ███████║██████╔╝██║
    ░░████╔═████║░██╔══╝░░██╔══██╗██╔══██║██╔═══╝░██╔═══╝░  ██╔══██║██╔═══╝░██║
    ░░╚██╔╝░╚██╔╝░███████╗██████╦╝██║░░██║██║░░░░░██║░░░░░  ██║░░██║██║░░░░░██║
    ░░░╚═╝░░░╚═╝░░╚══════╝╚═════╝░╚═╝░░╚═╝╚═╝░░░░░╚═╝░░░░░  ╚═╝░░╚═╝╚═╝░░░░░╚═╝
    '''

    '''
    ' Usata per generare i dati subfilter period 
    ' per Hawkeye
    '''
    def get_result_for_walk_net(self, start_date, end_date, penalty=32, stop_loss=1000, index_walk=0, net=0, epoch=1, set_type='validation'):

        if index_walk < 0  or index_walk > len(self.walks_list):
            return {}
        
        if net < 0  or net > len(self.nets_list):
            return {}

        walk_str = 'walk_' + str(index_walk)
        net_str = 'net_' + str(net) + '.csv'

        date_list, epochs_list = self.get_date_epochs_walk(path=self.original_predictions_test_folder, walk=walk_str)

        if epoch < 1  or epoch > len(epochs_list) + 1:
            return {}

        # leggo le predizioni fatte con l'esnemble
        df = pd.read_csv(self.experiment_original_path + self.iperparameters['experiment_name'] + '/predictions/predictions_during_training/' + set_type + '/' + walk_str + '/' + net_str)

        # mergio con le label, così ho un subset del df con le date che mi servono e la predizione 
        df_merge_with_label = df_date_merger(df=df, columns=['date_time', 'delta_next_day', 'delta_current_day', 'close', 'open', 'high', 'low'], dataset=self.iperparameters['predictions_dataset'], thr_hold=self.iperparameters['hold_labeling'])
        df_merge_with_label = Market.get_df_by_data_range(df=df_merge_with_label, start_date=start_date, end_date=end_date)

        if df_merge_with_label.shape[0] == 0:
            net_json = {
                "ls_return": 0,
                "lh_return": 0,
                "sh_return": 0,
                "bh_return": 0,
                "bh_2_return": 0,

                "ls_romad": 0,
                "lh_romad": 0,
                "sh_romad": 0,
                "bh_romad": 0,  
                "bh_2_romad": 0,

                "ls_mdd": 0,
                "lh_mdd": 0,
                "sh_mdd": 0,
                "bh_mdd": 0,
                "bh_mdd": 0,

                "longs_precision": 0,
                "shorts_precision": 0,
                "longs_label_coverage": 0,
                "shorts_label_coverage": 0,

                "long_operation": 0,
                "short_operation": 0,
                "hold_operation": 0,

                "long_poc": 0,
                "short_poc": 0,
                "accuracy": 0,

                "ls_equity_line": [],            
                "lh_equity_line": [],            
                "sh_equity_line": [],
                "bh_equity_line": [],
                "bh_2_equity_line": [],

                "date_list": []           
            }

            return net_json

        label_coverage = Measures.get_delta_coverage(delta=df_merge_with_label['delta_next_day'].tolist())

        bh_equity_line, bh_global_return, bh_mdd, bh_romad, bh_i, bh_j  = Measures.get_return_mdd_romad_bh(close=df_merge_with_label['close'].tolist(), multiplier=self.iperparameters['return_multiplier'])

        date_list = df_merge_with_label['date_time'].tolist()

        df_epoch_rename = df_merge_with_label.copy()
        df_epoch_rename = df_epoch_rename.rename(columns={'epoch_' + str(epoch): 'decision'})
        
        # ALLINEO LE DATE PER UTILIZZARE DATA - DECISIONE SULLA STESSA LINEA [rende i risultati uguali a MC ma diversi dai plot]
        #df_epoch_rename['decision'] = df_epoch_rename['decision'].shift(1)
        #df_epoch_rename = df_epoch_rename.dropna()
        #df_epoch_rename['decision'] = df_epoch_rename['decision'].astype(int)

        # Effettuo i calcoli. Usare delta_current_day se si usano le tre righe di su per allineare i dati con MC
        ls_equity_line, ls_global_return, ls_mdd, ls_romad, ls_i, ls_j  = Measures.get_equity_return_mdd_romad(df=df_epoch_rename.copy(), multiplier=self.iperparameters['return_multiplier'], type='long_short', penalty=penalty, stop_loss=stop_loss, delta_to_use='delta_next_day')
        lh_equity_line, lh_global_return, lh_mdd, lh_romad, lh_i, lh_j  = Measures.get_equity_return_mdd_romad(df=df_epoch_rename.copy(), multiplier=self.iperparameters['return_multiplier'], type='long_only', penalty=penalty, stop_loss=stop_loss, delta_to_use='delta_next_day')
        sh_equity_line, sh_global_return, sh_mdd, sh_romad, sh_i, sh_j  = Measures.get_equity_return_mdd_romad(df=df_epoch_rename.copy(), multiplier=self.iperparameters['return_multiplier'], type='short_only', penalty=penalty, stop_loss=stop_loss, delta_to_use='delta_next_day')

        bh_2_equity_line, bh_2_global_return, bh_2_mdd, bh_2_romad, bh_2_i, bh_2_j  = Measures.get_equity_return_mdd_romad(df=df_epoch_rename.copy(), multiplier=self.iperparameters['return_multiplier'], type='bh_long', penalty=penalty, stop_loss=stop_loss, delta_to_use='delta_next_day')
        
        long, short, hold, general = Measures.get_precision_count_coverage(df=df_epoch_rename.copy(), multiplier=self.iperparameters['return_multiplier'], stop_loss=0, penalty=0, delta_to_use='delta_next_day')
        long_poc, short_poc = Measures.get_precision_over_coverage(df=df_epoch_rename.copy(), multiplier=self.iperparameters['return_multiplier'], stop_loss=0, penalty=0, delta_to_use='delta_next_day')

        net_json = {
            "ls_return": ls_global_return,
            "lh_return": lh_global_return,
            "sh_return": sh_global_return,
            "bh_return": bh_global_return,
            "bh_2_return": bh_2_global_return,

            "ls_romad": ls_romad,
            "lh_romad": lh_romad,
            "sh_romad": sh_romad,
            "bh_romad": bh_romad,
            "bh_2_romad": bh_2_romad, 

            "ls_mdd": ls_mdd,
            "lh_mdd": lh_mdd,
            "sh_mdd": sh_mdd,
            "bh_mdd": bh_mdd,
            "bh_2_mdd": bh_2_mdd, 

            "longs_precision": long['precision'] * 100,
            "shorts_precision": short['precision'] * 100,
            "longs_label_coverage": label_coverage['long'] * 100,
            "shorts_label_coverage": label_coverage['short'] * 100,

            "long_operation": long['coverage'] * 100,
            "short_operation": short['coverage'] * 100,
            "hold_operation": hold['coverage'] * 100,

            "long_poc": long_poc,
            "short_poc": short_poc,
            "accuracy": general['accuracy'] * 100,

            "ls_equity_line": ls_equity_line.tolist(),            
            "lh_equity_line": lh_equity_line.tolist(),            
            "sh_equity_line": sh_equity_line.tolist(),
            "bh_equity_line": bh_equity_line,
            "bh_2_equity_line": bh_2_equity_line.tolist(), 

            "date_list": date_list           
        }

        return net_json

    '''
    ' Usato per generare i dati per il modal Swipe 
    ' di Hawkeye. Codice in parte preso da get_report_excel_swipe
    ' decision_folder = [valiation, test, test_alg_4]
    '''
    def get_result_swipe(self, type, thrs_ensemble_magg=[], thrs_ensemble_exclusive=[], thrs_ensemble_elimination=[], 
        epoch_selection_policy='long_short', decision_folder='test', stop_loss=1000, penalty=32):
        thr_global = []

        if type == 'ensemble_magg':
            thr_global = thrs_ensemble_magg
        
        if type == 'ensemble_exclusive':
            thr_global = thrs_ensemble_exclusive

        if type == 'ensemble_exclusive_short':
            thr_global = thrs_ensemble_exclusive

        if type == 'ensemble_elimination':
            thr_global = thrs_ensemble_elimination


        ls_results = [dict() for x in range(len(thr_global))] 
        l_results = [dict() for x in range(len(thr_global))] 
        s_results = [dict() for x in range(len(thr_global))] 
        bh_results = [dict() for x in range(len(thr_global))] 
        bh_intraday_results = [dict() for x in range(len(thr_global))] 
        general_info = [dict() for x in range(len(thr_global))] 

        if type == 'ensemble_magg':
            for i, thr in enumerate(thr_global):
                ls_results[i], l_results[i], s_results[i], bh_results[i], bh_intraday_results[i], general_info[i] = self.get_results(ensemble_type=type, epoch_selection_policy=epoch_selection_policy, 
                    thr_ensemble_magg=thr, stop_loss=stop_loss, penalty=penalty, decision_folder=decision_folder)

        if type == 'ensemble_exclusive':
            for i, thr in enumerate(thr_global):
                ls_results[i], l_results[i], s_results[i], bh_results[i], bh_intraday_results[i], general_info[i] = self.get_results(ensemble_type=type, epoch_selection_policy=epoch_selection_policy, 
                    thr_ensemble_exclusive=thr, stop_loss=stop_loss, penalty=penalty, decision_folder=decision_folder)

        if type == 'ensemble_exclusive_short':
            for i, thr in enumerate(thr_global):
                ls_results[i], l_results[i], s_results[i], bh_results[i], bh_intraday_results[i], general_info[i] = self.get_results(ensemble_type=type, epoch_selection_policy=epoch_selection_policy, 
                    thr_ensemble_exclusive=thr, stop_loss=stop_loss, penalty=penalty, decision_folder=decision_folder)
        
        result = {
            'thr': thr_global,
            
            'ls_returns' : [x['return'] for x in ls_results], 
            'ls_mdds' : [x['mdd'] for x in ls_results], 
            'ls_romads' : [x['romad'] for x in ls_results], 
            
            'lh_returns' : [x['return'] for x in l_results], 
            'lh_mdds' : [x['mdd'] for x in l_results], 
            'lh_romads' : [x['romad'] for x in l_results], 

            'sh_returns' : [x['return'] for x in s_results], 
            'sh_mdds' : [x['mdd'] for x in s_results], 
            'sh_romads' : [x['romad'] for x in s_results], 
            
            'bh_returns' : [x['return'] for x in bh_results], 
            'bh_mdds' : [x['mdd'] for x in bh_results], 
            'bh_romads' : [x['romad'] for x in bh_results], 

            'bh_2_returns' : [x['return'] for x in bh_intraday_results], 
            'bh_2_mdds' : [x['mdd'] for x in bh_intraday_results], 
            'bh_2_romads' : [x['romad'] for x in bh_intraday_results], 
            
            'long_operations': [x['long_info']['coverage'] for x in general_info],
            'short_operations': [x['short_info']['coverage'] for x in general_info],
            'hold_operations': [x['hold_info']['coverage'] for x in general_info],

            'longs_precisions': [x['long_info']['precision'] for x in general_info],
            'longs_label_coverage': [x['long_info']['random_perc'] for x in general_info],
            'shorts_precisions': [x['short_info']['precision'] for x in general_info],
            'shorts_label_coverage': [x['short_info']['random_perc'] for x in general_info]
        }
        return result
        
    '''
    ' Usato per generare i dati per il modal Swipe 
    ' di Hawkeye. Codice in parte preso da get_report_excel_swipe
    ' decision_folder = [valiation, test, test_alg_4]
    '''
    def get_result_swipe_vix(self, thrs, stop_loss=1000, penalty=32):

        ls_results = [dict() for x in range(len(thrs))] 
        l_results = [dict() for x in range(len(thrs))] 
        s_results = [dict() for x in range(len(thrs))] 
        bh_results = [dict() for x in range(len(thrs))] 
        bh_intraday_results = [dict() for x in range(len(thrs))] 
        general_info = [dict() for x in range(len(thrs))] 

        for i, thr in enumerate(thrs):
            ls_results[i], l_results[i], s_results[i], bh_results[i], bh_intraday_results[i], general_info[i] = self.get_results_vix(thr=thr, stop_loss=stop_loss, penalty=penalty)
        
        result = {
            'thr': thrs,
            
            'ls_returns' : [x['return'] for x in ls_results], 
            'ls_mdds' : [x['mdd'] for x in ls_results], 
            'ls_romads' : [x['romad'] for x in ls_results], 
            
            'lh_returns' : [x['return'] for x in l_results], 
            'lh_mdds' : [x['mdd'] for x in l_results], 
            'lh_romads' : [x['romad'] for x in l_results], 

            'sh_returns' : [x['return'] for x in s_results], 
            'sh_mdds' : [x['mdd'] for x in s_results], 
            'sh_romads' : [x['romad'] for x in s_results], 
            
            'bh_returns' : [x['return'] for x in bh_results], 
            'bh_mdds' : [x['mdd'] for x in bh_results], 
            'bh_romads' : [x['romad'] for x in bh_results], 

            'bh_2_returns' : [x['return'] for x in bh_intraday_results], 
            'bh_2_mdds' : [x['mdd'] for x in bh_intraday_results], 
            'bh_2_romads' : [x['romad'] for x in bh_intraday_results], 
            
            'long_operations': [x['long_info']['coverage'] for x in general_info],
            'short_operations': [x['short_info']['coverage'] for x in general_info],
            'hold_operations': [x['hold_info']['coverage'] for x in general_info],

            'longs_precisions': [x['long_info']['precision'] for x in general_info],
            'longs_label_coverage': [x['long_info']['random_perc'] for x in general_info],
            'shorts_precisions': [x['short_info']['precision'] for x in general_info],
            'shorts_label_coverage': [x['short_info']['random_perc'] for x in general_info]
        }
        return result

    '''

    ░██████╗███████╗██╗░░░░░███████╗░█████╗░████████╗██╗░█████╗░███╗░░██╗
    ██╔════╝██╔════╝██║░░░░░██╔════╝██╔══██╗╚══██╔══╝██║██╔══██╗████╗░██║
    ╚█████╗░█████╗░░██║░░░░░█████╗░░██║░░╚═╝░░░██║░░░██║██║░░██║██╔██╗██║
    ░╚═══██╗██╔══╝░░██║░░░░░██╔══╝░░██║░░██╗░░░██║░░░██║██║░░██║██║╚████║
    ██████╔╝███████╗███████╗███████╗╚█████╔╝░░░██║░░░██║╚█████╔╝██║░╚███║
    ╚═════╝░╚══════╝╚══════╝╚══════╝░╚════╝░░░░╚═╝░░░╚═╝░╚════╝░╚═╝░░╚══╝
    '''

    '''
    ' Legge i CSV generati da sebastian "selection"
    ' li ordina secondo una metrica es. in ordine di romad decrescente
    ' Seleziona i primi top_number e utilizza quelle reti con quell'epoca 
    ' Calcola quindi return, mdd, romad e equity line complessima usandoli come 
    ' mini future (quindi divide per 10 il return)
    '''
    def calculate_selection_results(self, second_metric, operation='long', metric='valid_romad', top_number=10, penalty=25, stop_loss=1000, walk=0):

        # legge i csv di seba e seleziona i top_number (tipo i primi 10)
        df_selection = pd.read_csv(self.selection_folder + 'on_finish/long_UNselected.csv', delimiter=';', decimal=',')

        #print(df_selection.shape)

        #df_selection = df_selection.sort_values(by=["valid_por"], ascending=False) # ordino i valori per la metrica  
        df_selection = df_selection.loc[df_selection["valid_romad"] >= 0.8]
        df_selection = df_selection.loc[(df_selection["valid_por"] >= 7) & (df_selection["valid_por"] >= 10)]
        #df_selection = df_selection.loc[(df_selection["valid_cove"] >= 0.35) & (df_selection["valid_cove"] <= 0.65)]
        df_selection = df_selection.loc[(df_selection["valid_cove"] >= 0.4) & (df_selection["valid_cove"] <= 0.5)]
        
        df_selection = df_selection.sort_values(by=["valid_cove"], ascending=True)
        #df_selection = df_selection.sort_values(by=["valid_cove"], ascending=False)
        
        #print(df_selection.shape)

        df_selection = df_selection[:10] # prendo i tot iniziali

        # lista delle reti e delle epoche ad esse associate
        nets = df_selection['net'].tolist()  
        epochs = df_selection['epoch'].tolist()

        
        # Equity totale in cui sommo le singole equity delle reti selezionate
        global_equity = []

        # df in cui concateno tutti i df delle varie reti per calcolare poi precision e general info
        df_global = pd.DataFrame()
        df = pd.DataFrame()
        # Per ogni rete seleziono l'epoca indicata come migliore e calcolo return, romad, mdd
        for idx, net in enumerate(nets):
            df = pd.read_csv(self.original_predictions_test_folder + 'walk_' + str(walk) + '/net_' + str(int(net)) + '.csv')
            df = df_date_merger(df=df.copy(), columns=['date_time', 'delta_next_day', 'delta_current_day', 'close', 'open', 'high', 'low'], dataset=self.iperparameters['predictions_dataset'], thr_hold=self.iperparameters['hold_labeling'])
            
            df = df.rename(columns={'epoch_' + str(int(epochs[idx])) : 'decision'})

            df = df[['date_time', 'decision', 'delta_next_day', 'delta_current_day', 'close', 'open', 'high', 'low', 'label_next_day']]

            if operation == 'long':
                todo_op = 'long_only'

            if operation == 'short':
                todo_op = 'short_only'

            if operation == 'long_short':
                todo_op = 'long_short'

            equity_line, global_return, mdd, romad, i, j = Measures.get_equity_return_mdd_romad(df=df.copy(), multiplier=self.iperparameters['return_multiplier'], type=todo_op, penalty=penalty, stop_loss=stop_loss, delta_to_use='delta_next_day')
            
            '''
            df2 = df[['date_time', 'decision']]
            df2['decision'] = df2['decision'].shift(-1)
            df2 = df2.dropna()
            df2['decision'] = df2['decision'].astype(int)
            df2.to_csv('C:/Users/Utente/Desktop/controprova/' + str(idx) + '.csv', header=True, index=False)
            
            print('net_' + str(net))
            print('epoch_' + str(epochs[idx]))
            print(df.shape)
            print(global_return)
            input()
            '''


            if global_equity == []:
                global_equity = equity_line
            else: 
                global_equity = list( map(add, global_equity, equity_line) )

            df_global = pd.concat([df_global, df])
        
        # calcolo il BH con l'ultimo DF (tanto è uguale il periodo per qualsiasi rete)
        bh_equity_line, bh_global_return, bh_mdd, bh_romad, bh_i, bh_j = Measures.get_return_mdd_romad_bh(close=df['close'].tolist(), multiplier=self.iperparameters['return_multiplier'])

        # rimuovo le short se voglio analizzare le long
        if operation=='long':
            df_global['decision'] = df_global['decision'].apply(lambda x: 2 if x==2 else 1 )

        # rimuovo le long se voglio analizzare le short
        if operation=='short':
            df_global['decision'] = df_global['decision'].apply(lambda x: 0 if x==0 else 1 )
        
        # calcolo tutte le info per long + short, long + hold e short + hold
        long_info, short_info, hold_info, general_info = Measures.get_precision_count_coverage(df=df_global.copy(), multiplier=self.iperparameters['return_multiplier'], penalty=penalty, stop_loss=stop_loss, delta_to_use='delta_next_day')
        
        general_info = {
            "long_info": long_info,
            "short_info": short_info,
            "hold_info": hold_info,
            "total_trade": general_info['total_trade'],
            "total_guessed_trade": general_info['total_guessed_trade'],
            "total_operation": general_info['total_operation'],
            "dates": df['date_time'].tolist()
        }

        bh_results = {
            "return": bh_global_return,
            "mdd": bh_mdd,
            "romad": bh_romad,
            "equity_line": bh_equity_line
        }


        global_equity = [x / 10 for x in global_equity]

        global_return, global_mdd, global_romad, i, j = Measures.get_return_mdd_romad_from_equity(equity_line=global_equity)

        #print("Global Return:", global_return, "- MDD:", mdd, "- Romad: ", romad)

        return global_equity, global_return, global_mdd, global_romad, general_info, bh_results

    '''
    '
    '''
    def calculate_ensemble_selection_results(self, second_metric, operation='long', metric='valid_romad', top_number=10, penalty=25, stop_loss=1000, walk=0):
        walk = 1
        # legge i csv di seba e seleziona i top_number (tipo i primi 10)
        
        #df_selection = pd.read_csv(self.selection_folder + 'on_finish/walk_' + str(walk) + '/long_short_UNselected.csv')
        df_selection = pd.read_csv(self.selection_folder + 'on_finish/walk_' + str(walk) + '/long_UNselected.csv')

        #print(df_selection.shape)

        #df_selection = df_selection.sort_values(by=["valid_por"], ascending=False) # ordino i valori per la metrica  
        #df_selection = df_selection.sort_values(by=["valid_romad"], ascending=False) # ordino i valori per la metrica  
        
        #df_selection = df_selection.loc[df_selection["valid_romad"] < -0.11] # BH exp 73
        #df_selection = df_selection.loc[df_selection["valid_romad"] > 0.77] # BH exp 76
        #df_selection = df_selection.loc[(df_selection["valid_por"] >= 0) & (df_selection["valid_por"] <= 10)]
        
        # VECCHIO CRITERIO SEBA 
        #df_selection = df_selection.loc[(df_selection["valid_cove"] < 0.5) & (df_selection["valid_por"] > 0) & (df_selection["valid_por"] < 8) & (df_selection["valid_romad"] > 0.1)]
        
        # NUOVO CRITERIO SEBA 08/05/2020 18:25
        #df_selection = df_selection.loc[(df_selection["valid_cove"] > 0.4) & (df_selection["valid_cove"] < 0.6) & (df_selection["valid_por"] > 2) & (df_selection["valid_por"] < 12)]
        #df_selection = df_selection.loc[(df_selection["valid_cove"] > 0.4) & (df_selection["valid_cove"] < 0.6)]
        
        # NUOVO CRITERIO 11/05/2020 
        df_selection = df_selection.loc[(df_selection["valid_cove"] < 0.45) & (df_selection["valid_por"] > 0) & (df_selection["valid_por"] < 10) & (df_selection["valid_romad"] > 0)]


        #df_selection = df_selection.loc[(df_selection["valid_cove"] >= 0.35) & (df_selection["valid_cove"] <= 0.65)]
        #df_selection = df_selection.loc[(df_selection["valid_cove"] >= 0.4) & (df_selection["valid_cove"] <= 0.5)]
        #df_selection = df_selection.sort_values(by=["valid_romad"], ascending=False)
        #df_selection = df_selection.loc[df_selection["valid_cove"] <= 0.4]
        #df_selection = df_selection.sort_values(by=["valid_cove"], ascending=False)

        #print(df_selection.shape)

        #df_selection = df_selection[:10] # prendo i tot iniziali

        # lista delle reti e delle epoche ad esse associate
        nets = df_selection['net'].tolist()  
        epochs = df_selection['epoch'].tolist()

        dates = []
        
        # Equity totale in cui sommo le singole equity delle reti selezionate
        global_equity = []

        # df in cui concateno tutti i df delle varie reti per calcolare poi precision e general info
        df_global = pd.DataFrame()
        df = pd.DataFrame()
        # Per ogni rete seleziono l'epoca indicata come migliore e calcolo return, romad, mdd


        for idx, net in enumerate(nets):
            print("Prendo la rete Selection con id:", idx)
            df = pd.read_csv(self.original_predictions_test_folder + 'walk_' + str(walk) + '/net_' + str(int(net)) + '.csv')
            
            #df = df_date_merger(df=df.copy(), columns=['date_time', 'delta_next_day', 'delta_current_day', 'close', 'open', 'high', 'low'], dataset=self.iperparameters['predictions_dataset'], thr_hold=self.iperparameters['hold_labeling'])
            
            if idx == 0: 
                dates = df['date_time'].tolist()

            decision_column_name = 'decision_' + str(int(epochs[idx])) + '_' + str(idx)
            df = df.rename(columns={'epoch_' + str(int(epochs[idx])) : decision_column_name })

            #df = df[['date_time', decision_column_name, 'delta_next_day', 'delta_current_day', 'close', 'open', 'high', 'low', 'label_next_day']]

            df_global[decision_column_name] = df[decision_column_name]

        print(df_global.shape)

        print("Lancio ensemble classico")
        df_final = self.ensemble_magg_classic(df=df_global, thr=0.3)
        print("Fine ensemble classico")
        df_final['date_time'] = dates

        df_final = df_date_merger(df=df_final.copy(), columns=['date_time', 'delta_next_day', 'close', 'open', 'high', 'low'], dataset=self.iperparameters['predictions_dataset'], thr_hold=self.iperparameters['hold_labeling'])

        df_final.to_csv('C:/Users/Utente/Desktop/73-76-nuovo-criterio/' + self.iperparameters['experiment_name'] + '_' + operation + '_walk_' + str(walk) + '.csv', header=True, index=False)
        
        if operation == 'long':
            todo_op = 'long_only'

        if operation == 'short':
            todo_op = 'short_only'

        if operation == 'long_short':
            todo_op = 'long_short'

        global_equity, global_return, global_mdd, global_romad, i, j = Measures.get_equity_return_mdd_romad(df=df_final.copy(), multiplier=self.iperparameters['return_multiplier'], type=todo_op, penalty=penalty, stop_loss=stop_loss, delta_to_use='delta_next_day')

        # calcolo il BH con l'ultimo DF (tanto è uguale il periodo per qualsiasi rete)
        bh_equity_line, bh_global_return, bh_mdd, bh_romad, bh_i, bh_j = Measures.get_return_mdd_romad_bh(close=df_final['close'].tolist(), multiplier=self.iperparameters['return_multiplier'])

        # rimuovo le short se voglio analizzare le long
        if operation=='long':
            df_final['decision'] = df_final['decision'].apply(lambda x: 2 if x==2 else 1 )

        # rimuovo le long se voglio analizzare le short
        if operation=='short':
            df_final['decision'] = df_final['decision'].apply(lambda x: 0 if x==0 else 1 )
        
        
        # calcolo tutte le info per long + short, long + hold e short + hold
        long_info, short_info, hold_info, general_info = Measures.get_precision_count_coverage(df=df_final.copy(), multiplier=self.iperparameters['return_multiplier'], penalty=penalty, stop_loss=stop_loss, delta_to_use='delta_next_day')
        
        general_info = {
            "long_info": long_info,
            "short_info": short_info,
            "hold_info": hold_info,
            "total_trade": general_info['total_trade'],
            "total_guessed_trade": general_info['total_guessed_trade'],
            "total_operation": general_info['total_operation'],
            "dates": dates
        }

        bh_results = {
            "return": bh_global_return,
            "mdd": bh_mdd,
            "romad": bh_romad,
            "equity_line": bh_equity_line
        }

        return global_equity, global_return, global_mdd, global_romad, general_info, bh_results

    
    '''
    ' Genera i risultati nel formato utilizzato dal metodo 
    ' get_report_excel. 
    ' Utilizza i csv selection generati da Sebastian
    ' ex: (type='selection', epoch_selection_policy='short', stop_loss=1000, penalty=25, report_name=experiment_number + "- Rep Selection - Short Only - SL1000 Pen25")
    '''
    def get_selection_results(self, second_metric="", operation='long', metric='valid_romad', top_number=10, penalty=25, stop_loss=1000, walk=0):
        
        #ls_global_equity, ls_global_return, ls_global_mdd, ls_global_romad, ls_general_info, ls_bh_results = self.calculate_selection_results(second_metric=second_metric, operation="long_short", metric=metric, top_number=top_number, penalty=penalty, stop_loss=stop_loss, walk=walk)
        #l_global_equity, l_global_return, l_global_mdd, l_global_romad, l_general_info, l_bh_results = self.calculate_selection_results(second_metric=second_metric, operation="long", metric=metric, top_number=top_number, penalty=penalty, stop_loss=stop_loss, walk=walk)
        #s_global_equity, s_global_return, s_global_mdd, s_global_romad, s_general_info, s_bh_results = self.calculate_selection_results(second_metric=second_metric, operation="short", metric=metric, top_number=top_number, penalty=penalty, stop_loss=stop_loss, walk=walk)

        ls_global_equity, ls_global_return, ls_global_mdd, ls_global_romad, ls_general_info, ls_bh_results = self.calculate_ensemble_selection_results(second_metric=second_metric, operation="long_short", metric=metric, top_number=top_number, penalty=penalty, stop_loss=stop_loss, walk=walk)
        l_global_equity, l_global_return, l_global_mdd, l_global_romad, l_general_info, l_bh_results = self.calculate_ensemble_selection_results(second_metric=second_metric, operation="long", metric=metric, top_number=top_number, penalty=penalty, stop_loss=stop_loss, walk=walk)
        s_global_equity, s_global_return, s_global_mdd, s_global_romad, s_general_info, s_bh_results = self.calculate_ensemble_selection_results(second_metric=second_metric, operation="short", metric=metric, top_number=top_number, penalty=penalty, stop_loss=stop_loss, walk=walk)

        
        ls_results = {
            "return": ls_global_return,
            "mdd": ls_global_mdd,
            "romad": ls_global_romad,
            "equity_line": ls_global_equity
        }

        l_results = {
            "return": l_global_return,
            "mdd": l_global_mdd,
            "romad": l_global_romad,
            "equity_line": l_global_equity
        }

        s_results = {
            "return": s_global_return,
            "mdd": s_global_mdd,
            "romad": s_global_romad,
            "equity_line": s_global_equity
        }


        return ls_results, l_results, s_results, ls_bh_results, ls_general_info

    '''
    ' @sebastian
    '''
    def calculate_epoch_selection(self):

        print("*** EPOCH SELECTION: sto generando csv con migliori epoche...")

        headers = ['net', 'epoch', 'valid_por', 'valid_romad', 'valid_return', 'valid_mdd', 'valid_cove', 'test_por', 'test_romad', 'test_return', 'test_mdd', 'test_cove']

        for index_walk in range(0, len(self.walks_list)):

            output_path = self.experiment_base_path + 'selection/on_finish/walk_' + str(index_walk) + '/'
            create_folder(output_path)

            df_nets_valid = pd.DataFrame()
            df_nets_test = pd.DataFrame()

            df_long_short = pd.DataFrame(columns=headers)
            df_long = pd.DataFrame(columns=headers)
            df_short = pd.DataFrame(columns=headers)
            
            startt = time.time()
            print('Walk ' + str(index_walk) + ' iniziato:')

            for index_net in range(0, len(self.nets_list)):

                
                df_cm_valid = pd.read_json(self.experiment_original_path + self.iperparameters['experiment_name'] + '/calculated_metrics/validation/walk_' + str(index_walk) + '/net_' + str(index_net) + '.json')
                df_cm_valid['net'] = index_net

                epoch_list = range(1, df_cm_valid.shape[0] + 1)

                df_cm_valid['epoch'] = epoch_list

                df_cm_test = pd.read_json(self.experiment_original_path + self.iperparameters['experiment_name'] + '/calculated_metrics/test/walk_' + str(index_walk) + '/net_' + str(index_net) + '.json')
                df_cm_test['net'] = index_net
                df_cm_test['epoch'] = epoch_list
                
                
                df_nets_valid = pd.concat([df_nets_valid, df_cm_valid], axis=0)
                df_nets_test = pd.concat([df_nets_test, df_cm_test], axis=0)

            df_long_short = self.get_all_epochs(valid=df_nets_valid.copy(), test=df_nets_test.copy(), rule_type='long_short')
            df_long = self.get_all_epochs(valid=df_nets_valid.copy(), test=df_nets_test.copy(), rule_type='long')
            df_short = self.get_all_epochs(valid=df_nets_valid.copy(), test=df_nets_test.copy(), rule_type='short')

            endt = time.time()
            print('Walk ' + str(index_walk) + ' concluso in:', endt-startt)

            df_long_short.to_csv(output_path + 'long_short_UNselected.csv', header=True, index=False)
            df_long.to_csv(output_path + 'long_UNselected.csv', header=True, index=False)
            df_short.to_csv(output_path + 'short_UNselected.csv', header=True, index=False)

    '''
    ' old @seba
    ' new @andrea
    '''
    def get_all_epochs(self, valid, test, rule_type):
        df = pd.DataFrame()

        if rule_type == 'long':
            df['valid_por'] = valid['longs_poc'].tolist()
            df['valid_romad'] = valid['lh_romads'].tolist()
            df['valid_return'] = valid['lh_returns'].tolist()
            df['valid_mdd'] = valid['lh_mdds'].tolist()
            df['valid_cove'] = valid['long_operations'].tolist()
            df['test_por'] = test['longs_poc'].tolist()
            df['test_romad'] = test['lh_romads'].tolist()
            df['test_mdd'] = test['lh_mdds'].tolist()
            df['test_return'] = test['lh_returns'].tolist()
            df['test_cove'] = test['long_operations'].tolist()
            df['net'] = valid['net'].tolist()
            df['epoch'] = valid['epoch'].tolist()

        if rule_type == 'short':
            df['valid_por'] = valid['shorts_poc'].tolist()
            df['valid_romad'] = valid['sh_romads'].tolist()
            df['valid_return'] = valid['sh_returns'].tolist()
            df['valid_mdd'] = valid['sh_mdds'].tolist()
            df['valid_cove'] = valid['short_operations'].tolist()
            df['test_por'] = test['shorts_poc'].tolist()
            df['test_romad'] = test['sh_romads'].tolist()
            df['test_mdd'] = test['sh_mdds'].tolist()
            df['test_return'] = test['sh_returns'].tolist()
            df['test_cove'] = test['short_operations'].tolist()
            df['net'] = valid['net'].tolist()
            df['epoch'] = valid['epoch'].tolist()

        if rule_type == 'long_short':
            df['valid_por'] = np.average([valid['longs_poc'].tolist(), valid['shorts_poc'].tolist()], axis=0)
            df['valid_romad'] = valid['ls_romads'].tolist()
            df['valid_return'] = valid['ls_returns'].tolist()
            df['valid_mdd'] = valid['ls_mdds'].tolist()
            df['valid_cove'] = np.add(valid['long_operations'].tolist(), valid['short_operations'].tolist())

            df['test_por'] = np.average([test['longs_poc'].tolist(), test['shorts_poc'].tolist()], axis=0)
            df['test_romad'] = test['ls_romads'].tolist()
            df['test_mdd'] = test['ls_mdds'].tolist()
            df['test_return'] = test['ls_returns'].tolist()
            df['test_cove'] = np.add(test['long_operations'].tolist(), test['short_operations'].tolist())
            df['net'] = valid['net'].tolist()
            df['epoch'] = valid['epoch'].tolist()

        return df

    '''
    ' Calcola la correlazione tra due colonne 
    ' del dataframe
    '''
    def calculate_correlation(self, operation):
        walk = 0

        # legge i csv di seba e seleziona i top_number (tipo i primi 10)
        df = pd.read_csv(self.selection_folder + 'on_finish/' + operation + '_UNselected.csv', delimiter=';', decimal=',')
        
        #df = df.loc[df["valid_cove"] >= 0.7]
        #df = df.loc[df["valid_cove"] <= 0.96]

        #df = df.sort_values(by=["valid_cove"], ascending=False)
        
        print(df.shape)
    
        print("valid_romad / test_romad:", '{:.2f}'.format(df['valid_romad'].corr(df['test_romad'])))

        print("valid_por / test_por:", '{:.2f}'.format(df['valid_por'].corr(df['test_por'])))

        print("valid_return / test_return:", '{:.2f}'.format(df['valid_return'].corr(df['test_return'])))

        print("valid_mdd / test_mdd:", '{:.2f}'.format(df['valid_mdd'].corr(df['test_mdd'])))

        print("valid_cove / test_cove:", '{:.2f}'.format(df['valid_cove'].corr(df['test_cove'])))

        print("valid_cove / test_romad:", '{:.2f}'.format(df['valid_cove'].corr(df['test_romad'])))

        print("valid_cove / test_por:", '{:.2f}'.format(df['valid_cove'].corr(df['test_por'])))

        print("valid_cove / test_return:", '{:.2f}'.format(df['valid_cove'].corr(df['test_return'])))
        
        print("valid_cove / test_mdd:", '{:.2f}'.format(df['valid_cove'].corr(df['test_mdd'])))

        print("valid_por / test_romad:", '{:.2f}'.format(df['valid_por'].corr(df['test_romad'])))

        print("test_por / test_romad:", '{:.2f}'.format(df['test_por'].corr(df['test_romad'])))

        print("valid_return / test_romad:", '{:.2f}'.format(df['valid_return'].corr(df['test_romad'])))
        print("\n")

        min_v = 0.6
        max_v = 0.9
        #for i in range(1, 10):
        df_2 = pd.DataFrame()
        df_2 = df.loc[(df["valid_por"] >= min_v) & (df["valid_por"] <= max_v)]

        print("Shape:", df_2.shape, "Min:", '{:.2f}'.format(min_v), "Max:", '{:.2f}'.format(max_v))
    
        print("valid_romad / test_romad:", '{:.2f}'.format(df_2['valid_romad'].corr(df_2['test_romad'])))

        print("valid_por / test_por:", '{:.2f}'.format(df_2['valid_por'].corr(df_2['test_por'])))

        print("valid_return / test_return:", '{:.2f}'.format(df_2['valid_return'].corr(df_2['test_return'])))

        print("valid_mdd / test_mdd:", '{:.2f}'.format(df_2['valid_mdd'].corr(df_2['test_mdd'])))

        print("valid_cove / test_cove:", '{:.2f}'.format(df_2['valid_cove'].corr(df_2['test_cove'])))

        print("valid_cove / test_romad:", '{:.2f}'.format(df_2['valid_cove'].corr(df_2['test_romad'])))

        print("valid_cove / test_por:", '{:.2f}'.format(df_2['valid_cove'].corr(df_2['test_por'])))

        print("valid_cove / test_return:", '{:.2f}'.format(df_2['valid_cove'].corr(df_2['test_return'])))
        
        print("valid_cove / test_mdd:", '{:.2f}'.format(df_2['valid_cove'].corr(df_2['test_mdd'])))

        print("valid_por / test_romad:", '{:.2f}'.format(df_2['valid_por'].corr(df_2['test_romad'])))

        print("test_por / test_romad:", '{:.2f}'.format(df_2['test_por'].corr(df_2['test_romad'])))

        print("valid_return / test_romad:", '{:.2f}'.format(df_2['valid_return'].corr(df_2['test_romad'])))
        
        print("\n")

        #min_v = min_v + 0.1
        #max_v = max_v + 0.1

        '''
        thrs = [0.96, 0.97]

        print("Media Return:", '{:.2f}'.format(df['test_return'].mean()))
        print("Media MDD:", '{:.2f}'.format(df['test_mdd'].mean()))
        print("Media Romad:", '{:.2f}'.format(df['test_romad'].mean()))
        print("\n")

        for thr in thrs: 
            df_min = df.loc[df['valid_cove'] > thr]
            
            print("Media Return valid_cove > " + str(thr) + ":", '{:.2f}'.format(df_min['test_return'].mean()))
            print("Media MDD valid_cove > " + str(thr) + ":", '{:.2f}'.format(df_min['test_mdd'].mean()))
            print("Media Romad valid_cove > " + str(thr) + ":", '{:.2f}'.format(df_min['test_romad'].mean()))
            print("\n")

        '''


    '''
    ' @sebastian
    '''
    def get_best_epoch(self, path, rule_type, min_thr, step, net, valid, test):

        if rule_type == 'long': # TODO longshort not handled
            val_poc = valid['longs_poc']
            val_rom = valid['lh_romads']
            val_ret = valid['lh_returns']
            val_mdd = valid['lh_mdds']
            val_op = valid['long_operations']
            test_rom = test['lh_romads']
            test_mdd = test['lh_mdds']
            test_ret = test['lh_returns']
            test_op = test['long_operations']
        else:
            val_poc = valid['shorts_poc']
            val_rom = valid['sh_romads']
            val_ret = valid['sh_returns']
            val_mdd = valid['sh_mdds']
            val_op = valid['short_operations']
            test_rom = test['sh_romads']
            test_mdd = test['sh_mdds']
            test_ret = test['sh_returns']
            test_op = test['short_operations']

        min_group = 100
        lazy_group = 50
        strict_group = 20

        # criteri
        rule_1 = '0'
        rule_2 = '0'
        rule_3 = '0'
        rule_4 = '0'
        rule_5 = '0'
        rule_6 = '0'
        rule_7 = '0'
        rule_8 = '0'
        rule_9 = '0'

        # Get best candidate over PoR (escluding first 10 epochs)
        bc_index = np.argmax(val_poc[10:]) + 10
        bc_poc = val_poc[bc_index]

        if bc_poc < min_thr:
            return

        # Validation
        bc_romad = val_rom[bc_index]
        bc_mdd = val_mdd[bc_index]
        bc_return = val_ret[bc_index]
        bc_op = val_op[bc_index]

        # Test
        bc_tromad = test_rom[bc_index]
        bc_tmdd = test_mdd[bc_index]
        bc_treturn = test_ret[bc_index]
        bc_top = test_op[bc_index]

        filtered = (np.array(val_rom) > bc_romad) & (np.array(val_rom) > 0)
        num_filtered = filtered.sum()

        if num_filtered > min_group:
            return

        if bc_romad < 1:
            return

        if bc_return < 1000:
            return

        # rule compliance
        if (bc_poc >= min_thr + 2*step) and (num_filtered < strict_group):
            rule_1 = '1'
        
        if (bc_poc >= min_thr + 2*step) and (num_filtered < lazy_group):
            rule_2 = '1'
        
        if (bc_poc >= min_thr + 2*step) and (num_filtered < min_group):
            rule_3 = '1'

        if (bc_poc >= min_thr + step) and (num_filtered < strict_group):
            rule_4 = '1'

        if (bc_poc >= min_thr + step) and (num_filtered < lazy_group):
            rule_5 = '1'

        if (bc_poc >= min_thr + step) and (num_filtered < min_group):
            rule_6 = '1'

        if (bc_poc >= min_thr) and (num_filtered < strict_group):
            rule_7 = '1'

        if (bc_poc >= min_thr) and (num_filtered < lazy_group):
            rule_8 = '1'

        if (bc_poc >= min_thr) and (num_filtered < min_group):
            rule_9 = '1'
        

        # string formatting
        str_net = str(net)
        str_index = str(bc_index + 1)
        str_por = str(bc_poc).replace('.', ',')
        str_valrom = '{:.2f}'.format(bc_romad).replace('.', ',')
        str_valret = str(bc_return).replace('.', ',')
        str_valmdd = str(bc_mdd).replace('.', ',')
        str_valcov = '{:.2f}'.format(bc_op).replace('.', ',')
        str_testrom = '{:.2f}'.format(bc_tromad).replace('.', ',')
        str_testret = str(bc_treturn).replace('.', ',')
        str_testmdd = str(bc_tmdd).replace('.', ',')
        str_testcov = '{:.2f}'.format(bc_top).replace('.', ',')

        f=open(path + rule_type + "_selected.csv", "a+")
        f.write(str_net + ';' + str_index + ';'+ str_por + ';' + str_valrom + ';' + str_valret + ';' + str_valmdd + ';' + str_valcov + ';' + str_testrom  + ';' + str_testret + ';' + str_testmdd + ';' + str_testcov + ';' + rule_1 + ';' + rule_2 + ';' + rule_3 + ';' + rule_4 + ';' + rule_5 + ';' + rule_6 + ';' + rule_7 + ';' + rule_8 + ';' + rule_9 + '\n')

        return
    



