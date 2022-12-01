import os
import numpy as np
import pandas as pd
from datetime import timedelta
from os import listdir
from os.path import isfile, join
from classes.Market import Market
from classes.Measures import Measures
from classes.Utils import natural_keys, df_date_merger
from datetime import timedelta
import statistics
import time
import threading
import xlsxwriter
import matplotlib
import matplotlib.pyplot as plt
import statistics
import json 

'''
'
'''


class ResultsHandler:

    experiment_name = ''

    experiment_original_path = 'C:/Users/Utente/Documents/GitHub/PhD-Market-Nets/experiments/Esperimenti Vacanze/'
    experiment_original_path = 'C:/Users/Utente/Documents/GitHub/PhD-Market-Nets/experiments/'  # locale
    # experiment_original_path = '/media/unica/HDD 9TB Raid0 - 1/experiments/' # server

    original_predictions_validation_folder = ''
    original_predictions_test_folder = ''
    ensemble_base_folder = ''
    final_decision_folder = ''

    dataset = ''

    walks_list = []
    nets_list = []

    def __init__(self, experiment_name):
        self.experiment_name = experiment_name

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
    def ensemble_magg(self, df, thr=0.5):
        n_short = (df.eq(0).sum(1) / df.shape[1]).gt(thr)
        n_long = (df.eq(2).sum(1) / df.shape[1]).gt(thr)
        n_hold = (df.eq(1).sum(1) / df.shape[1]).gt(thr)

        # n_long.astype(int).mul(2).add(n_short)

        # DECOMMENTARE QUESTA RIGA PER ESCLUDERE LE HOLD DAL CALCOLO
        #m = pd.DataFrame(np.select([n_short, n_long], [0, 2], 1), index=df.index, columns=['ensemble'])
        m = pd.DataFrame(np.select([n_short, n_hold, n_long], [0, 1, 2], 1), index=df.index, columns=['ensemble'])

        return m

    '''
    ' Calcolo l'ensemble ad eliminazione
    ' Se il numero è positivo sarà una Long, altrimenti sarà una short
    ' Es: 10 long, 2 short = 8
    ' Es: 10 short, 3 long = -7
    ' Es: 10 long, 10 short = 0, hold
    '''
    def elimination_ensemble(self, df):
        df = df.drop(['date_time'], axis=1)
        return pd.DataFrame(df.eq(2).sum(1) + (-df.eq(0).sum(1)), index=df.index, columns=['ensemble'])

    '''
    '
    '''
    def elimination_ensemble_exclusive(self, df):
        df = df.drop(['date_time'], axis=1)
        return pd.DataFrame(df.eq(2).sum(1), index=df.index, columns=['ensemble'])

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
    '
    ''
    def calculate_ensemble_NUOVOOO(self, type):
        # A seconda del set creo input e output path
        if type == 'Validation':
            input_path = self.original_predictions_validation_folder
            output_path = self.ensemble_base_folder + 'validation/'

        if type == 'Test':
            input_path = self.original_predictions_test_folder
            output_path = self.ensemble_base_folder + 'test/'

        if not os.path.isdir(output_path + 'triple_csv/'):
            os.makedirs(output_path + 'triple_csv/')

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

                ensemble_magg = self.generate_triple(df=df_net)

                df_ensemble_magg[epoch] = ensemble_magg['ensemble']
                
                end = time.time()
                print(self.experiment_name + " | Ensemble magg | " + type + " | " + walk + " | epoca: " + str(idx + 1) + " | ETA: " + "{:.3f}".format(end-start))

            df_ensemble_magg.to_csv(output_path + 'triple_csv/' + walk + '.csv', header=True, index=False)

            end_walk_time = time.time()
            print(self.experiment_name + " | Ensemble magg | " + type + " | Total walk elapsed time:" + "{:.3f}".format(end_walk_time - start_walk_time))

    ''' #fine ensemble nuovo

    '''
    '
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

                    ensemble_magg = self.ensemble_magg(df=df_net, thr=thr)

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

        if not os.path.isdir(output_path + 'ensemble_el_exclusive/'):
            os.makedirs(output_path + 'ensemble_el_exclusive/')

        # per ogni walk
        for walk in self.walks_list:
            start_walk_time = time.time()
            date_list, epochs_list = self.get_date_epochs_walk(path=input_path, walk=walk)

            # Qui salvo il risultato dell'ensemble di ogni rete per ogni epoca
            df_ensemble_el = pd.DataFrame(columns=[epochs_list])
            df_ensemble_el['date_time'] = date_list

            df_ensemble_el_exclusive = pd.DataFrame(columns=[epochs_list])
            df_ensemble_el_exclusive['date_time'] = date_list

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
                ensemble_el_exclusive = self.elimination_ensemble_exclusive(df=df_net)

                hold_thr = 0.2
                ensemble_offset = self.offset_thr_ensemble(df=df_net, hold_thr=hold_thr)

                df_ensemble_el[epoch] = ensemble_el['ensemble']
                df_ensemble_offset[epoch] = ensemble_offset['ensemble']
                df_ensemble_el_exclusive[epoch] = ensemble_el_exclusive['ensemble']

                end = time.time()
                print(self.experiment_name + " | Ensembles: " + type + " | " + walk + " | epoca: " + str(idx + 1) + " | ETA: " + "{:.3f}".format(end-start))

            df_ensemble_el.to_csv(output_path + 'ensemble_el/' + walk + '.csv', header=True, index=False)
            df_ensemble_offset.to_csv(output_path + 'ensemble_offset/' + walk + '.csv', header=True, index=False)
            df_ensemble_el_exclusive.to_csv(output_path + 'ensemble_el_exclusive/' + walk + '.csv', header=True, index=False)

            end_walk_time = time.time()
            print(self.experiment_name + " | Ensembles: " + type + "| Total walk elapsed time:" + "{:.3f}".format(end_walk_time - start_walk_time))

    '''
    '
    '''
    def run_ensemble(self, thrs, remove_nets=False):
        
        t1 = threading.Thread(target=self.calculate_ensemble_magg, args=([thrs, 'Validation', remove_nets]))
        t2 = threading.Thread(target=self.calculate_ensemble_magg, args=([thrs, 'Test', remove_nets]))
        
        #t3 = threading.Thread(target=self.calculate_ensembles, args=(['Validation', remove_nets]))
        #t4 = threading.Thread(target=self.calculate_ensembles, args=(['Test', remove_nets]))
        
        t1.start()
        t2.start()
        #t3.start()
        #t4.start()

        t1.join()
        t2.join()
        #t3.join()
        #t4.join()


    def test_ensemble_magg(self, thr_ensemble_magg=[], thr_exclusive=[], thr_elimination=[]): 
        df_all = pd.DataFrame()

        for walk in self.walks_list:
            df_walk = pd.read_csv(self.ensemble_base_folder + 'validation/triple_csv/' + walk + '.csv')
            df_all = pd.concat([df_all, df_walk])
            
        print(df_all)       
    '''
    █▀▀█ █▀▀ ▀▀█▀▀ █░░█ █▀▀█ █▀▀▄ █▀▀   ░ ░   █▀▄▀█ █▀▀▄ █▀▀▄   ░ ░   █▀▀█ █▀▀█ █▀▄▀█ █▀▀█ █▀▀▄
    █▄▄▀ █▀▀ ░░█░░ █░░█ █▄▄▀ █░░█ ▀▀█   ▀ ▀   █░▀░█ █░░█ █░░█   ▀ ▀   █▄▄▀ █░░█ █░▀░█ █▄▄█ █░░█
    ▀░▀▀ ▀▀▀ ░░▀░░ ░▀▀▀ ▀░▀▀ ▀░░▀ ▀▀▀   ░ ░   ▀░░░▀ ▀▀▀░ ▀▀▀░   ░ ░   ▀░▀▀ ▀▀▀▀ ▀░░░▀ ▀░░▀ ▀▀▀░
    '''

    '''
    ' Genero i file di decisioni finali per i vari ensemble
    ' type = [ensemble_magg, ensemble_el_longonly, ensemble_el_exclusive]
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
    ' ensemble_el_exclusive
    ' get_final_decision_from_ensemble(type='ensemble_el_exclusive', num_agreement=10, remove_nets=False, validation_thr=15)
    '
    ' Calcolo le decisioni finali sull'ensemble el, facendo il long only usando la % 
    ' Viene calcolato sull'ensemble generato da elimination_ensemble_longonly 
    '(quindi facendo +1 long e basta)
    ' # TOLTO num_agreement
    '''
    def get_final_decision_from_ensemble(self, type='ensemble_magg', validation_thr=15, validation_metric='romad', ensemble_thr=0.35, perc_agreement=0.3, remove_nets=False):
        print("Generating final decision for: " + type + "...")

        remove_nets_str = 'con-rimozione-reti/'
        if remove_nets == False:
            remove_nets_str = 'senza-rimozione-reti/'

        if type is 'ensemble_magg':
            output_path = self.final_decision_folder + 'ensemble_magg/' + remove_nets_str
            validation_input_path = self.ensemble_base_folder +  remove_nets_str + 'validation/ensemble_magg/' + str(ensemble_thr) + '/'
            test_input_path = self.ensemble_base_folder + remove_nets_str + 'test/ensemble_magg/' + str(ensemble_thr) + '/'


        if type is 'ensemble_el_long_only':
            output_path = self.final_decision_folder + 'ensemble_el_longonly/' + remove_nets_str
            validation_input_path = self.ensemble_base_folder  + remove_nets_str + 'validation/ensemble_el/'
            test_input_path = self.ensemble_base_folder + remove_nets_str + 'test/ensemble_el/'

        if type is 'ensemble_el_exclusive': 
            output_path = self.final_decision_folder + 'ensemble_el_exclusive/' + remove_nets_str
            validation_input_path = self.ensemble_base_folder + remove_nets_str + 'validation/ensemble_el_exclusive/'
            test_input_path = self.ensemble_base_folder + remove_nets_str + 'test/ensemble_el_exclusive/'
        
        # creo la path finale     
        if not os.path.isdir(output_path):
            os.makedirs(output_path)

        df_global = pd.DataFrame(columns=['date_time', 'close', 'delta_current_day', 'delta_next_day', 'label'])


        dataset = Market(dataset=self.dataset)
        dataset_label = dataset.get_label(freq='1d', columns=['open', 'close', 'delta_current_day', 'delta_next_day'])
        dataset_label = dataset_label.reset_index()
        dataset_label['date_time'] = dataset_label['date_time'].astype(str)

        for index_walk, walk in enumerate(self.walks_list):
            df_ensemble_val = pd.read_csv(validation_input_path + walk + '.csv')
            df_merge_with_label = pd.merge(df_ensemble_val, dataset_label, how="inner")

            val_idx, romad, return_value, mdd_value = self.get_max_idx_from_validation(df_merge_with_label=df_merge_with_label, validation_thr=validation_thr, metric='romad')
            #print("Walk n° " +  str(index_walk) + " - ID validation scelto: " + str(val_idx))
            #input()

            df_test = pd.read_csv(test_input_path + walk + '.csv')

            # mergio con le label, così ho un subset del df con le date che mi servono e la predizione
            df_test = pd.merge(df_test, dataset_label, how="inner")
            #df_merge_with_label = df_merge_with_label.set_index('index')
            df_test = df_test[['epoch_' + str(val_idx), 'open', 'close', 'delta_current_day', 'delta_next_day', 'date_time']]
            df_test = df_test.rename(columns={"epoch_" + str(val_idx): "decision"})

            df_global = pd.concat([df_global, df_test], sort=True)

        df_global = df_global.drop_duplicates(subset='date_time', keep="first")

        df_global['date_time'] = df_global['date_time'].shift(-1)
        #df_global = df_global.drop(columns=['close', 'delta'], axis=1)
        df_global = df_global[['date_time', 'decision']]
        df_global = df_global.drop(df_global.index[0])

        df_global = df_date_merger(df=df_global, columns=['open', 'close', 'delta_current_day', 'delta_next_day'], dataset='sp500_cet')
        df_global['decision'] = df_global['decision'].astype(int)

        #print("CONTO TOTALE DELLE RETIIIIIII")
        #print(len(self.nets_list))
        #input()

        number_of_nets = len(self.nets_list)

        if type is 'ensemble_el_longonly':
            # CALCOLO DI QUANTO DEV'ESSERE IL VALORE DI LABEL PER ESSERE IN LINEA CON LA % DI AGREEMENT
            jolly_number = int((number_of_nets * (perc_agreement * 100) / 100))
            
            df_global['decision'] = df_global['decision'].apply(lambda x: 2 if x > jolly_number else 1)
            df_global.to_csv(output_path + 'decisions_ensemble_el_long_only_' + str(perc_agreement) + '.csv', header=True, index=False)

        if type is 'ensemble_el_exclusive': 
            # CALCOLO DI QUANTO DEV'ESSERE IL VALORE DI LABEL PER ESSERE IN LINEA CON LA % DI AGREEMENT
            jolly_number = int((number_of_nets * (perc_agreement * 100) / 100))

            #df_global['decision'] = df_global['decision'].apply(lambda x: 2 if x > num_agreement else 1)
            df_global['decision'] = df_global['decision'].apply(lambda x: 2 if x > jolly_number else 1)
            df_global.to_csv(output_path + 'decisions_ensemble_el_exclusive_' + str(perc_agreement) + '.csv', header=True, index=False)

        if type is 'ensemble_magg': 
            df_global.to_csv(output_path + 'decisions_ensemble_' + str(ensemble_thr) + '.csv', header=True, index=False)


    '''
    '
    '''
    def get_final_decision_without_ensemble(self, type='without_ensemble', validation_thr=15, validation_metric='romad'):
        print("Generating final decision for: " + type + "...")

        # dataset wrapper
        dataset = Market(dataset=self.dataset)
        dataset_label = dataset.get_label(freq='1d', columns=['open', 'close', 'delta_current_day', 'delta_next_day'])
        dataset_label = dataset_label.reset_index()
        dataset_label['date_time'] = dataset_label['date_time'].astype(str)

        df_global = pd.DataFrame(columns=['date_time', 'close', 'delta_current_day', 'delta_next_day', 'label'])

        output_path = self.final_decision_folder + 'without_ensemble/'

        if not os.path.isdir(output_path):
            os.makedirs(output_path)

        for walk in self.walks_list:
            reti = []

            choose_net, choose_val_idx, choose_romad, choose_return, choose_mdd = -1000, -1000, -1000, -100000, 100000

            for net in self.nets_list:
                df_net = pd.read_csv(self.original_predictions_validation_folder + walk + '/' + net)
                df_merge_with_label = pd.merge(df_net, dataset_label, how="inner")

                if type is 'without_ensemble':
                    val_idx, romad, global_return, mdd = self.get_max_idx_from_validation(df_merge_with_label=df_merge_with_label, validation_thr=validation_thr, metric='romad')
                if type is 'without_ensemble_last_epoch':
                    y_pred = df_merge_with_label['epoch_' + str(self.iperparameters['epochs'])].to_list()
                    delta = df_merge_with_label['delta_next_day'].to_list()

                    equity_line, global_return, mdd, romad, i, j = Measures.get_equity_return_mdd_romad(y_pred, delta, self.iperparameters['return_multiplier'], type='long_short')

                if  (validation_metric == 'romad' and romad > choose_romad and romad != 10.0) \
                or (validation_metric == 'return' and global_return > choose_return) \
                or (validation_metric == 'mdd' and mdd < choose_mdd and mdd != 0.001):
                    choose_val_idx = val_idx
                    choose_romad = romad
                    choose_net = net
                    choose_return = global_return
                    choose_mdd = mdd
                
            # for net in nets_list

            print("Walk: ", walk, " - Rete finale scelta: ", choose_net, " - Indice epoca:", choose_val_idx, " - Valore romad: ", choose_romad)
            # input()
            test_net = pd.read_csv(self.original_predictions_test_folder + walk + '/' + choose_net)
            # print(test_net)
            # input()
            df_test = pd.merge(test_net, dataset_label, how="inner")

            # CODICE DI PROVA PER PRECISION -> CAMBIATO IL SUBSET DELLE COLONNE, AGGIUNTO LABEL CHE CONTERRA' LA LABEL ORIGINALE
            df_test = df_test[['epoch_' + str(choose_val_idx), 'open', 'close', 'delta_current_day', 'delta_next_day', 'date_time']]
            
            if type is 'without_ensemble':
                df_test = df_test.rename(columns={"epoch_" + str(choose_val_idx): "decision"})

            if type is 'without_ensemble_last_epoch':
                df_test = df_test.rename(columns={'epoch_' + str(self.iperparameters['epochs']): "decision"})

            df_global = pd.concat([df_global, df_test], sort=True)

            df_global = df_global.drop_duplicates(subset='date_time', keep="first")

        # AGGIUNGERE UN GIORNO AL DATETIME INVECE CHE FARE LO SHIFT
        df_global['date_time'] = pd.to_datetime(df_global['date_time']) + timedelta(days=1)
        df_global = df_global[['date_time', 'decision']]
        df_global = df_global.drop(df_global.index[0])
        df_global['decision'] = df_global['decision'].astype(int)

        if type is 'without_ensemble':
            df_global.to_csv(output_path + 'decisions_without_ensemble.csv', header=True, index=False)

        if type is 'without_ensemble_last_epoch':
            df_global.to_csv(output_path + 'decision_with_last_epoch.csv', header=True, index=False)


    '''
    ' Scelgo l'epoca migliore dal validation utilizzando come metrica o il return, 
    ' il romad oppure l'mdd
    ' return: epoch_index, romad_self.iperparameters['epochs'][epoch_index], return_self.iperparameters['epochs'][epoch_index], mdd_self.iperparameters['epochs'][epoch_index]
    '''
    def get_max_idx_from_validation(self, df_merge_with_label, validation_thr, metric='return'):
        # mergio con le label, così ho un subset del df con le date che mi servono e la predizione
        #df_merge_with_label = pd.merge(net, dataset_label, how="inner")

        # conto le epoche - 1 (c'è il date_time e l'id)
        number_of_epochs = self.iperparameters['epochs']

        romad_epochs = np.zeros(number_of_epochs + 1)
        return_epochs = np.zeros(number_of_epochs + 1)
        mdd_epochs = np.zeros(number_of_epochs + 1)

        # calcolo il return per un epoca, lascio volontariamente il primo elemento a 0
        for i in range(1, number_of_epochs + 1):
            y_pred = df_merge_with_label['epoch_' + str(i)].to_list()
            delta = df_merge_with_label['delta_next_day'].to_list()

            equity_line, global_return, mdd, romad, ii, j = Measures.get_equity_return_mdd_romad(y_pred=y_pred, delta=delta, multiplier=self.iperparameters['return_multiplier'], type='long_short')
            #print("Epoca N° ", i, " Romad: ", romad)
            #print("Epoca N°", i, " - Return: ", global_return, " - Romad: ", romad)
            romad_epochs[i] = romad
            return_epochs[i] = global_return
            mdd_epochs[i] = mdd

        '''
        plt.figure(figsize=(30,18))
        plt.style.use("ggplot")
        plt.plot(range(1, self.iperparameters['epochs'] + 1), return_epochs)
        plt.savefig("C:/Users/Utente/Documents/GitHub/PhD-Market-Nets/experiments/BBB - Debug Romad e Return diversi - 5 reti/accuracy_loss_plots/walk_0/AA-plot-return.png")
        plt.close()
        plt.figure(figsize=(30,18))
        plt.style.use("ggplot")
        plt.plot(range(1, self.iperparameters['epochs'] + 1), mdd_epochs)
        plt.savefig("C:/Users/Utente/Documents/GitHub/PhD-Market-Nets/experiments/BBB - Debug Romad e Return diversi - 5 reti/accuracy_loss_plots/walk_0/AA-plot-mdd.png")
        plt.close()
        plt.figure(figsize=(30,18))
        plt.style.use("ggplot")
        plt.plot(range(1, self.iperparameters['epochs'] + 1), romad_epochs)
        plt.savefig("C:/Users/Utente/Documents/GitHub/PhD-Market-Nets/experiments/BBB - Debug Romad e Return diversi - 5 reti/accuracy_loss_plots/walk_0/AA-plot-romad.png")
        plt.close()
        '''

        '''
        print(romad_epochs)
        print(np.argmax(romad_epochs))
        print(romad_epochs[np.argmax(romad_epochs)])
        input()
        '''

        # Ritorno medio con l'intorno
        mean_values = np.zeros(number_of_epochs)
        array_values = []  # la uso per salvare la metrica scelta

        if metric is 'return':
            array_values = return_epochs
        if metric is 'romad':
            array_values = romad_epochs
        if metric is 'mdd':
            array_values = mdd_epochs

        for index_for, value in enumerate(array_values):
            # entro solo tra i valori compresi dalla soglia, se soglia = 15, parto dal 15° sino a i-15
            if index_for >= validation_thr and index_for <= (number_of_epochs - validation_thr):
                mean_values[index_for] = statistics.mean(array_values[index_for-validation_thr:index_for+validation_thr])
                #print("mean_values id", index_for, "valore: ", mean_values[index_for])
        # seleziono l'epoca con il return migliore a cui aggiungo l'intorno per selezionarlo int est
        epoch_index = np.argmax(mean_values) + 1

        return epoch_index, romad_epochs[epoch_index], return_epochs[epoch_index], mdd_epochs[epoch_index]

    '''
    ' Leggo il file con le decisioni ultime
    ' Quindi calcolo tutte le metriche per quel CSV (unico per tutti gli walk)
    ' Utilizzo il delta_current_day poiché il vettore delle label è già 
    ' allineato al giorno corrente (giorno in cui deve venir fatta l'operazione)
    '''
    def get_results(self, ensemble_type='ensemble_magg', ensemble_thr=0.35, perc_agreement=0.3, remove_nets=False):
        remove_nets_str = 'con-rimozione-reti/'

        if remove_nets == False:
            remove_nets_str = 'senza-rimozione-reti/'

        #ensemble magg
        if ensemble_type is 'ensemble_magg':
            df = pd.read_csv(self.final_decision_folder + 'ensemble_magg/' + remove_nets_str + 'decisions_ensemble_' + str(ensemble_thr) + '.csv')
        if ensemble_type is 'ensemble_el_long_only':
            df = pd.read_csv(self.final_decision_folder + 'ensemble_el_longonly/' + remove_nets_str + 'decisions_ensemble_el_long_only_' + str(perc_agreement) + '.csv')

        if ensemble_type is 'ensemble_el_exclusive':
            df = pd.read_csv(self.final_decision_folder + 'ensemble_el_exclusive/' + remove_nets_str + 'decisions_ensemble_el_exclusive_' + str(perc_agreement) + '.csv')

        # without ensemble
        if ensemble_type is 'without_ensemble':
            df = pd.read_csv(self.final_decision_folder + 'without_ensemble/decisions_without_ensemble.csv')

        close = df['close'].tolist()
        y_pred = df['decision'].tolist()
        delta = df['delta_current_day'].tolist()
        dates = df['date_time'].tolist()

        # calcolo il b&h
        bh_equity_line, bh_global_return, bh_mdd, bh_romad, i, j = Measures.get_return_mdd_romad_bh(close=close, multiplier=self.iperparameters['return_multiplier'])

        # calcolo tutte le info per long + short, long + hold e short + hold
        long_info, short_info, hold_info, general_info = Measures.get_precision_count_coverage(y_pred=y_pred, delta=delta)
        ls_equity_line, ls_global_return, ls_mdd, ls_romad, ls_i, ls_j = Measures.get_equity_return_mdd_romad(y_pred=y_pred, delta=delta, multiplier=self.iperparameters['return_multiplier'], type='long_short')
        lh_equity_line, lh_global_return, lh_mdd, lh_romad, lh_i, lh_j = Measures.get_equity_return_mdd_romad(y_pred=y_pred, delta=delta, multiplier=self.iperparameters['return_multiplier'], type='long_only')
        sh_equity_line, sh_global_return, sh_mdd, sh_romad, sh_i, sh_j = Measures.get_equity_return_mdd_romad(y_pred=y_pred, delta=delta, multiplier=self.iperparameters['return_multiplier'], type='short_only')

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
    ''
    def get_report_excel(self, report_name, ensemble_thr=0.35, remove_nets=False, type='ensemble_magg'):
        remove_nets_str = 'con-rimozione-reti/'

        if remove_nets == False:
            remove_nets_str = 'senza-rimozione-reti/'

        #ensemble magg
        if ensemble_type is 'ensemble_magg':
            df = pd.read_csv(self.final_decision_folder + 'ensemble_magg/' + remove_nets_str + 'decisions_ensemble_' + str(ensemble_thr) + '.csv')
        if ensemble_type is 'ensemble_el_long_only':
            df = pd.read_csv(self.final_decision_folder + 'ensemble_el_longonly/' + remove_nets_str + 'decisions_ensemble_el_long_only.csv')

        if ensemble_type is 'ensemble_el_exclusive':
            df = pd.read_csv(self.final_decision_folder + 'ensemble_el_exclusive/' + remove_nets_str + 'decisions_ensemble_el_exclusive.csv')

        # without ensemble
        if ensemble_type is 'without_ensemble':
            df = pd.read_csv(self.final_decision_folder + 'without_ensemble/decisions_without_ensemble.csv')

        close = df['close'].tolist()
        y_pred = df['decision'].tolist()
        delta = df['delta_current_day'].tolist()
        dates = df['date_time'].tolist()

        # calcolo il b&h
        bh_equity_line, bh_global_return, bh_mdd, bh_romad, i, j = Measures.get_return_mdd_romad_bh(close=close, multiplier=self.iperparameters['return_multiplier'])

        # calcolo tutte le info per long + short, long + hold e short + hold
        long_info, short_info, hold_info, general_info = Measures.get_precision_count_coverage(y_pred=y_pred, delta=delta)
        ls_equity_line, ls_global_return, ls_mdd, ls_romad, ls_i, ls_j = Measures.get_equity_return_mdd_romad(y_pred=y_pred, delta=delta, multiplier=self.iperparameters['return_multiplier'], type='long_short')
        lh_equity_line, lh_global_return, lh_mdd, lh_romad, lh_i, lh_j = Measures.get_equity_return_mdd_romad(y_pred=y_pred, delta=delta, multiplier=self.iperparameters['return_multiplier'], type='long_only')
        sh_equity_line, sh_global_return, sh_mdd, sh_romad, sh_i, sh_j = Measures.get_equity_return_mdd_romad(y_pred=y_pred, delta=delta, multiplier=self.iperparameters['return_multiplier'], type='short_only')

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

    '''
    ' Leggo tutte le metriche in output da un tipo di ensemble con get_results
    ' e salvo un file excel con tute le informazioni che mi servono
    '''
    def get_report_excel(self, report_name, ensemble_thr=0.35, perc_agreement=0.3, remove_nets=False, type='ensemble_magg'):
        print("Generating excel for: " + type + "...")
        
        if type is 'ensemble_magg':
            ls_results, l_results, s_results, bh_results, general_info = self.get_results(ensemble_type=type, ensemble_thr=ensemble_thr, remove_nets=remove_nets)

        if type is 'ensemble_el_long_only':
            ls_results, l_results, s_results, bh_results, general_info = self.get_results(ensemble_type=type, remove_nets=remove_nets, perc_agreement=perc_agreement)

        if type is 'ensemble_el_exclusive':
            ls_results, l_results, s_results, bh_results, general_info = self.get_results(ensemble_type=type, remove_nets=remove_nets, perc_agreement=perc_agreement)

        if type is 'without_ensemble':
            ls_results, l_results, s_results, bh_results, general_info = self.get_results(ensemble_type=type, remove_nets=remove_nets)

        # Se non esiste la cartella, la creo
        report_path = self.experiment_original_path + self.experiment_name + '/reports/'
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
        chart.add_series({'name': 'Curva Equity',
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
        chart.add_series({'name': 'Curva Equity',
                          'categories': '=CurvaEquityLongOnly!$AA$1:$AA$' + str(len(general_info['dates'])),
                          'values': '=CurvaEquityLongOnly!$AB$1:$AB$' + str(len(l_results['equity_line']))
                          })
        chart.add_series({'name': 'Buy & Hold',
                          'categories': '=CurvaEquityLongOnly!$AA$1:$AA$' + str(len(general_info['dates'])),
                          'values': '=CurvaEquityLongOnly!$AC$1:$AC$' + str(len(bh_results['equity_line']))
                          })

        chart.set_title({'name': 'Curva Equity Long Only'})
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
        chart.add_series({'name': 'Curva Equity',
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
    █▀▀ █░░█ █▀▀ ▀▀█▀▀ █▀▀█ █▀▄▀█   █▀▄▀█ █▀▀ ▀▀█▀▀ █▀▀█ ░▀░ █▀▀ █▀▀
    █░░ █░░█ ▀▀█ ░░█░░ █░░█ █░▀░█   █░▀░█ █▀▀ ░░█░░ █▄▄▀ ▀█▀ █░░ ▀▀█
    ▀▀▀ ░▀▀▀ ▀▀▀ ░░▀░░ ▀▀▀▀ ▀░░░▀   ▀░░░▀ ▀▀▀ ░░▀░░ ▀░▀▀ ▀▀▀ ▀▀▀ ▀▀▀
    '''

    '''
    '
    '''
    def generate_custom_metrics(self, type='Validation'):

        for index_walk, walk in enumerate(self.walks_list):
            all_ls_returns = np.zeros(shape=(30, self.iperparameters['epochs']))
            all_lh_returns = np.zeros(shape=(30, self.iperparameters['epochs']))
            all_sh_returns = np.zeros(shape=(30, self.iperparameters['epochs']))

            all_ls_romads = np.zeros(shape=(30, self.iperparameters['epochs']))
            all_lh_romads = np.zeros(shape=(30, self.iperparameters['epochs']))
            all_sh_romads = np.zeros(shape=(30, self.iperparameters['epochs']))

            all_ls_mdds = np.zeros(shape=(30, self.iperparameters['epochs']))
            all_lh_mdds = np.zeros(shape=(30, self.iperparameters['epochs']))
            all_sh_mdds = np.zeros(shape=(30, self.iperparameters['epochs']))

            all_longs_precisions = np.zeros(shape=(30, self.iperparameters['epochs']))
            all_shorts_precisions = np.zeros(shape=(30, self.iperparameters['epochs']))
            
            all_longs_coverage = np.zeros(shape=(30, self.iperparameters['epochs']))
            all_shorts_coverage = np.zeros(shape=(30, self.iperparameters['epochs']))

            all_long_operations = np.zeros(shape=(30, self.iperparameters['epochs']))
            all_short_operations = np.zeros(shape=(30, self.iperparameters['epochs']))
            all_hold_operations = np.zeros(shape=(30, self.iperparameters['epochs']))

            all_longs_poc = np.zeros(shape=(30, self.iperparameters['epochs']))
            all_shorts_poc = np.zeros(shape=(30, self.iperparameters['epochs']))

            for index_net, current_net in enumerate(self.nets_list):
                net = 'net_' + str(index_net) + '.csv'
                # leggo le predizioni fatte con l'esnemble
                df = pd.read_csv(self.experiment_original_path + self.experiment_name + '/predictions/predictions_during_training/' + type + '/walk_' + str(index_walk) + '/' + net)

                # mergio con le label, così ho un subset del df con le date che mi servono e la predizione 
                df_merge_with_label = df_date_merger(df=df, columns=['date_time', 'delta_next_day', 'close', 'open'], dataset=self.dataset)

                ls_returns = []
                lh_returns = []
                sh_returns = []

                ls_romads = []
                lh_romads = []
                sh_romads = []

                ls_mdds = []
                lh_mdds = []
                sh_mdds = []

                longs_precisions = []
                shorts_precisions = []
                longs_coverage = []
                shorts_coverage = []
                longs_poc = []
                shorts_poc = []


                long_operations = []
                short_operations = []
                hold_operations = []

                label_coverage = Measures.get_delta_coverage(delta=df_merge_with_label['delta_next_day'].tolist())

                # calcolo il return per un epoca
                for epoch in range(1, self.iperparameters['epochs'] + 1): 
                    y_pred = df_merge_with_label['epoch_' + str(epoch)].tolist()
                    delta = df_merge_with_label['delta_next_day'].tolist()

                    ls_equity_line, ls_global_return, ls_mdd, ls_romad, ls_i, ls_j  = Measures.get_equity_return_mdd_romad(y_pred=y_pred, delta=delta, multiplier=MULTIPLIER, type='long_short')
                    lh_equity_line, lh_global_return, lh_mdd, lh_romad, lh_i, lh_j  = Measures.get_equity_return_mdd_romad(y_pred=y_pred, delta=delta, multiplier=MULTIPLIER, type='long_only')
                    sh_equity_line, sh_global_return, sh_mdd, sh_romad, sh_i, sh_j  = Measures.get_equity_return_mdd_romad(y_pred=y_pred, delta=delta, multiplier=MULTIPLIER, type='short_only')

                    long, short, hold, general = Measures.get_precision_count_coverage(y_pred=y_pred, delta=delta)
                    
                    long_poc, short_poc = Measures.get_precision_over_coverage(y_pred=y_pred, delta=delta)

                    ls_returns.append(ls_global_return)
                    lh_returns.append(lh_global_return)
                    sh_returns.append(sh_global_return)

                    ls_romads.append(ls_romad)
                    lh_romads.append(lh_romad)
                    sh_romads.append(sh_romad)

                    ls_mdds.append(ls_mdd)
                    lh_mdds.append(lh_mdd)
                    sh_mdds.append(sh_mdd)

                    longs_precisions.append(long['precision'])
                    shorts_precisions.append(short['precision'])

                    longs_coverage.append(label_coverage['long'])
                    shorts_coverage.append(label_coverage['short'])

                    longs_poc.append(long_poc)
                    shorts_poc.append(short_poc)

                    long_operations.append(long['count'])
                    short_operations.append(short['count'])
                    hold_operations.append(hold['count'])

                    
                net_json = {
                    "ls_returns": ls_returns,
                    "lh_returns": lh_returns,
                    "sh_returns": sh_returns,
                    "ls_romads": ls_romads,
                    "lh_romads": lh_romads,
                    "sh_romads": sh_romads,
                    "ls_mdds": ls_mdds,
                    "lh_mdds": lh_mdds,
                    "sh_mdds": sh_mdds,
                    "longs_precisions": longs_precisions,
                    "shorts_precisions": shorts_precisions,
                    "longs_coverage": longs_coverage,
                    "shorts_coverage": shorts_coverage,
                    "longs_poc": longs_poc,
                    "shorts_poc": shorts_poc,
                    "long_operations": long_operations,
                    "short_operations": short_operations,
                    "hold_operations": hold_operations
                }

                output_path = self.experiment_original_path + experiment_name + '/calculated_metrics/' + type + '/walk_' + str(index_walk) + '/' 

                if not os.path.isdir(output_path):
                    os.makedirs(output_path)

                with open(output_path + 'net_' + str(index_net) + '.json', 'w') as json_file:
                    json.dump(net_json, json_file, indent=4)
                    #json.dump(net_json, json_file)
                
                print(type + " - Salvate le metriche per walk n° ", index_walk, " rete: ", net)

                do_plot(walk=index_walk, net=index_net, type=type)

                all_ls_returns[index_net] = ls_returns
                all_lh_returns[index_net] = lh_returns
                all_sh_returns[index_net] = sh_returns

                all_ls_romads[index_net] = ls_romads
                all_lh_romads[index_net] = lh_romads
                all_sh_romads[index_net] = sh_romads

                all_ls_mdds[index_net] = ls_mdds
                all_lh_mdds[index_net] = lh_mdds
                all_sh_mdds[index_net] = sh_mdds

                all_longs_precisions[index_net] = longs_precisions
                all_shorts_precisions[index_net] = shorts_precisions

                all_longs_poc[index_net] = longs_poc
                all_shorts_poc[index_net] = shorts_poc

                all_longs_coverage[index_net] = longs_coverage
                all_shorts_coverage[index_net] = shorts_coverage

                all_long_operations[index_net] = long_operations
                all_short_operations[index_net] = short_operations
                all_hold_operations[index_net] = hold_operations

            avg_ls_returns = np.average(all_ls_returns, axis=0)
            avg_lh_returns = np.average(all_lh_returns, axis=0)
            avg_sh_returns = np.average(all_sh_returns, axis=0)

            avg_ls_mdds = np.average(all_ls_mdds, axis=0)
            avg_lh_mdds = np.average(all_lh_mdds, axis=0)
            avg_sh_mdds = np.average(all_sh_mdds, axis=0)

            avg_ls_romads = np.divide(avg_ls_returns, avg_ls_mdds)
            avg_lh_romads = np.divide(avg_lh_returns, avg_lh_mdds)
            avg_sh_romads = np.divide(avg_sh_returns, avg_sh_mdds)

            avg_longs_precisions = np.average(all_longs_precisions, axis=0)
            avg_shorts_precisions = np.average(all_shorts_precisions, axis=0)

            avg_longs_coverage = np.average(all_longs_coverage, axis=0)
            avg_shorts_coverage = np.average(all_shorts_coverage, axis=0)

            #avg_longs_poc = np.average(all_longs_poc, axis=0)
            #avg_shorts_poc = np.average(all_shorts_poc, axis=0)

            # NUOVO TEST AVG POC
            avg_longs_poc = np.divide(avg_longs_precisions, avg_longs_coverage)
            avg_shorts_poc = np.divide(avg_shorts_precisions, avg_shorts_coverage)

            '''
            for avg_id, avg in enumerate(avg_longs_poc):
                if avg_longs_poc[avg_id] > 0.8:
                    avg_longs_poc[avg_id] = (avg_longs_poc[avg_id] - 1 ) * 100

            for avg_id, avg in enumerate(avg_shorts_poc):
                if avg_shorts_poc[avg_id] > 0.8:
                    avg_shorts_poc[avg_id] = (avg_shorts_poc[avg_id] - 1 ) * 100
            '''

            avg_longs_poc = (avg_longs_poc - 1 ) * 100
            avg_shorts_poc = (avg_shorts_poc - 1 ) * 100

            avg_long_operations = np.average(all_long_operations, axis=0)
            avg_short_operations= np.average(all_short_operations, axis=0)
            avg_hold_operations = np.average(all_hold_operations, axis=0)



            avg_json = {
                    "ls_returns": avg_ls_returns,
                    "lh_returns": avg_lh_returns,
                    "sh_returns": avg_sh_returns,
                    "ls_romads": avg_ls_romads,
                    "lh_romads": avg_lh_romads,
                    "sh_romads": avg_sh_romads,
                    "ls_mdds": avg_ls_mdds,
                    "lh_mdds": avg_lh_mdds,
                    "sh_mdds": avg_sh_mdds,
                    "longs_precisions": avg_longs_precisions,
                    "shorts_precisions": avg_shorts_precisions,
                    "longs_coverage": avg_longs_coverage,
                    "shorts_coverage": avg_shorts_coverage,
                    "longs_poc": avg_longs_poc,
                    "shorts_poc": avg_shorts_poc,
                    "long_operations": avg_long_operations,
                    "short_operations": avg_short_operations,
                    "hold_operations": avg_hold_operations
                }

            avg_output_path = self.experiment_original_path + self.experiment_name + '/calculated_metrics/' + type + '/average/' 

            if not os.path.isdir(avg_output_path):
                os.makedirs(avg_output_path)

            with open(avg_output_path + 'walk_' + str(index_walk) + '.json', 'w') as json_file:
                json.dump(net_json, json_file, indent=4)
                #json.dump(net_json, json_file)
            
            print(type + " - Salvate le metriche AVG per walk n° ", index_walk)

            #do_plot_avg(walk=index_walk, type=type)
    

    '''
    '
    '''
    def calculate_custom_metrics(self):
        
        t1 = threading.Thread(target=self.generate_custom_metrics, args=(['Validation']))
        t2 = threading.Thread(target=self.generate_custom_metrics, args=(['Test']))

        t1.start()
        t2.start()

        t1.join()
        t2.join()
