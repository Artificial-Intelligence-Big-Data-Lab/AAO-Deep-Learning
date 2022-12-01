import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import sys
import cv2
import time
import pickle
import datetime
import functools
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from datetime import timedelta
from classes.Market import Market
from classes.Metrics import Metrics
from classes.Set import Set
from classes.Measures import Measures
from classes.Utils import create_folder, df_date_merger_binary
from cycler import cycler
import sklearn.metrics as sklm  
from sklearn.metrics import confusion_matrix
import json
import platform

# classification moment
from sklearn import model_selection
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, balanced_accuracy_score
from sklearn.preprocessing import LabelEncoder
import xgboost
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
import xlsxwriter


class XGBHandler:
    exp_base_path = 'C:/Users/Utente/Desktop/' 
    iperparameters = {}

    dataset = pd.DataFrame()
    dataset_label = pd.DataFrame()

    experiment_name = ''
    is_dates = {}
    
    #size_of_feature = 227 # risoluzione oraria
    # il numero di risoluzioni: 1h, 4h, 8h etc
    resolutions = 1

    thr_binary_labeling = -0.5
    balance_binary = True 
    predictions_dataset = 'sp500_cet'


    '''
    '
    '''
    def __init__(self, iperparameters):
        self.iperparameters = iperparameters
        
        self.experiment_name = iperparameters['experiment_name']
        self.is_dates = iperparameters['is_dates']

        self.thr_binary_labeling = iperparameters['thr_binary_labeling']
        self.balance_binary = iperparameters['balance_binary']
        self.predictions_dataset = iperparameters['predictions_dataset']

        self.dataset = Market(dataset=self.predictions_dataset)

        self.dataset_label = self.dataset.get_binary_labels(freq='1d', columns=['delta_current_day', 'delta_next_day', 'open', 'close', 'high', 'low'], thr=self.thr_binary_labeling)
        self.dataset_label = self.dataset_label.reset_index()
        self.dataset_label['date_time'] = self.dataset_label['date_time'].astype(str)
    
    '''
    ' Creo dei file json che verranno usati come 
    ' dataset per training e test set. Li creo per ridurre i tempi 
    ' di testing e caricamento del modello
    '''
    def create_x_y_dataset(self, dates=[], index_walk=0, size_of_feature=60):
        market = Market(dataset=self.predictions_dataset)
        market = market.get_binary_labels(freq='1d', columns=['delta_current_day', 'delta_next_day', 'close'], thr=self.thr_binary_labeling).reset_index()
        market = Market.get_df_by_data_range(df=market.copy(), start_date=dates[index_walk][0], end_date=dates[index_walk][1])


        # self.dataset contiene un dataset modificato a monte
        five_min_df = self.dataset.get()

        start_date = pd.to_datetime(dates[index_walk][0]) + timedelta(hours=23)
        end_date = pd.to_datetime(dates[index_walk][1]) + timedelta(hours=23)
        five_min_df = Market.get_df_by_data_range(df=five_min_df.copy(), start_date=start_date, end_date=end_date)

        # tolgo i primi TOT elementi della lista
        #x = np.zeros(shape=(market.shape[0] - self.size_of_feature, self.resolutions * self.size_of_feature))
        x = np.zeros(shape=(market.shape[0] - 1, self.resolutions * size_of_feature))

        # tolgo i primi TOT elementi della lista
        date_list       = market['date_time'].tolist()[1:]
        y_current_day   = market['label_current_day'].tolist()[1:]
        y_next_day      = market['label_next_day'].tolist()[1:]
        delta_next_day  = market['delta_next_day'].tolist()[1:]
        delta_curr_day  = market['delta_current_day'].tolist()[1:]
        close           = market['close'].tolist()[1:]

        date_list.reverse()
        y_current_day.reverse()
        y_next_day.reverse()
        delta_next_day.reverse()
        delta_curr_day.reverse()
        close.reverse()

        for i, date in enumerate(date_list): 
            subset_one_h = five_min_df.loc[five_min_df['date_time'] <= date + timedelta(hours=23)]
            
            feature = np.array(subset_one_h['delta_current_day_percentage'].tolist()[-size_of_feature:])
            x[i] = feature

        x = x.tolist()
        
        date_list = [d.strftime('%Y-%m-%d') for d in date_list]

        data = {
            'date_time': date_list,
            'x': x, 
            #'y_current_day': y_current_day,
            #'y_next_day': y_next_day,
            #'delta_current_day': delta_curr_day, 
            #'delta_next_day': delta_next_day, 
            #'close': close

        }

        return data
    
    '''
    ' Leggo i file json dataset e li restituisco dentro la classe Set
    ' così li posso usare nel nostro formato standard per fare training e test set
    '''
    def read_json_dataset(self, file, balance_binary): 
        dataset = {}
        with open(file) as json_file:
            dataset = json.load(json_file)

        x = np.array(dataset['x'])

        date_list       = dataset['date_time']
        #date_list       = dataset['date_list']

        df = pd.DataFrame()
        df['date_time'] = date_list
        df['date_time'] = df['date_time'].astype(str)
        #df = df_date_merger_binary(df=df.copy(), thr=self.thr_binary_labeling, columns=['delta_current_day', 'delta_next_day', 'open', 'close', 'high', 'low'], dataset=self.predictions_dataset)

        df = pd.merge(df, self.dataset_label, how="inner")

        y_current_day   = df['label_current_day'].tolist()
        y_next_day      = df['label_next_day'].tolist()
        delta_next_day  = df['delta_next_day'].tolist()
        delta_curr_day  = df['delta_current_day'].tolist()
        close           = df['close'].tolist()
        set = Set(date_time=date_list, x=x, y_current_day=y_current_day, y_next_day=y_next_day, delta_current_day=delta_curr_day, delta_next_day=delta_next_day, close=close, balance_binary=balance_binary)
        return set

    '''
    ' Metodo usato per generare tutti i json dataset
    ' C'è un for per IS1 e IS2, per la dimensione degli walk, 
    ' per la dimensione delle feature e per ogni walk.
    ' In tutto 4 for annidati
    '''
    def create_json_dataset(self): 
             
        #for index_is in ['is_1', 'is_2']: 
        #    for size_of_walk in ['2_anni', '1_anno', '6_mesi', '3_mesi']:
        #        for size_of_feature in range(60, 130, 10):
        
        for index_is in ['is_1']: 
            for size_of_walk in ['2_anni']:
                for size_of_feature in [96]:
                    training_dates = self.is_dates[index_is + '_training_set_' + size_of_walk]
                    test_dates = self.is_dates[index_is + '_test_set_' + size_of_walk]
                    number_of_walks = len(training_dates)

                    for index_walk in range(0, number_of_walks): 
                        print("[" + index_is + "]" + "[" + size_of_walk + "]" + "[N° Sample: " + str(size_of_feature) + "]" + "[WALK: " + str(index_walk) + "]")
                        
                        training_data = self.create_x_y_dataset(dates=training_dates, index_walk=index_walk, size_of_feature=size_of_feature)
                        test_data = self.create_x_y_dataset(dates=test_dates, index_walk=index_walk, size_of_feature=size_of_feature)

                        path = self.exp_base_path + self.experiment_name + '/json/' + index_is + '/walk_' + size_of_walk + '/' + str(size_of_feature) + '_sample/'
                        create_folder(path)

                        with open(path + 'training_walk_' + str(index_walk) + '.json', 'w') as training_json_file:
                            json.dump(training_data, training_json_file, indent=4)

                        with open(path + 'test_walk_' + str(index_walk) + '.json', 'w') as test_json_file:
                            json.dump(test_data, test_json_file, indent=4)

                            
                    print("\n")

    '''
    ' 4 for annidati. Per ogni dataset in .json
    ' creo il CSV con le predizioni per il test set
    '''
    def create_csv_predictions(self): 

        #for index_is in ['is_1', 'is_2']: 
        #    for size_of_walk in ['2_anni', '1_anno', '6_mesi', '3_mesi']:
        #        for size_of_feature in range(60, 130, 10):
        for index_is in ['is_1']: 
            for size_of_walk in ['2_anni']:
                for size_of_feature in [96]:
                #for size_of_feature in range(90, 130, 10):
                    input_path = self.exp_base_path + self.experiment_name + '/json/' + index_is + '/walk_' + size_of_walk + '/' + str(size_of_feature) + '_sample/'
                    
                    training_dates = self.is_dates[index_is + '_training_set_' + size_of_walk]
                    number_of_walks = len(training_dates)

                    for index_walk in range(0, number_of_walks): 
                        print("[" + index_is + "]" + "[" + size_of_walk + "]" + "[N° Sample: " + str(size_of_feature) + "]" + "[WALK: " + str(index_walk) + "]")
                        
                        
                        
                        #training_set = self.read_json_dataset(file='C:/Users/Utente/Desktop/json-csv risultati classificatori/walk 2 anni/classifier_json_dataset_96_sample_5min/sp500_walk_0_perc_0_training.json', balance_binary=True)
                        #test_set = self.read_json_dataset(file='C:/Users/Utente/Desktop/json-csv risultati classificatori/walk 2 anni/classifier_json_dataset_96_sample_5min/sp500_walk_0_perc_0_test.json', balance_binary=False)
                        training_set = self.read_json_dataset(file=input_path + 'training_walk_' + str(index_walk) + '.json', balance_binary=True)
                        test_set = self.read_json_dataset(file=input_path + 'test_walk_' + str(index_walk) + '.json', balance_binary=False)


                        learning_rate = 0.1
                        estimators = 500 #500 1000
                        max_depth = 4
                        scale_pos_weight = 4

                        model = xgboost.XGBClassifier(learning_rate=learning_rate, n_estimators=estimators, max_depth=max_depth, scale_pos_weight=scale_pos_weight)

                        x_train = training_set.get_x()
                        y_train = training_set.get_y(referred_to='next_day')

                        x_test = test_set.get_x()
                        #y_test = test_set.get_y(referred_to='next_day')

                        model.fit(x_train, y_train)
                        y_pred = model.predict(x_test)

                        dates_test = test_set.get_date_time()
                        dates_test.reverse()
                        y_pred = y_pred.tolist()
                        y_pred.reverse()

                        # shifto i valori per poterli usare su MC
                        df = pd.DataFrame()
                        df['date_time'] = dates_test
                        df['predictions'] = y_pred 
                        df['date_time'] = df['date_time'].shift(-1)
                        df = df.dropna()

                        output_path = self.exp_base_path + self.experiment_name + '/csv/' + index_is + '/walk_' + size_of_walk + '/' + str(size_of_feature) + '_sample/'
                        create_folder(output_path)

                        df.to_csv(output_path + 'test_walk_' + str(index_walk) + '.csv', header=True, index=False)
                    print("\n")
    
    '''
    ' 4 for annidati. 
    ' Creo un csv contenente per predizioni di tutti gli walk
    ' messi assieme con i valori del mercato per quel giorno
    '''
    def create_all_walks_csv(self):     
        #for index_is in ['is_1', 'is_2']: 
        #    for size_of_walk in ['2_anni', '1_anno', '6_mesi', '3_mesi']:
        #        for size_of_feature in range(60, 130, 10): 
        for index_is in ['is_1']: 
            for size_of_walk in ['2_anni']:
                for size_of_feature in [96]:
                    path = self.exp_base_path + self.experiment_name + '/csv/' + index_is + '/walk_' + size_of_walk + '/' + str(size_of_feature) + '_sample/'
                    
                    training_dates = self.is_dates[index_is + '_training_set_' + size_of_walk]
                    number_of_walks = len(training_dates)

                    df_final = pd.DataFrame()
                    for index_walk in range(0, number_of_walks): 
                        print("[" + index_is + "]" + "[" + size_of_walk + "]" + "[N° Sample: " + str(size_of_feature) + "]" + "[WALK: " + str(index_walk) + "]")                        
                        
                        test_set = pd.read_csv(path + 'test_walk_' + str(index_walk) + '.csv')

                        df_final = pd.concat([df_final, test_set])

                    df_final = df_date_merger_binary(df=df_final.copy(), thr=self.thr_binary_labeling, columns=['delta_current_day', 'open', 'close', 'high', 'low'], dataset=self.predictions_dataset)
                    df_final = df_final.rename(columns={"predictions": "decision"})
                    df_final.to_csv(path + 'test_all_walks.csv', header=True, index=False)
                    print("\n")

    '''
    '
    '''
    def calculate_results(self, insample='is_1'): 
        list_size_of_feature = list(range(60, 130, 10))

        returns_avg = []
        returns_std = []
        returns_wc = []
        returns_bc = []
        returns_walk_2y = []


        mdds_avg = []
        mdds_std = []
        mdds_wc = []
        mdds_bc = []
        mdds_walk_2y = []


        romads_avg = []
        romads_std = []
        romads_wc = []
        romads_bc = []
        romads_walk_2y = []

        returns_bh = []
        mdds_bh = []
        romads_bh = []

        returns_bh_i = []
        mdds_bh_i = []
        romads_bh_i = []

        precision_avg_up_label = []
        precision_avg_down_label = []
        precision_walk_2y_down_label = []

        cvg_random_up_label = []
        cvg_random_down_label = []
        cvg_up_label = []
        cvg_down_label = []
        cvg_up_walk_2y_label = []
        cvg_down_walk_2y_label = []

        precision_avg_up_delta = []
        precision_avg_down_delta = []
        precision_walk_2y_down_delta = []

        precision_random_up_delta = []
        precision_random_down_delta = []

        # per l'area 
        precision_bc_down_delta = []
        precision_wc_down_delta = []
        precision_bc_down_label = []
        precision_wc_down_label = []

        for i, size_of_feature in enumerate(list_size_of_feature):
            print("Calcolo i risultati con", size_of_feature, "numero di feature nei sample.")
            df_2y = pd.read_csv(self.exp_base_path + self.experiment_name + '/csv/' + insample + '/walk_2_anni/' + str(size_of_feature) + '_sample/test_all_walks.csv')

            df_1y = pd.read_csv(self.exp_base_path + self.experiment_name + '/csv/' + insample + '/walk_1_anno/' + str(size_of_feature) + '_sample/test_all_walks.csv')
            df_6m = pd.read_csv(self.exp_base_path + self.experiment_name + '/csv/' + insample + '/walk_6_mesi/' + str(size_of_feature) + '_sample/test_all_walks.csv')
            df_3m = pd.read_csv(self.exp_base_path + self.experiment_name + '/csv/' + insample + '/walk_3_mesi/' + str(size_of_feature) + '_sample/test_all_walks.csv')


            res_1y = Measures.get_equity_return_mdd_romad(df=df_1y.copy(), multiplier=50, type='short_only', penalty=25, stop_loss=1000, delta_to_use='delta_current_day', compact_results=True)
            res_6m = Measures.get_equity_return_mdd_romad(df=df_6m.copy(), multiplier=50, type='short_only', penalty=25, stop_loss=1000, delta_to_use='delta_current_day', compact_results=True)
            res_3m = Measures.get_equity_return_mdd_romad(df=df_3m.copy(), multiplier=50, type='short_only', penalty=25, stop_loss=1000, delta_to_use='delta_current_day', compact_results=True)

            confusion_1y = confusion_matrix(df_1y['label_current_day'].tolist(), df_1y['decision'].tolist(), normalize='pred', labels = [0, 1])
            confusion_6m = confusion_matrix(df_6m['label_current_day'].tolist(), df_6m['decision'].tolist(), normalize='pred', labels = [0, 1])
            confusion_3m = confusion_matrix(df_3m['label_current_day'].tolist(), df_3m['decision'].tolist(), normalize='pred', labels = [0, 1])

            cvg_1y = Measures.get_binary_coverage(y=df_1y['decision'].tolist())
            cvg_6m = Measures.get_binary_coverage(y=df_6m['decision'].tolist())
            cvg_3m = Measures.get_binary_coverage(y=df_3m['decision'].tolist())
            cvg_random = Measures.get_binary_coverage(y=df_1y['label_current_day'].tolist())
            
        
            delta_precision_1y = Measures.get_binary_delta_precision(y=df_1y['decision'].tolist(), delta=df_1y['delta_current_day'].tolist(), delta_val=-25)
            delta_precision_6m = Measures.get_binary_delta_precision(y=df_6m['decision'].tolist(), delta=df_6m['delta_current_day'].tolist(), delta_val=-25)
            delta_precision_3m = Measures.get_binary_delta_precision(y=df_3m['decision'].tolist(), delta=df_3m['delta_current_day'].tolist(), delta_val=-25)

            # Baseline BH
            res_bh = Measures.get_return_mdd_romad_bh(close=df_1y['close'].tolist(), multiplier=50, compact_results=True)
            res_bh_i = Measures.get_equity_return_mdd_romad(df=df_1y.copy(), multiplier=50, type='bh_long', penalty=25, stop_loss=1000, delta_to_use='delta_current_day', compact_results=True)

            # Results
            returns_avg.append(round((res_1y['return'] + res_6m['return'] + res_3m['return']) / 3, 2)) 
            returns_std.append(round(np.std([res_1y['return'], res_6m['return'], res_3m['return']]), 2))
            returns_wc.append(min([res_1y['return'], res_6m['return'], res_3m['return']]))
            returns_bc.append(max([res_1y['return'], res_6m['return'], res_3m['return']]))

            mdds_avg.append(round((res_1y['mdd'] + res_6m['mdd'] + res_3m['mdd']) / 3, 2))
            mdds_std.append(round(np.std([res_1y['mdd'], res_6m['mdd'], res_3m['mdd']]), 2))
            mdds_wc.append(max([res_1y['mdd'], res_6m['mdd'], res_3m['mdd']]))
            mdds_bc.append(min([res_1y['mdd'], res_6m['mdd'], res_3m['mdd']]))

            romads_avg.append(round((res_1y['romad'] + res_6m['romad'] + res_3m['romad']) / 3, 2))
            romads_std.append(round(np.std([res_1y['romad'], res_6m['romad'], res_3m['romad']]), 2))
            romads_wc.append(min([res_1y['romad'], res_6m['romad'], res_3m['romad']]))
            romads_bc.append(max([res_1y['romad'], res_6m['romad'], res_3m['romad']]))

            returns_bh.append(res_bh['return'])
            mdds_bh.append(res_bh['mdd'])
            romads_bh.append(round(res_bh['romad'], 2))

            returns_bh_i.append(res_bh_i['return'])
            mdds_bh_i.append(res_bh_i['mdd'])
            romads_bh_i.append(round(res_bh_i['romad'], 2))

            precision_avg_up_label.append(round(((confusion_1y[1][1] + confusion_6m[1][1] + confusion_3m[1][1]) / 3) * 100, 2)) 
            precision_avg_down_label.append(round(((confusion_1y[0][0] + confusion_6m[0][0] + confusion_3m[0][0]) / 3) * 100, 2)) 

            cvg_random_up_label.append(cvg_random['up_perc'])
            cvg_random_down_label.append(cvg_random['down_perc'])
            cvg_up_label.append(round((cvg_1y['up_perc'] + cvg_6m['up_perc'] + cvg_3m['up_perc']) / 3, 2))
            cvg_down_label.append(round((cvg_1y['down_perc'] + cvg_6m['down_perc'] + cvg_3m['down_perc']) / 3, 2))

            precision_avg_up_delta.append(round((delta_precision_1y['up'] + delta_precision_6m['up'] + delta_precision_3m['up']) / 3, 2))
            precision_avg_down_delta.append(round((delta_precision_1y['down'] + delta_precision_6m['down'] + delta_precision_3m['down']) / 3, 2))
            precision_random_up_delta.append(round((delta_precision_1y['random_up'] + delta_precision_6m['random_up'] + delta_precision_3m['random_up']) / 3, 2))
            precision_random_down_delta.append(round((delta_precision_1y['random_down'] + delta_precision_6m['random_down'] + delta_precision_3m['random_down']) / 3, 2))

            precision_bc_down_delta.append(max([delta_precision_1y['down'], delta_precision_6m['down'], delta_precision_3m['down']]) )
            precision_wc_down_delta.append(min([delta_precision_1y['down'], delta_precision_6m['down'], delta_precision_3m['down']]) )
            precision_bc_down_label.append(max([confusion_1y[0][0], confusion_6m[0][0], confusion_3m[0][0]]) * 100)
            precision_wc_down_label.append(min([confusion_1y[0][0], confusion_6m[0][0], confusion_3m[0][0]]) * 100)

            # WALK 2 ANNI COME LINEA SEPARATA
            res_2y = Measures.get_equity_return_mdd_romad(df=df_2y.copy(), multiplier=50, type='short_only', penalty=25, stop_loss=1000, delta_to_use='delta_current_day', compact_results=True)
            confusion_2y = confusion_matrix(df_2y['label_current_day'].tolist(), df_2y['decision'].tolist(), normalize='pred', labels = [0, 1])
            cvg_2y = Measures.get_binary_coverage(y=df_2y['decision'].tolist())
            delta_precision_2y = Measures.get_binary_delta_precision(y=df_2y['decision'].tolist(), delta=df_2y['delta_current_day'].tolist(), delta_val=-25)

            returns_walk_2y.append(res_2y['return'])
            mdds_walk_2y.append(res_2y['mdd'])
            romads_walk_2y.append(round(res_2y['romad'], 2))
            precision_walk_2y_down_label.append(confusion_2y[0][0] * 100)
            precision_walk_2y_down_delta.append(delta_precision_2y['down'])
            cvg_up_walk_2y_label.append(cvg_2y['up_perc'])
            cvg_down_walk_2y_label.append(cvg_2y['down_perc'])

        data = {
            'list_size_of_feature': list_size_of_feature,

            'returns_avg': returns_avg,
            'returns_wc': returns_wc,
            'returns_bc': returns_bc,
            'returns_std': returns_std,
            'returns_conf_up': returns_bc, #np.add(returns_avg, returns_std),
            'returns_conf_down': returns_wc, #np.subtract(returns_avg, returns_std),
            'returns_conf_diff': np.subtract(returns_wc, returns_bc), #np.subtract(np.subtract(returns_avg, returns_std), np.add(returns_avg, returns_std)),

            'mdds_avg': mdds_avg,
            'mdds_wc': mdds_wc,
            'mdds_bc': mdds_bc,
            'mdds_std': mdds_std,
            'mdds_conf_up': mdds_bc, #np.add(mdds_avg, mdds_std),
            'mdds_conf_down': mdds_wc, #np.subtract(mdds_avg, mdds_std),
            'mdds_conf_diff': np.subtract(mdds_bc, mdds_wc), #np.subtract(np.add(mdds_avg, mdds_std), np.subtract(mdds_avg, mdds_std)),

            'romads_avg': romads_avg,
            'romads_wc': romads_wc,
            'romads_bc': romads_bc,
            'romads_std': romads_std,
            'romads_conf_up': romads_bc, #np.add(romads_avg, romads_std),
            'romads_conf_down': romads_wc, #np.subtract(romads_avg, romads_std),
            'romads_conf_diff': np.subtract(romads_wc, romads_bc), #np.subtract(np.subtract(romads_avg, romads_std), np.add(romads_avg, romads_std)),

            'bh_romad': romads_bh, 
            'bh_return': returns_bh,
            'bh_mdd': mdds_bh,

            'bh_i_romad': romads_bh_i, 
            'bh_i_return': returns_bh_i,
            'bh_i_mdd': mdds_bh_i,

            'precision_avg_up_label': precision_avg_up_label,
            'precision_avg_down_label': precision_avg_down_label,

            'cvg_random_up_label': cvg_random_up_label,
            'cvg_random_down_label': cvg_random_down_label,
            'cvg_up_label': cvg_up_label, 
            'cvg_down_label': cvg_down_label,

            'precision_avg_up_delta': precision_avg_up_delta,
            'precision_avg_down_delta': precision_avg_down_delta,
            'precision_random_up_delta': precision_random_up_delta,
            'precision_random_down_delta': precision_random_down_delta,

            'precision_bc_down_delta': precision_bc_down_delta,
            'precision_wc_down_delta': precision_wc_down_delta,
            'precision_bc_down_label': precision_bc_down_label,
            'precision_wc_down_label': precision_wc_down_label,

            'precision_down_delta_diff': np.subtract(precision_wc_down_delta, precision_bc_down_delta),
            'precision_down_label_diff': np.subtract(precision_wc_down_label, precision_bc_down_label),

            'returns_walk_2y': returns_walk_2y,
            'mdds_walk_2y': mdds_walk_2y,
            'romads_walk_2y': romads_walk_2y,
            'precision_walk_2y_down_label': precision_walk_2y_down_label,
            'precision_walk_2y_down_delta': precision_walk_2y_down_delta,
            'cvg_up_walk_2y_label': cvg_up_walk_2y_label,
            'cvg_down_walk_2y_label': cvg_down_walk_2y_label
        }
        return data
    
    '''
    '
    '''
    def generate_excel(self):
        is_1 = self.calculate_results(insample='is_1')
        is_2 = self.calculate_results(insample='is_2')

        number_of_row = len(is_1['list_size_of_feature']) + 1
        # Create an new Excel file and add a worksheet.
        workbook = xlsxwriter.Workbook(self.exp_base_path + self.experiment_name + '/' + self.experiment_name + ' labeling.xlsx')
        worksheet = workbook.add_worksheet('GXB_Swipe')

        # Stampo n° sample
        worksheet.write('AAZ1', 'N° Sample')
        worksheet.write_column('AAZ2', is_1['list_size_of_feature'])
        
        # stampo i valori per is1
        worksheet.write('ABA1', 'IS1 Return Avg') # RETURN
        worksheet.write('ABB1', 'IS1 Return Var')
        worksheet.write('ABC1', 'IS1 Return WC')
        worksheet.write('ABD1', 'IS1 MDD Avg') # MDD
        worksheet.write('ABE1', 'IS1 MDD Var')
        worksheet.write('ABF1', 'IS1 MDD WC')

        worksheet.write('ABG1', 'IS1 Romad Avg') # ROMAD
        worksheet.write('ABH1', 'IS1 Romad Var')
        worksheet.write('ABI1', 'IS1 Romad WC')

        worksheet.write('ABJ1', 'IS1 BH Return') # BH
        worksheet.write('ABK1', 'IS1 BH MDD')
    
        worksheet.write('ABL1', 'IS1 BH Romad')

        worksheet.write('ABM1', 'IS1 BH Intra Return') # BH INTRA
        worksheet.write('ABN1', 'IS1 BH Intra MDD')

        worksheet.write('ABO1', 'IS1 BH Intra Romad')

        worksheet.write('ABP1', 'IS1 Return Conf Up')
        worksheet.write('ABQ1', 'IS1 Return Conf Down')
        worksheet.write('ABR1', 'IS1 Return Conf Diff')
        worksheet.write('ABS1', 'IS1 MDD Conf Up')
        worksheet.write('ABT1', 'IS1 MDD Conf Down')
        worksheet.write('ABU1', 'IS1 MDD Conf Diff')

        worksheet.write('ABV1', 'IS1 Romad Conf Up')
        worksheet.write('ABW1', 'IS1 Romad Conf Down')
        worksheet.write('ABZ1', 'IS1 Romad Conf Diff')

        worksheet.write('ACA1', 'Precision avg up label')
        worksheet.write('ACB1', 'Precision avg down label')
        worksheet.write('ACC1', 'CVG Random Up Label')
        worksheet.write('ACD1', 'CVG Random Down Label')
        worksheet.write('ACE1', 'CVG Up Label')
        worksheet.write('ACF1', 'CVG Down Label')
        worksheet.write('ACG1', 'Precision avg up delta')
        worksheet.write('ACH1', 'Precision avg down delta')
        worksheet.write('ACI1', 'Precision random up delta')
        worksheet.write('ACJ1', 'Precision random up delta')

        worksheet.write('ACK1', 'Precision bc down delta')
        worksheet.write('ACL1', 'Precision diff delta')
        worksheet.write('ACM1', 'Precision bc down label')
        worksheet.write('ACN1', 'Precision diff label')

        worksheet.write_column('ABA2', is_1['returns_avg']) # RETURN
        worksheet.write_column('ABB2', is_1['returns_std'])
        worksheet.write_column('ABC2', is_1['returns_wc'])
        worksheet.write_column('ABD2', is_1['mdds_avg']) # MDD
        worksheet.write_column('ABE2', is_1['mdds_std'])
        worksheet.write_column('ABF2', is_1['mdds_wc'])
        worksheet.write_column('ABG2', is_1['romads_avg']) # ROMAD
        worksheet.write_column('ABH2', is_1['romads_std'])
        worksheet.write_column('ABI2', is_1['romads_wc'])
        worksheet.write_column('ABJ2', is_1['bh_return']) # BH INTRA
        worksheet.write_column('ABK2', is_1['bh_mdd'])
        worksheet.write_column('ABL2', is_1['bh_romad'])
        worksheet.write_column('ABM2', is_1['bh_i_return']) # BH INTRA
        worksheet.write_column('ABN2', is_1['bh_i_mdd'])
        worksheet.write_column('ABO2', is_1['bh_i_romad'])
        worksheet.write_column('ABP2', is_1['returns_conf_up'])
        worksheet.write_column('ABQ2', is_1['returns_conf_down'])
        worksheet.write_column('ABR2', is_1['returns_conf_diff'])
        worksheet.write_column('ABS2', is_1['mdds_conf_up'])
        worksheet.write_column('ABT2', is_1['mdds_conf_down'])
        worksheet.write_column('ABU2', is_1['mdds_conf_diff'])
        worksheet.write_column('ABV2', is_1['romads_conf_up'])
        worksheet.write_column('ABW2', is_1['romads_conf_down'])
        worksheet.write_column('ABZ2', is_1['romads_conf_diff'])

        worksheet.write_column('ACA2', is_1['precision_avg_up_label'])
        worksheet.write_column('ACB2', is_1['precision_avg_down_label'])
        worksheet.write_column('ACC2', is_1['cvg_random_up_label'])
        worksheet.write_column('ACD2', is_1['cvg_random_down_label'])
        worksheet.write_column('ACE2', is_1['cvg_up_label'])
        worksheet.write_column('ACF2', is_1['cvg_down_label'])

        worksheet.write_column('ACG2', is_1['precision_avg_up_delta'])
        worksheet.write_column('ACH2', is_1['precision_avg_down_delta'])
        worksheet.write_column('ACI2', is_1['precision_random_up_delta'])
        worksheet.write_column('ACJ2', is_1['precision_random_down_delta'])

        worksheet.write_column('ACK2', is_1['precision_bc_down_delta'])
        worksheet.write_column('ACL2', is_1['precision_down_delta_diff'])

        worksheet.write_column('ACM2', is_1['precision_bc_down_label'])
        worksheet.write_column('ACN2', is_1['precision_down_label_diff'])

        # stampo i valori per is2 - Salto alla lettera D PER avere più spazio
        worksheet.write('ADA1', 'IS2 Return Avg') # RETURN
        worksheet.write('ADB1', 'IS2 Return Var')
        worksheet.write('ADC1', 'IS2 Return WC')
        worksheet.write('ADD1', 'IS2 MDD Avg') # MDD
        worksheet.write('ADE1', 'IS2 MDD Var')
        worksheet.write('ADF1', 'IS2 MDD WC')
        worksheet.write('ADG1', 'IS2 Romad Avg') # ROMAD
        worksheet.write('ADH1', 'IS2 Romad Var')
        worksheet.write('ADI1', 'IS2 Romad WC')
        worksheet.write('ADJ1', 'IS2 BH Return') # BH
        worksheet.write('ADK1', 'IS2 BH MDD')
        worksheet.write('ADL1', 'IS2 BH Romad')
        worksheet.write('ADM1', 'IS2 BH Intra Return') # BH INTRA
        worksheet.write('ADN1', 'IS2 BH Intra MDD')
        worksheet.write('ADO1', 'IS2 BH Intra Romad')
        worksheet.write('ADP1', 'IS2 Return Conf Up')
        worksheet.write('ADQ1', 'IS2 Return Conf Down')
        worksheet.write('ADR1', 'IS2 Return Conf Diff')
        worksheet.write('ADS1', 'IS2 MDD Conf Up')
        worksheet.write('ADT1', 'IS2 MDD Conf Down')
        worksheet.write('ADU1', 'IS2 MDD Conf Diff')
        worksheet.write('ADV1', 'IS2 Romad Conf Up')
        worksheet.write('ADW1', 'IS2 Romad Conf Down')
        worksheet.write('ADZ1', 'IS2 Romad Conf Diff')
        worksheet.write('AEA1', 'Precision avg up label')
        worksheet.write('AEB1', 'Precision avg down label')
        worksheet.write('AEC1', 'CVG Random Up Label')
        worksheet.write('AED1', 'CVG Random Down Label')
        worksheet.write('AEE1', 'CVG Up Label')
        worksheet.write('AEF1', 'CVG Down Label')
        worksheet.write('AEG1', 'Precision avg up delta')
        worksheet.write('AEH1', 'Precision avg down delta')
        worksheet.write('AEI1', 'Precision random up delta')
        worksheet.write('AEJ1', 'Precision random up delta')
        worksheet.write('AEK1', 'Precision bc down delta')
        worksheet.write('AEL1', 'Precision diff delta')
        worksheet.write('AEM1', 'Precision bc down label')
        worksheet.write('AEN1', 'Precision diff label')

        worksheet.write_column('ADA2', is_2['returns_avg']) # RETURN
        worksheet.write_column('ADB2', is_2['returns_std'])
        worksheet.write_column('ADC2', is_2['returns_wc'])
        worksheet.write_column('ADD2', is_2['mdds_avg']) # MDD
        worksheet.write_column('ADE2', is_2['mdds_std'])
        worksheet.write_column('ADF2', is_2['mdds_wc'])
        worksheet.write_column('ADG2', is_2['romads_avg']) # ROMAD
        worksheet.write_column('ADH2', is_2['romads_std'])
        worksheet.write_column('ADI2', is_2['romads_wc'])
        worksheet.write_column('ADJ2', is_2['bh_return']) # BH INTRA
        worksheet.write_column('ADK2', is_2['bh_mdd'])
        worksheet.write_column('ADL2', is_2['bh_romad'])
        worksheet.write_column('ADM2', is_2['bh_i_return']) # BH INTRA
        worksheet.write_column('ADN2', is_2['bh_i_mdd'])
        worksheet.write_column('ADO2', is_2['bh_i_romad'])
        worksheet.write_column('ADP2', is_2['returns_conf_up'])
        worksheet.write_column('ADQ2', is_2['returns_conf_down'])
        worksheet.write_column('ADR2', is_2['returns_conf_diff'])
        worksheet.write_column('ADS2', is_2['mdds_conf_up'])
        worksheet.write_column('ADT2', is_2['mdds_conf_down'])
        worksheet.write_column('ADU2', is_2['mdds_conf_diff'])
        worksheet.write_column('ADV2', is_2['romads_conf_up'])
        worksheet.write_column('ADW2', is_2['romads_conf_down'])
        worksheet.write_column('ADZ2', is_2['romads_conf_diff'])

        worksheet.write_column('AEA2', is_2['precision_avg_up_label'])
        worksheet.write_column('AEB2', is_2['precision_avg_down_label'])
        worksheet.write_column('AEC2', is_2['cvg_random_up_label'])
        worksheet.write_column('AED2', is_2['cvg_random_down_label'])
        worksheet.write_column('AEE2', is_2['cvg_up_label'])
        worksheet.write_column('AEF2', is_2['cvg_down_label'])

        worksheet.write_column('AEG2', is_2['precision_avg_up_delta'])
        worksheet.write_column('AEH2', is_2['precision_avg_down_delta'])
        worksheet.write_column('AEI2', is_2['precision_random_up_delta'])
        worksheet.write_column('AEJ2', is_2['precision_random_down_delta'])

        worksheet.write_column('AEK2', is_2['precision_bc_down_delta'])
        worksheet.write_column('AEL2', is_2['precision_down_delta_diff'])
        worksheet.write_column('AEM2', is_2['precision_bc_down_label'])
        worksheet.write_column('AEN2', is_2['precision_down_label_diff'])

        # walk 2 baseline
        worksheet.write_column('BAA1', ['Return IS1', -1200, -1200, -1200, -1200, -1200, -1200, -1200])
        worksheet.write_column('BAB1', ['MDD IS1', 2662.5, 2662.5, 2662.5, 2662.5, 2662.5, 2662.5, 2662.5])
        worksheet.write_column('BAC1', ['Romad IS1', -0.45, -0.45, -0.45, -0.45, -0.45, -0.45, -0.45])

        worksheet.write_column('BAD1', ['Return IS2', -6912.5, -6912.5, -6912.5, -6912.5, -6912.5, -6912.5, -6912.5])
        worksheet.write_column('BAE1', ['MDD IS2', 6912.5, 6912.5, 6912.5, 6912.5, 6912.5, 6912.5, 6912.5])
        worksheet.write_column('BAF1', ['Romad IS2', -1, -1, -1, -1, -1, -1, -1])

        worksheet.write_column('BAG1', ['Label precision Down IS1', 28, 28, 28, 28, 28, 28, 28, ])
        worksheet.write_column('BAH1', ['Delta precision Down IS1', 59.7, 59.7, 59.7, 59.7, 59.7, 59.7, 59.7])
        worksheet.write_column('BAI1', ['Label precision down IS2', 14.2, 14.2, 14.2, 14.2, 14.2, 14.2, 14.2])
        worksheet.write_column('BAJ1', ['Delta precision down IS2', 42.8, 42.8, 42.8, 42.8, 42.8, 42.8, 42.8])

        worksheet.write_column('BBA2', is_1['returns_walk_2y'])
        worksheet.write_column('BBB2', is_1['mdds_walk_2y'])
        worksheet.write_column('BBC2', is_1['romads_walk_2y'])
        worksheet.write_column('BBD2', is_1['precision_walk_2y_down_label'])
        worksheet.write_column('BBE2', is_1['precision_walk_2y_down_delta'])
        worksheet.write_column('BBF2', is_1['cvg_up_walk_2y_label'])
        worksheet.write_column('BBG2', is_1['cvg_down_walk_2y_label'])

        worksheet.write_column('BBH2', is_2['returns_walk_2y'])
        worksheet.write_column('BBI2', is_2['mdds_walk_2y'])
        worksheet.write_column('BBJ2', is_2['romads_walk_2y'])
        worksheet.write_column('BBK2', is_2['precision_walk_2y_down_label'])
        worksheet.write_column('BBL2', is_2['precision_walk_2y_down_delta'])
        worksheet.write_column('BBM2', is_2['cvg_up_walk_2y_label'])
        worksheet.write_column('BBN2', is_2['cvg_down_walk_2y_label'])
        # IS1 RETURNS
        chart = workbook.add_chart({'type': 'line'})
        chart.add_series({'name': 'AVG',
                          'categories': '=GXB_Swipe!$AAZ$2:$AAZ$' + str(number_of_row),
                          'values': '=GXB_Swipe!$ABA$2:$ABA$' + str(number_of_row)
                          })
        chart.add_series({'name': 'BH',
                          'categories': '=GXB_Swipe!$AAZ$2:$AAZ$' + str(number_of_row),
                          'values': '=GXB_Swipe!$ABJ$2:$ABJ$' + str(number_of_row),
                          'line':   {'color': 'orange', 'dash_type': 'dash'}
                          })
        chart.add_series({'name': 'BH Intra',
                          'categories': '=GXB_Swipe!$AAZ$2:$AAZ$' + str(number_of_row),
                          'values': '=GXB_Swipe!$ABM$2:$ABM$' + str(number_of_row),
                          'line':   {'color': 'purple', 'dash_type': 'dash'}
                          })
        chart.add_series({'name': 'walk 2 anni 96 sample',
                          'categories': '=GXB_Swipe!$AAZ$2:$AAZ$' + str(number_of_row),
                          'values': '=GXB_Swipe!$BAA$2:$BAA$' + str(number_of_row),
                          'line':   {'color': 'green', 'dash_type': 'dash'}
                          })
        chart.add_series({'name': 'walk 2 anni',
                          'categories': '=GXB_Swipe!$AAZ$2:$AAZ$' + str(number_of_row),
                          'values': '=GXB_Swipe!$BBA$2:$BBA$' + str(number_of_row),
                          'line':   {'color': 'brown', 'dash_type': 'dash'}
                          })
        chart_area = workbook.add_chart({'type': 'area', 'subtype': 'stacked'})
        chart_area.add_series({'name': '',
                          'categories': '=GXB_Swipe!$AAZ$2:$AAZ$' + str(number_of_row),
                          'values': '=GXB_Swipe!$ABP$2:$ABP$' + str(number_of_row),
                          'fill':   {'color': 'white', 'transparency': 100}
                          })
        chart_area.add_series({'name': 'confidence',
                          'categories': '=GXB_Swipe!$AAZ$2:$AAZ$' + str(number_of_row),
                          'values': '=GXB_Swipe!$ABR$2:$ABR$' + str(number_of_row),
                          'fill':   {'color': '#E4EDF8'}
                          })

        chart.combine(chart_area)

        chart.set_title({'name': 'Returns IS1'})
        chart.set_x_axis({'name': 'N° Sample'})
        chart.set_y_axis({'name': '($)'})
        chart.set_style(2)
        chart.set_size({'width': 600, 'height': 350})
        worksheet.insert_chart('A1', chart)
        
        # IS2 RETURNS
        chart = workbook.add_chart({'type': 'line'})
        chart.add_series({'name': 'AVG',
                          'categories': '=GXB_Swipe!$AAZ$2:$AAZ$' + str(number_of_row),
                          'values': '=GXB_Swipe!$ADA$2:$ADA$' + str(number_of_row)
                          })
        chart.add_series({'name': 'BH',
                          'categories': '=GXB_Swipe!$AAZ$2:$AAZ$' + str(number_of_row),
                          'values': '=GXB_Swipe!$ADJ$2:$ADJ$' + str(number_of_row),
                          'line':   {'color': 'orange', 'dash_type': 'dash'}
                          })
        chart.add_series({'name': 'BH Intra',
                          'categories': '=GXB_Swipe!$AAZ$2:$AAZ$' + str(number_of_row),
                          'values': '=GXB_Swipe!$ADM$2:$ADM$' + str(number_of_row),
                          'line':   {'color': 'purple', 'dash_type': 'dash'}
                          })
        chart.add_series({'name': 'walk 2 anni 96 sample',
                          'categories': '=GXB_Swipe!$AAZ$2:$AAZ$' + str(number_of_row),
                          'values': '=GXB_Swipe!$BAD$2:$BAD$' + str(number_of_row),
                          'line':   {'color': 'green', 'dash_type': 'dash'}
                          })
        chart.add_series({'name': 'walk 2 anni',
                          'categories': '=GXB_Swipe!$AAZ$2:$AAZ$' + str(number_of_row),
                          'values': '=GXB_Swipe!$BBH$2:$BBH$' + str(number_of_row),
                          'line':   {'color': 'brown', 'dash_type': 'dash'}
                          })
        chart_area = workbook.add_chart({'type': 'area', 'subtype': 'stacked'})
        chart_area.add_series({'name': '',
                          'categories': '=GXB_Swipe!$AAZ$2:$AAZ$' + str(number_of_row),
                          'values': '=GXB_Swipe!$ADP$2:$ADP$' + str(number_of_row),
                          'fill':   {'color': 'white', 'transparency': 100}
                          })
        chart_area.add_series({'name': 'confidence',
                          'categories': '=GXB_Swipe!$AAZ$2:$AAZ$' + str(number_of_row),
                          'values': '=GXB_Swipe!$ADR$2:$ADR$' + str(number_of_row),
                          'fill':   {'color': '#E4EDF8'}
                          })

        chart.combine(chart_area)

        chart.set_title({'name': 'Returns IS2'})
        chart.set_x_axis({'name': 'N° Sample'})
        chart.set_y_axis({'name': '($)'})
        chart.set_style(2)
        chart.set_size({'width': 600, 'height': 350})
        worksheet.insert_chart('A19', chart)
        
        # IS1 MDDS
        chart = workbook.add_chart({'type': 'line'})
        chart.add_series({'name': 'AVG',
                            'categories': '=GXB_Swipe!$AAZ$2:$AAZ$' + str(number_of_row),
                            'values': '=GXB_Swipe!$ABD$2:$ABD$' + str(number_of_row)
                            })
        chart.add_series({'name': 'BH',
                            'categories': '=GXB_Swipe!$AAZ$2:$AAZ$' + str(number_of_row),
                            'values': '=GXB_Swipe!$ABK$2:$ABK$' + str(number_of_row),
                            'line':   {'color': 'orange', 'dash_type': 'dash'}
                            })
        chart.add_series({'name': 'BH Intra',
                            'categories': '=GXB_Swipe!$AAZ$2:$AAZ$' + str(number_of_row),
                            'values': '=GXB_Swipe!$ABN$2:$ABN$' + str(number_of_row),
                            'line':   {'color': 'purple', 'dash_type': 'dash'}
                            })
        chart.add_series({'name': 'walk 2 anni 96 sample',
                          'categories': '=GXB_Swipe!$AAZ$2:$AAZ$' + str(number_of_row),
                          'values': '=GXB_Swipe!$BAB$2:$BAB$' + str(number_of_row),
                          'line':   {'color': 'green', 'dash_type': 'dash'}
                          })
        chart.add_series({'name': 'walk 2 anni',
                          'categories': '=GXB_Swipe!$AAZ$2:$AAZ$' + str(number_of_row),
                          'values': '=GXB_Swipe!$BBB$2:$BBB$' + str(number_of_row),
                          'line':   {'color': 'brown', 'dash_type': 'dash'}
                          })
        chart_area = workbook.add_chart({'type': 'area', 'subtype': 'stacked'})
        chart_area.add_series({'name': '',
                            'categories': '=GXB_Swipe!$AAZ$2:$AAZ$' + str(number_of_row),
                            'values': '=GXB_Swipe!$ABT$2:$ABT$' + str(number_of_row),
                            'fill':   {'color': 'white', 'transparency': 100}
                            })
        chart_area.add_series({'name': 'confidence',
                            'categories': '=GXB_Swipe!$AAZ$2:$AAZ$' + str(number_of_row),
                            'values': '=GXB_Swipe!$ABU$2:$ABU$' + str(number_of_row),
                            'fill':   {'color': '#E4EDF8'}
                            })

        chart.combine(chart_area)

        chart.set_title({'name': 'MDD IS1'})
        chart.set_x_axis({'name': 'N° Sample'})
        chart.set_y_axis({'name': '($)'})
        chart.set_style(2)
        chart.set_size({'width': 600, 'height': 350})
        worksheet.insert_chart('K1', chart)

        # IS2 MDDS
        chart = workbook.add_chart({'type': 'line'})
        chart.add_series({'name': 'AVG',
                            'categories': '=GXB_Swipe!$AAZ$2:$AAZ$' + str(number_of_row),
                            'values': '=GXB_Swipe!$ADD$2:$ADD$' + str(number_of_row)
                            })
        chart.add_series({'name': 'BH',
                            'categories': '=GXB_Swipe!$AAZ$2:$AAZ$' + str(number_of_row),
                            'values': '=GXB_Swipe!$ADK$2:$ADK$' + str(number_of_row),
                            'line':   {'color': 'orange', 'dash_type': 'dash'}
                            })
        chart.add_series({'name': 'BH Intra',
                            'categories': '=GXB_Swipe!$AAZ$2:$AAZ$' + str(number_of_row),
                            'values': '=GXB_Swipe!$ADN$2:$ADN$' + str(number_of_row),
                            'line':   {'color': 'purple', 'dash_type': 'dash'}
                            })
        chart.add_series({'name': 'walk 2 anni 96 sample',
                          'categories': '=GXB_Swipe!$AAZ$2:$AAZ$' + str(number_of_row),
                          'values': '=GXB_Swipe!$BAE$2:$BAE$' + str(number_of_row),
                          'line':   {'color': 'green', 'dash_type': 'dash'}
                          })
        chart.add_series({'name': 'walk 2 anni',
                          'categories': '=GXB_Swipe!$AAZ$2:$AAZ$' + str(number_of_row),
                          'values': '=GXB_Swipe!$BBI$2:$BBI$' + str(number_of_row),
                          'line':   {'color': 'brown', 'dash_type': 'dash'}
                          })
        chart_area = workbook.add_chart({'type': 'area', 'subtype': 'stacked'})
        chart_area.add_series({'name': '',
                            'categories': '=GXB_Swipe!$AAZ$2:$AAZ$' + str(number_of_row),
                            'values': '=GXB_Swipe!$ADT$2:$ADT$' + str(number_of_row),
                            'fill':   {'color': 'white', 'transparency': 100}
                            })
        chart_area.add_series({'name': 'confidence',
                            'categories': '=GXB_Swipe!$AAZ$2:$AAZ$' + str(number_of_row),
                            'values': '=GXB_Swipe!$ADU$2:$ADU$' + str(number_of_row),
                            'fill':   {'color': '#E4EDF8'}
                            })

        chart.combine(chart_area)

        chart.set_title({'name': 'MDD IS2'})
        chart.set_x_axis({'name': 'N° Sample'})
        chart.set_y_axis({'name': '($)'})
        chart.set_style(2)
        chart.set_size({'width': 600, 'height': 350})
        worksheet.insert_chart('K19', chart)

        # IS1 ROMADS
        chart = workbook.add_chart({'type': 'line'})
        chart.add_series({'name': 'AVG',
                            'categories': '=GXB_Swipe!$AAZ$2:$AAZ$' + str(number_of_row),
                            'values': '=GXB_Swipe!$ABG$2:$ABG$' + str(number_of_row)
                            })
        chart.add_series({'name': 'BH',
                            'categories': '=GXB_Swipe!$AAZ$2:$AAZ$' + str(number_of_row),
                            'values': '=GXB_Swipe!$ABL$2:$ABL$' + str(number_of_row),
                            'line':   {'color': 'orange', 'dash_type': 'dash'}
                            })
        chart.add_series({'name': 'BH Intra',
                            'categories': '=GXB_Swipe!$AAZ$2:$AAZ$' + str(number_of_row),
                            'values': '=GXB_Swipe!$ABO$2:$ABO$' + str(number_of_row),
                            'line':   {'color': 'purple', 'dash_type': 'dash'}
                            })
        chart.add_series({'name': 'walk 2 anni 96 sample',
                          'categories': '=GXB_Swipe!$AAZ$2:$AAZ$' + str(number_of_row),
                          'values': '=GXB_Swipe!$BAC$2:$BAC$' + str(number_of_row),
                          'line':   {'color': 'green', 'dash_type': 'dash'}
                          })
        chart.add_series({'name': 'walk 2 anni',
                          'categories': '=GXB_Swipe!$AAZ$2:$AAZ$' + str(number_of_row),
                          'values': '=GXB_Swipe!$BBC$2:$BBC$' + str(number_of_row),
                          'line':   {'color': 'brown', 'dash_type': 'dash'}
                          })
                          
        chart_area = workbook.add_chart({'type': 'area', 'subtype': 'stacked'})
        chart_area.add_series({'name': '',
                            'categories': '=GXB_Swipe!$AAZ$2:$AAZ$' + str(number_of_row),
                            'values': '=GXB_Swipe!$ABV$2:$ABV$' + str(number_of_row),
                            'fill':   {'color': 'white', 'transparency': 100}
                            })
        chart_area.add_series({'name': 'confidence',
                            'categories': '=GXB_Swipe!$AAZ$2:$AAZ$' + str(number_of_row),
                            'values': '=GXB_Swipe!$ABZ$2:$ABZ$' + str(number_of_row),
                            'fill':   {'color': '#E4EDF8'}
                            })

        chart.combine(chart_area)

        chart.set_title({'name': 'Romads IS1'})
        chart.set_x_axis({'name': 'N° Sample'})
        chart.set_y_axis({'name': 'Points'})
        chart.set_style(2)
        chart.set_size({'width': 600, 'height': 350})
        worksheet.insert_chart('U1', chart)

        # IS2 ROMADS
        chart = workbook.add_chart({'type': 'line'})
        chart.add_series({'name': 'AVG',
                            'categories': '=GXB_Swipe!$AAZ$2:$AAZ$' + str(number_of_row),
                            'values': '=GXB_Swipe!$ADG$2:$ADG$' + str(number_of_row)
                            })
        chart.add_series({'name': 'BH',
                            'categories': '=GXB_Swipe!$AAZ$2:$AAZ$' + str(number_of_row),
                            'values': '=GXB_Swipe!$ADL$2:$ADL$' + str(number_of_row),
                            'line':   {'color': 'orange', 'dash_type': 'dash'}
                            })
        chart.add_series({'name': 'BH Intra',
                            'categories': '=GXB_Swipe!$AAZ$2:$AAZ$' + str(number_of_row),
                            'values': '=GXB_Swipe!$ADO$2:$ADO$' + str(number_of_row),
                            'line':   {'color': 'purple', 'dash_type': 'dash'}
                            })
        chart.add_series({'name': 'walk 2 anni 96 sample',
                          'categories': '=GXB_Swipe!$AAZ$2:$AAZ$' + str(number_of_row),
                          'values': '=GXB_Swipe!$BAF$2:$BAF$' + str(number_of_row),
                          'line':   {'color': 'green', 'dash_type': 'dash'}
                          })
        chart.add_series({'name': 'walk 2 anni',
                          'categories': '=GXB_Swipe!$AAZ$2:$AAZ$' + str(number_of_row),
                          'values': '=GXB_Swipe!$BBJ$2:$BBJ$' + str(number_of_row),
                          'line':   {'color': 'brown', 'dash_type': 'dash'}
                          })
        chart_area = workbook.add_chart({'type': 'area', 'subtype': 'stacked'})
        chart_area.add_series({'name': '',
                            'categories': '=GXB_Swipe!$AAZ$2:$AAZ$' + str(number_of_row),
                            'values': '=GXB_Swipe!$ADV$2:$ADV$' + str(number_of_row),
                            'fill':   {'color': 'white', 'transparency': 100}
                            })
        chart_area.add_series({'name': 'confidence',
                            'categories': '=GXB_Swipe!$AAZ$2:$AAZ$' + str(number_of_row),
                            'values': '=GXB_Swipe!$ADZ$2:$ADZ$' + str(number_of_row),
                            'fill':   {'color': '#E4EDF8'}
                            })

        chart.combine(chart_area)

        chart.set_title({'name': 'Romads IS2'})
        chart.set_x_axis({'name': 'N° Sample'})
        chart.set_y_axis({'name': 'Points'})
        chart.set_style(2)
        chart.set_size({'width': 600, 'height': 350})
        worksheet.insert_chart('U19', chart)


        # Precision Label IS1
        chart = workbook.add_chart({'type': 'line'})
        chart.add_series({'name': 'AVG Up',
                            'categories': '=GXB_Swipe!$AAZ$2:$AAZ$' + str(number_of_row),
                            'values': '=GXB_Swipe!$ACA$2:$ACA$' + str(number_of_row)
                            })
        chart.add_series({'name': 'Avg Down',
                            'categories': '=GXB_Swipe!$AAZ$2:$AAZ$' + str(number_of_row),
                            'values': '=GXB_Swipe!$ACB$2:$ACB$' + str(number_of_row)
                            })
        chart.add_series({'name': 'Random Up',
                            'categories': '=GXB_Swipe!$AAZ$2:$AAZ$' + str(number_of_row),
                            'values': '=GXB_Swipe!$ACC$2:$ACC$' + str(number_of_row),
                            'line':   {'color': 'orange', 'dash_type': 'dash'}
                            })
        chart.add_series({'name': 'Random Down',
                            'categories': '=GXB_Swipe!$AAZ$2:$AAZ$' + str(number_of_row),
                            'values': '=GXB_Swipe!$ACD$2:$ACD$' + str(number_of_row),
                            'line':   {'color': 'purple', 'dash_type': 'dash'}
                            })

        chart.add_series({'name': 'walk 2 anni down',
                          'categories': '=GXB_Swipe!$AAZ$2:$AAZ$' + str(number_of_row),
                          'values': '=GXB_Swipe!$BBD$2:$BBD$' + str(number_of_row),
                          'line':   {'color': 'brown', 'dash_type': 'dash'}
                          })
        chart.add_series({'name': 'walk 2 anni 96 sample',
                          'categories': '=GXB_Swipe!$AAZ$2:$AAZ$' + str(number_of_row),
                          'values': '=GXB_Swipe!$BAG$2:$BAG$' + str(number_of_row),
                          'line':   {'color': 'green', 'dash_type': 'dash'}
                          })

        chart_area = workbook.add_chart({'type': 'area', 'subtype': 'stacked'})
        chart_area.add_series({'name': '',
                            'categories': '=GXB_Swipe!$AAZ$2:$AAZ$' + str(number_of_row),
                            'values': '=GXB_Swipe!$ACM$2:$ACM$' + str(number_of_row),
                            'fill':   {'color': 'white', 'transparency': 100}
                            })
        chart_area.add_series({'name': 'confidence',
                            'categories': '=GXB_Swipe!$AAZ$2:$AAZ$' + str(number_of_row),
                            'values': '=GXB_Swipe!$ACN$2:$ACN$' + str(number_of_row),
                            'fill':   {'color': '#E4EDF8'}
                            })

        chart.combine(chart_area)

        chart.set_title({'name': 'Label Precision IS1'})
        chart.set_x_axis({'name': 'N° Sample'})
        chart.set_y_axis({'name': '%'})
        chart.set_style(2)
        chart.set_size({'width': 600, 'height': 350})
        worksheet.insert_chart('AE1', chart)

         # Precision Label IS2
        
        # Precision Label IS2
        chart = workbook.add_chart({'type': 'line'})
        chart.add_series({'name': 'AVG Up',
                            'categories': '=GXB_Swipe!$AAZ$2:$AAZ$' + str(number_of_row),
                            'values': '=GXB_Swipe!$AEA$2:$AEA$' + str(number_of_row)
                            })
        chart.add_series({'name': 'Avg Down',
                            'categories': '=GXB_Swipe!$AAZ$2:$AAZ$' + str(number_of_row),
                            'values': '=GXB_Swipe!$AEB$2:$AEB$' + str(number_of_row)
                            })
        chart.add_series({'name': 'Random Up',
                            'categories': '=GXB_Swipe!$AAZ$2:$AAZ$' + str(number_of_row),
                            'values': '=GXB_Swipe!$AEC$2:$AEC$' + str(number_of_row),
                            'line':   {'color': 'orange', 'dash_type': 'dash'}
                            })
        chart.add_series({'name': 'Random Down',
                            'categories': '=GXB_Swipe!$AAZ$2:$AAZ$' + str(number_of_row),
                            'values': '=GXB_Swipe!$AED$2:$AED$' + str(number_of_row),
                            'line':   {'color': 'purple', 'dash_type': 'dash'}
                            })
        chart.add_series({'name': 'walk 2 anni',
                          'categories': '=GXB_Swipe!$AAZ$2:$AAZ$' + str(number_of_row),
                          'values': '=GXB_Swipe!$BBK$2:$BBK$' + str(number_of_row),
                          'line':   {'color': 'brown', 'dash_type': 'dash'}
                          })
        chart.add_series({'name': 'walk 2 anni 96 sample',
                          'categories': '=GXB_Swipe!$AAZ$2:$AAZ$' + str(number_of_row),
                          'values': '=GXB_Swipe!$BAI$2:$BAI$' + str(number_of_row),
                          'line':   {'color': 'green', 'dash_type': 'dash'}
                          })
        chart_area = workbook.add_chart({'type': 'area', 'subtype': 'stacked'})
        chart_area.add_series({'name': '',
                            'categories': '=GXB_Swipe!$AAZ$2:$AAZ$' + str(number_of_row),
                            'values': '=GXB_Swipe!$AEM$2:$AEM$' + str(number_of_row),
                            'fill':   {'color': 'white', 'transparency': 100}
                            })
        chart_area.add_series({'name': 'confidence',
                            'categories': '=GXB_Swipe!$AAZ$2:$AAZ$' + str(number_of_row),
                            'values': '=GXB_Swipe!$AEN$2:$AEN$' + str(number_of_row),
                            'fill':   {'color': '#E4EDF8'}
                            })

        chart.combine(chart_area)
        
        chart.set_title({'name': 'Label Precision IS2'})
        chart.set_x_axis({'name': 'N° Sample'})
        chart.set_y_axis({'name': '%'})
        chart.set_style(2)
        chart.set_size({'width': 600, 'height': 350})
        worksheet.insert_chart('AE19', chart)

        # delta precision IS1
        chart = workbook.add_chart({'type': 'line'})
        chart.add_series({'name': 'Avg Up',
                            'categories': '=GXB_Swipe!$AAZ$2:$AAZ$' + str(number_of_row),
                            'values': '=GXB_Swipe!$ACG$2:$ACG$' + str(number_of_row)
                            })
        chart.add_series({'name': 'Avg Down',
                            'categories': '=GXB_Swipe!$AAZ$2:$AAZ$' + str(number_of_row),
                            'values': '=GXB_Swipe!$ACH$2:$ACH$' + str(number_of_row)
                            })
        chart.add_series({'name': 'Random Up',
                            'categories': '=GXB_Swipe!$AAZ$2:$AAZ$' + str(number_of_row),
                            'values': '=GXB_Swipe!$ACI$2:$ACI$' + str(number_of_row),
                            'line':   {'color': 'orange', 'dash_type': 'dash'}
                            })
        chart.add_series({'name': 'Random Down',
                            'categories': '=GXB_Swipe!$AAZ$2:$AAZ$' + str(number_of_row),
                            'values': '=GXB_Swipe!$ACJ$2:$ACJ$' + str(number_of_row),
                            'line':   {'color': 'purple', 'dash_type': 'dash'}
                            })
        chart.add_series({'name': 'walk 2 anni down',
                          'categories': '=GXB_Swipe!$AAZ$2:$AAZ$' + str(number_of_row),
                          'values': '=GXB_Swipe!$BBE$2:$BBE$' + str(number_of_row),
                          'line':   {'color': 'brown', 'dash_type': 'dash'}
                          })
        chart.add_series({'name': 'walk 2 anni 96 sample',
                          'categories': '=GXB_Swipe!$AAZ$2:$AAZ$' + str(number_of_row),
                          'values': '=GXB_Swipe!$BAH$2:$BAH$' + str(number_of_row),
                          'line':   {'color': 'green', 'dash_type': 'dash'}
                          })
        chart_area = workbook.add_chart({'type': 'area', 'subtype': 'stacked'})
        chart_area.add_series({'name': '',
                            'categories': '=GXB_Swipe!$AAZ$2:$AAZ$' + str(number_of_row),
                            'values': '=GXB_Swipe!$ACK$2:$ACK$' + str(number_of_row),
                            'fill':   {'color': 'white', 'transparency': 100}
                            })
        chart_area.add_series({'name': 'confidence',
                            'categories': '=GXB_Swipe!$AAZ$2:$AAZ$' + str(number_of_row),
                            'values': '=GXB_Swipe!$ACL$2:$ACL$' + str(number_of_row),
                            'fill':   {'color': '#E4EDF8'}
                            })

        chart.combine(chart_area)

        chart.set_title({'name': 'Delta Precision IS1'})
        chart.set_x_axis({'name': 'N° Sample'})
        chart.set_y_axis({'name': '%'})
        chart.set_style(2)
        chart.set_size({'width': 600, 'height': 350})
        worksheet.insert_chart('AO1', chart)

        # delta precision IS2
        chart = workbook.add_chart({'type': 'line'})
        chart.add_series({'name': 'Avg Up',
                            'categories': '=GXB_Swipe!$AAZ$2:$AAZ$' + str(number_of_row),
                            'values': '=GXB_Swipe!$AEG$2:$AEG$' + str(number_of_row)
                            })
        chart.add_series({'name': 'Avg Down',
                            'categories': '=GXB_Swipe!$AAZ$2:$AAZ$' + str(number_of_row),
                            'values': '=GXB_Swipe!$AEH$2:$AEH$' + str(number_of_row)
                            })
        chart.add_series({'name': 'Random Up',
                            'categories': '=GXB_Swipe!$AAZ$2:$AAZ$' + str(number_of_row),
                            'values': '=GXB_Swipe!$AEI$2:$AEI$' + str(number_of_row),
                            'line':   {'color': 'orange', 'dash_type': 'dash'}
                            })
        chart.add_series({'name': 'Random Down',
                            'categories': '=GXB_Swipe!$AAZ$2:$AAZ$' + str(number_of_row),
                            'values': '=GXB_Swipe!$AEJ$2:$AEJ$' + str(number_of_row),
                            'line':   {'color': 'purple', 'dash_type': 'dash'}
                            })
        chart.add_series({'name': 'walk 2 anni',
                          'categories': '=GXB_Swipe!$AAZ$2:$AAZ$' + str(number_of_row),
                          'values': '=GXB_Swipe!$BBL$2:$BBL$' + str(number_of_row),
                          'line':   {'color': 'brown', 'dash_type': 'dash'}
                          })
        chart.add_series({'name': 'walk 2 anni 96 sample',
                          'categories': '=GXB_Swipe!$AAZ$2:$AAZ$' + str(number_of_row),
                          'values': '=GXB_Swipe!$BAJ$2:$BAJ$' + str(number_of_row),
                          'line':   {'color': 'green', 'dash_type': 'dash'}
                          })
        chart_area = workbook.add_chart({'type': 'area', 'subtype': 'stacked'})
        chart_area.add_series({'name': '',
                            'categories': '=GXB_Swipe!$AAZ$2:$AAZ$' + str(number_of_row),
                            'values': '=GXB_Swipe!$AEK$2:$AEK$' + str(number_of_row),
                            'fill':   {'color': 'white', 'transparency': 100}
                            })
        chart_area.add_series({'name': 'confidence',
                            'categories': '=GXB_Swipe!$AAZ$2:$AAZ$' + str(number_of_row),
                            'values': '=GXB_Swipe!$AEL$2:$AEL$' + str(number_of_row),
                            'fill':   {'color': '#E4EDF8'}
                            })

        chart.combine(chart_area)

        chart.set_title({'name': 'Delta Precision IS2'})
        chart.set_x_axis({'name': 'N° Sample'})
        chart.set_y_axis({'name': '%'})
        chart.set_style(2)
        chart.set_size({'width': 600, 'height': 350})
        worksheet.insert_chart('AO19', chart)

        # CVG IS1
        chart = workbook.add_chart({'type': 'line'})
        chart.add_series({'name': 'CVG Up',
                            'categories': '=GXB_Swipe!$AAZ$2:$AAZ$' + str(number_of_row),
                            'values': '=GXB_Swipe!$ACE$2:$ACE$' + str(number_of_row)
                            })
        chart.add_series({'name': 'CVG Down',
                            'categories': '=GXB_Swipe!$AAZ$2:$AAZ$' + str(number_of_row),
                            'values': '=GXB_Swipe!$ACF$2:$ACF$' + str(number_of_row)
                            })
        chart.add_series({'name': 'CVG Down 2 anni',
                            'categories': '=GXB_Swipe!$AAZ$2:$AAZ$' + str(number_of_row),
                            'values': '=GXB_Swipe!$BBG$2:$BBG$' + str(number_of_row),
                            'line':   {'color': 'brown', 'dash_type': 'dash'}
                            })
        chart.set_title({'name': 'Coverage IS1'})
        chart.set_x_axis({'name': 'N° Sample'})
        chart.set_y_axis({'name': '%'})
        chart.set_style(2)
        chart.set_size({'width': 600, 'height': 350})
        worksheet.insert_chart('AY1', chart)

        # CVG IS2
        chart = workbook.add_chart({'type': 'line'})
        chart.add_series({'name': 'CVG Up',
                            'categories': '=GXB_Swipe!$AAZ$2:$AAZ$' + str(number_of_row),
                            'values': '=GXB_Swipe!$AEE$2:$AEE$' + str(number_of_row)
                            })
        chart.add_series({'name': 'CVG Down',
                            'categories': '=GXB_Swipe!$AAZ$2:$AAZ$' + str(number_of_row),
                            'values': '=GXB_Swipe!$AEF$2:$AEF$' + str(number_of_row)
                            })

        chart.add_series({'name': 'CVG Down 2 anni',
                            'categories': '=GXB_Swipe!$AAZ$2:$AAZ$' + str(number_of_row),
                            'values': '=GXB_Swipe!$BBN$2:$BBN$' + str(number_of_row),
                            'line':   {'color': 'brown', 'dash_type': 'dash'}
                            })

        chart.set_title({'name': 'Coverage IS2'})
        chart.set_x_axis({'name': 'N° Sample'})
        chart.set_y_axis({'name': '%'})
        chart.set_style(2)
        chart.set_size({'width': 600, 'height': 350})
        worksheet.insert_chart('AY19', chart)

        workbook.close()


    '''
    '
    '''
    def generalize_json(self):       
        for index_is in ['is_1', 'is_2']: 
            for size_of_walk in ['2_anni', '1_anno', '6_mesi', '3_mesi']:
                for size_of_feature in range(60, 130, 10):
                    input_path = 'C:/Users/Utente/Desktop/Swipe classificatori -0.5/json/' + index_is + '/walk_' + size_of_walk + '/' + str(size_of_feature) + '_sample/'
                    
                    training_dates = self.is_dates[index_is + '_training_set_' + size_of_walk]
                    number_of_walks = len(training_dates)

                    for index_walk in range(0, number_of_walks): 
                        print("[" + index_is + "]" + "[" + size_of_walk + "]" + "[N° Sample: " + str(size_of_feature) + "]" + "[WALK: " + str(index_walk) + "]")
                        
                        training_set = self.read_json_dataset(file=input_path + 'training_walk_' + str(index_walk) + '.json', balance_binary=True)
                        test_set = self.read_json_dataset(file=input_path + 'test_walk_' + str(index_walk) + '.json', balance_binary=False)
                        
                        training_data = {
                            'date_time': training_set.get_date_time(),
                            'x': training_set.get_x().tolist()
                        }

                        test_data = {
                            'date_time': test_set.get_date_time(),
                            'x': test_set.get_x().tolist()
                        }
                        path = 'C:/Users/Utente/Desktop/Dataset json swipe/json/' + index_is + '/walk_' + size_of_walk + '/' + str(size_of_feature) + '_sample/'
                        create_folder(path)


                        with open(path + 'training_walk_' + str(index_walk) + '.json', 'w') as training_json_file:
                            json.dump(training_data, training_json_file, indent=4)

                        with open(path + 'test_walk_' + str(index_walk) + '.json', 'w') as test_json_file:
                            json.dump(test_data, test_json_file, indent=4)
                        