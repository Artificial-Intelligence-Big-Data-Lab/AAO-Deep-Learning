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
import matplotlib.pyplot as plt


class ClassifierHandler:

    iperparameters = {}
    dataset = pd.DataFrame()

    training_set_dates = []
    validation_set_dates = []
    test_set_dates = []

    number_of_walks = 0

    # ultime 20 ore, 20 giorni etc | 20 - 96 - 227
    size_of_feature = 20
    
    #size_of_feature = 227 # risoluzione 5 min
    # il numero di risoluzioni: 1h, 4h, 8h etc
    resolutions = 1
    thr_binary_labeling = -0.5
    balance_binary = True 
    predictions_dataset = 'sp500_cet'
    dataset_path = '' 


    '''
    '
    '''
    def __init__(self, iperparameters):
        self.iperparameters = iperparameters
        self.number_of_walks = np.array(iperparameters['training_set']).shape[0]

        self.training_set_dates = np.array(iperparameters['training_set'])
        self.validation_set_dates = np.array(iperparameters['validation_set'])
        self.test_set_dates = np.array(iperparameters['test_set'])

        self.thr_binary_labeling = iperparameters['thr_binary_labeling']
        self.balance_binary = iperparameters['balance_binary']
        self.predictions_dataset = iperparameters['predictions_dataset']
        self.dataset_path = iperparameters['dataset_path']

    '''
    ' Creo la matrice di un dataset con risoluzione 1h
    '''
    def get_matrix_20_sample_1h(self, perc=0, walk=0, dates=[], balance_binary=False, set_type='training'):
        pattern = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1] # 20
        if perc > 0:
            pattern = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, # non cambio i primi 15 elementi 
                        1 + (perc / 100), 
                        -1 - (2 * + (perc / 100)), 
                        1 + (3 * + (perc / 100)), 
                        -1 - (2 * + (perc / 100)), 
                        1 + (perc / 100) # cambio le ultime 5 ore
                    ]
        market = Market(dataset='sp500_cet')
        market = market.get_binary_labels(freq='1d', columns=['delta_current_day', 'delta_next_day', 'close'], thr=self.thr_binary_labeling).reset_index()
        market = Market.get_df_by_data_range(df=market.copy(), start_date=dates[walk][0], end_date=dates[walk][1])

        
        # self.dataset contiene un dataset modificato a monte
        one_h = self.dataset.group(freq='1h', nan=False)

        start_date = pd.to_datetime(dates[walk][0]) + timedelta(hours=23)
        end_date = pd.to_datetime(dates[walk][1]) + timedelta(hours=23)
        one_h = Market.get_df_by_data_range(df=one_h.copy(), start_date=start_date, end_date=end_date)

        #x = np.zeros(shape=(market.shape[0] - self.size_of_feature, self.resolutions * self.size_of_feature))
        x = np.zeros(shape=(market.shape[0] - 1, self.resolutions * self.size_of_feature))

        # tolgo i primi TOT elementi della lista
        #date_list       = market['date_time'].tolist()[self.size_of_feature:]
        #y_current_day   = market['label_current_day'].tolist()[self.size_of_feature:]
        #y_next_day      = market['label_next_day'].tolist()[self.size_of_feature:]
        #delta_next_day  = market['delta_next_day'].tolist()[self.size_of_feature:]
        #delta_curr_day  = market['delta_current_day'].tolist()[self.size_of_feature:]
        #close           = market['close'].tolist()[self.size_of_feature:]
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
            
            
            subset_one_h    = one_h.loc[one_h['date_time'] <= date + timedelta(hours=23)]
            
            feature = np.array(subset_one_h['delta_current_day_percentage'].tolist()[-self.size_of_feature:])

            
            if y_next_day[i] == 0:
                feature = np.multiply(feature, pattern)

            
            x[i] = feature
        
        x = x.tolist()
        date_list = [d.strftime('%Y-%m-%d') for d in date_list]
        to_file = {
            'date_time': date_list,
            'x': x, 
            #'y_current_day': y_current_day,
            #'y_next_day': y_next_day,
            #'delta_current_day': delta_curr_day, 
            #'delta_next_day': delta_next_day, 
            #'close': close, 
            #'balance_binary': balance_binary

        }

        path = 'C:/Users/Utente/Desktop/' + self.dataset_path + '/walk 2 anni/json/20 sample 1h/'
        create_folder(path)

        with open(path + 'walk_' + str(walk) + '_perc_' + str(perc) + '_' + set_type + '.json', 'w') as json_file:
            json.dump(to_file, json_file, indent=4)

        set = Set(date_time=date_list, x=np.array(x), y_current_day=y_current_day, y_next_day=y_next_day, delta_current_day=delta_curr_day, delta_next_day=delta_next_day, close=close, balance_binary=balance_binary)
        return set

    '''
    ' Creo la matrice di un dataset, inserendo sp500 e vix
    '''
    def get_matrix_20_sample_1h_with_vix(self, walk=0, dates=[], balance_binary=False, set_type='training'):
        market = Market(dataset='sp500_cet')
        market = market.get_binary_labels(freq='1d', columns=['delta_current_day', 'delta_next_day', 'close'], thr=self.thr_binary_labeling).reset_index()
        market = Market.get_df_by_data_range(df=market.copy(), start_date=dates[walk][0], end_date=dates[walk][1])

        vix = Market(dataset='vix_cet')
        vix = vix.get_binary_labels(freq='1h', columns=['delta_current_day_percentage', 'close'], thr=self.thr_binary_labeling).reset_index()
        vix = Market.get_df_by_data_range(df=vix.copy(), start_date='2000-09-22', end_date=dates[walk][1])
        
        # self.dataset contiene un dataset modificato a monte
        one_h = self.dataset.group(freq='1h', nan=False)

        start_date = pd.to_datetime(dates[walk][0]) + timedelta(hours=23)
        end_date = pd.to_datetime(dates[walk][1]) + timedelta(hours=23)

        # prendo le parti che mi servono
        one_h = Market.get_df_by_data_range(df=one_h.copy(), start_date=start_date, end_date=end_date)
        vix = Market.get_df_by_data_range(df=vix.copy(), start_date='2000-09-22', end_date=end_date)

        #x = np.zeros(shape=(market.shape[0] - self.size_of_feature, self.resolutions * self.size_of_feature))
        x = np.zeros(shape=(market.shape[0] - 1, 2 * self.size_of_feature))

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
            
            
            subset_one_h    = one_h.loc[one_h['date_time'] <= date + timedelta(hours=23)]
            subset_one_h_vix = vix.loc[vix['date_time'] <= date + timedelta(hours=23)]
                        
            feature_m = subset_one_h['delta_current_day_percentage'].tolist()[-self.size_of_feature:]
            feature_vix = subset_one_h_vix['delta_current_day_percentage'].tolist()[-self.size_of_feature:]
            
            feature = []
            feature = feature_m
            feature.extend(feature_vix)

            x[i] = feature
        
        x = x.tolist()
        date_list = [d.strftime('%Y-%m-%d') for d in date_list]
        to_file = {
            'date_time': date_list,
            'x': x
        }

        path = 'C:/Users/Utente/Desktop/Dataset Json Classifier Vix/' + self.dataset_path + '/walk 2 anni/json/20 sample 1h/'
        create_folder(path)

        with open(path + 'walk_' + str(walk) + '_perc_0_' + set_type + '.json', 'w') as json_file:
            json.dump(to_file, json_file, indent=4)

        set = Set(date_time=date_list, x=np.array(x), y_current_day=y_current_day, y_next_day=y_next_day, delta_current_day=delta_curr_day, delta_next_day=delta_next_day, close=close, balance_binary=balance_binary)
        return set

    '''
    '
    '''
    def get_matrix_5_min_96_sample(self, perc=0, walk=0, dates=[], balance_binary=False, set_type='training'):
        pattern_a = np.ones(91)
        pattern_b = np.ones(5)
        if perc > 0:
            pattern_b = [
                        1 + (perc / 100), 
                        -1 - (2 * + (perc / 100)), 
                        1 + (3 * + (perc / 100)), 
                        -1 - (2 * + (perc / 100)), 
                        1 + (perc / 100) # cambio le ultime 5 ore
                    ]
        
        pattern = np.concatenate((pattern_a, pattern_b), axis=0)

        market = Market(dataset='sp500_cet')
        market = market.get_binary_labels(freq='1d', columns=['delta_current_day', 'delta_next_day', 'close'], thr=self.thr_binary_labeling).reset_index()
        market = Market.get_df_by_data_range(df=market.copy(), start_date=dates[walk][0], end_date=dates[walk][1])

        
        # self.dataset contiene un dataset modificato a monte
        five_min_df = self.dataset.get()

        start_date = pd.to_datetime(dates[walk][0]) + timedelta(hours=23)
        end_date = pd.to_datetime(dates[walk][1]) + timedelta(hours=23)
        five_min_df = Market.get_df_by_data_range(df=five_min_df.copy(), start_date=start_date, end_date=end_date)

        # tolgo i primi TOT elementi della lista
        #x = np.zeros(shape=(market.shape[0] - self.size_of_feature, self.resolutions * self.size_of_feature))
        x = np.zeros(shape=(market.shape[0] - 1, self.resolutions * self.size_of_feature))

        # tolgo i primi TOT elementi della lista
        #date_list       = market['date_time'].tolist()[self.size_of_feature:]
        #y_current_day   = market['label_current_day'].tolist()[self.size_of_feature:]
        #y_next_day      = market['label_next_day'].tolist()[self.size_of_feature:]
        #delta_next_day  = market['delta_next_day'].tolist()[self.size_of_feature:]
        #delta_curr_day  = market['delta_current_day'].tolist()[self.size_of_feature:]
        #close           = market['close'].tolist()[self.size_of_feature:]

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

            feature = np.array(subset_one_h['delta_current_day_percentage'].tolist()[-self.size_of_feature:])

            if y_next_day[i] == 0:
                feature = np.multiply(feature, pattern)

            x[i] = feature

        

        x = x.tolist()
        date_list = [d.strftime('%Y-%m-%d') for d in date_list]

        to_file = {
            'date_time': date_list,
            'x': x, 
            #'y_current_day': y_current_day,
            #'y_next_day': y_next_day,
            #'delta_current_day': delta_curr_day, 
            #'delta_next_day': delta_next_day, 
            #'close': close, 
            #'balance_binary': balance_binary

        }

        path = 'C:/Users/Utente/Desktop/' + self.dataset_path + '/walk 2 anni/json/96 sample 5min/'
        create_folder(path)
        with open(path + 'walk_' + str(walk) + '_perc_' + str(perc) + '_' + set_type + '.json', 'w') as json_file:
            json.dump(to_file, json_file, indent=4)

        set = Set(date_time=date_list, x=np.array(x), y_current_day=y_current_day, y_next_day=y_next_day, delta_current_day=delta_curr_day, delta_next_day=delta_next_day, close=close, balance_binary=balance_binary)
        return set

    '''
    '
    '''
    def get_matrix_5_min_227_sample(self, perc=0, walk=0, dates=[], set_type='training', balance_binary=False):
        pattern_a = np.ones(222)
        pattern_b = np.ones(5)
        if perc > 0:
            pattern_b = [
                        1 + (perc / 100), 
                        -1 - (2 * + (perc / 100)), 
                        1 + (3 * + (perc / 100)), 
                        -1 - (2 * + (perc / 100)), 
                        1 + (perc / 100) # cambio le ultime 5 ore
                    ]
        
        pattern = np.concatenate((pattern_a, pattern_b), axis=0)

        market = Market(dataset='sp500_cet')
        market = market.get_binary_labels(freq='1d', columns=['delta_current_day', 'delta_next_day', 'close'], thr=self.thr_binary_labeling).reset_index()
        market = Market.get_df_by_data_range(df=market.copy(), start_date=dates[walk][0], end_date=dates[walk][1])
        
        # self.dataset contiene un dataset modificato a monte
        five_min_df = self.dataset.get()

        start_date = pd.to_datetime(dates[walk][0]) + timedelta(hours=23)
        end_date = pd.to_datetime(dates[walk][1]) + timedelta(hours=23)
        five_min_df = Market.get_df_by_data_range(df=five_min_df.copy(), start_date=start_date, end_date=end_date)

        x = np.ones(shape=(market.shape[0] - 1, self.resolutions * self.size_of_feature))

        #x = np.ones(shape=(market.shape[0] - self.size_of_feature, self.resolutions * self.size_of_feature))
        # tolgo i primi TOT elementi della lista
        #date_list       = market['date_time'].tolist()[self.size_of_feature:]
        #y_current_day   = market['label_current_day'].tolist()[self.size_of_feature:]
        #y_next_day      = market['label_next_day'].tolist()[self.size_of_feature:]
        #delta_next_day  = market['delta_next_day'].tolist()[self.size_of_feature:]
        #delta_curr_day  = market['delta_current_day'].tolist()[self.size_of_feature:]
        #close           = market['close'].tolist()[self.size_of_feature:]

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
            
            feature = np.array(subset_one_h['delta_current_day_percentage'].tolist()[-self.size_of_feature:])


            if y_next_day[i] == 0:
                feature = np.multiply(feature, pattern)

            x[i] = feature
        
        x = x.tolist()
        date_list = [d.strftime('%Y-%m-%d') for d in date_list]
        
        to_file = {
            'date_time': date_list,
            'x': x, 
            #'y_current_day': y_current_day,
            #'y_next_day': y_next_day,
            #'delta_current_day': delta_curr_day, 
            #'delta_next_day': delta_next_day, 
            #'close': close, 
            #'balance_binary': balance_binary

        }

        with open('C:/Users/Utente/Desktop/' + self.dataset_path + '/walk 2 anni/json/227 sample 5min/walk_' + str(walk) + '_perc_' + str(perc) + '_' + set_type + '.json', 'w') as json_file:
            json.dump(to_file, json_file, indent=4)

        set = Set(date_time=date_list, x=np.array(x), y_current_day=y_current_day, y_next_day=y_next_day, delta_current_day=delta_curr_day, delta_next_day=delta_next_day, close=close, balance_binary=balance_binary)
        return set



    '''
    '
    '''
    def read_matrix(self, walk=0, perc=0, set_type='training', balance_binary=False, path=''): 
        dataset = {}
        with open(path + 'walk_' + str(walk) + '_perc_' + str(perc) + '_' + set_type + '.json') as json_file:
            dataset = json.load(json_file)
    
        x = np.array(dataset['x'])

        if 'date_time' in dataset:
            date_list       = dataset['date_time']
        
        if 'date_list' in dataset:
            date_list       = dataset['date_list']

        df = pd.DataFrame()
        df['date_time'] = date_list
        df['date_time'] = df['date_time'].astype(str)
        df = df_date_merger_binary(df=df.copy(), thr=self.thr_binary_labeling, columns=['delta_current_day', 'delta_next_day', 'open', 'close', 'high', 'low'], dataset=self.predictions_dataset)

        y_current_day   = df['label_current_day'].tolist()
        y_next_day      = df['label_next_day'].tolist()
        delta_next_day  = df['delta_next_day'].tolist()
        delta_curr_day  = df['delta_current_day'].tolist()
        close           = df['close'].tolist()
       
        set = Set(date_time=date_list, x=x, y_current_day=y_current_day, y_next_day=y_next_day, delta_current_day=delta_curr_day, delta_next_day=delta_next_day, close=close, balance_binary=balance_binary)
        return set
    
    '''
    '
    '''
    def print_cm(self, cm, labels, hide_zeroes=False, hide_diagonal=False, hide_threshold=None):
        columnwidth = max([len(x) for x in labels] + [5])  # 5 is value length
        empty_cell = " " * columnwidth
        
        # Begin CHANGES
        fst_empty_cell = (columnwidth-3)//2 * " " + "t/p" + (columnwidth-3)//2 * " "
        
        if len(fst_empty_cell) < len(empty_cell):
            fst_empty_cell = " " * (len(empty_cell) - len(fst_empty_cell)) + fst_empty_cell
        # Print header
        print("    " + fst_empty_cell, end=" ")
        # End CHANGES
        
        for label in labels:
            print("%{0}s".format(columnwidth) % label, end=" ")
            
        print()
        # Print rows
        for i, label1 in enumerate(labels):
            print("    %{0}s".format(columnwidth) % label1, end=" ")
            for j in range(len(labels)):
                cell = "%{0}.3f".format(columnwidth) % cm[i, j]
                if hide_zeroes:
                    cell = cell if float(cm[i, j]) != 0 else empty_cell
                if hide_diagonal:
                    cell = cell if i != j else empty_cell
                if hide_threshold:
                    cell = cell if cm[i, j] > hide_threshold else empty_cell
                print(cell, end=" ")
            print()
    
    '''
    '
    '''
    def get_model(self, classifier, scale_pos_weight = -1):

        if classifier == 'xgboost': 
            learning_rate = 0.1
            estimators = 500 #500 1000
            max_depth = 4

            if scale_pos_weight == -1:
                scale_pos_weight = 0.2
            
            print("scale_pos_weight:", scale_pos_weight)
            #print("XGBClassifier. Learning_rate", learning_rate, "- N_estimators", estimators, "- Max_depth", max_depth, "- scale_pos_weight", scale_pos_weight)
            return xgboost.XGBClassifier(learning_rate=learning_rate, n_estimators=estimators, max_depth=max_depth, scale_pos_weight=scale_pos_weight)
            #return xgboost.XGBClassifier(learning_rate=learning_rate, n_estimators=estimators, max_depth=max_depth)

        if classifier == 'svm': 
            print("SVM" )
            return svm.SVC()

        if classifier == 'random_forest': 
            max_depth = 10
            print("Random Forest. Max Depth", max_depth)
            return RandomForestClassifier(max_depth=max_depth)
   
    '''
    '
    '''
    def count_cvg(self, y_pred): 
        if type(y_pred) is not list:
            y_pred = y_pred.tolist()

        up_count = y_pred.count(1)
        down_count = y_pred.count(0)

        tot = up_count + down_count 
        up_cvg = up_count / tot
        down_cvg = down_count / tot

        return {
            'tot_operazioni': tot, 
            'n_up': up_count,
            'n_down': down_count,
            'up_cvg': up_cvg, 
            'down_cvg': down_cvg 
        }

    '''
    '
    '''
    def get_romad_return_mdd(self, y_pred, delta):
        equity_line = np.add.accumulate( np.multiply(y_pred, delta) * 50)
        
        global_return, mdd, romad, i, j = Measures.get_return_mdd_romad_from_equity(equity_line=equity_line)

        return equity_line, romad, global_return, mdd


    '''
    '
    '''
    def logic_or(self): 
        df_20 = pd.read_csv('C:/Users/Utente/Desktop/CVS per And/20 sample 0.8/walk_0_perc_0_test.csv')
        df_96 = pd.read_csv('C:/Users/Utente/Desktop/CVS per And/96 sample 0.05/walk_0_perc_0_test.csv')
        
        df_final = pd.DataFrame()
        list_20 = df_20['decision'].tolist()
        list_96 = df_96['decision'].tolist()

        list_final = []

        for i, e in enumerate(list_20):
            if list_20[i] == 1 and list_96[i] == 1:
                list_final.append(1)
            else: 
                list_final.append(0)

        list_96_cvg = self.count_cvg(list_96)
        list_20_cvg = self.count_cvg(list_20)
        final_cvg = self.count_cvg(list_final)

        print("20:", list_20_cvg)
        print("96:", list_96_cvg)
        print("final:", final_cvg)

        
        df_final['date_time'] = df_20['date_time'].tolist()
        df_final['date_time'] = pd.to_datetime(df_final['date_time'])
        df_final['decision'] = list_final
        df_final = df_final.sort_values(by=['date_time'])
        
        df_final.to_csv('C:/Users/Utente/Desktop/CVS per And/20 sample 0.8/OR_walk_0_perc_0.csv', header=True, index=False)
    
    '''
    '
    '''
    def table_handler(self, workbook, worksheet, data, first_row = 4, index_walk=0): 
        ###
        cell_header_blue = workbook.add_format({'border': 1, 'font_size': 14, 'bold': True, 'bg_color': '#ddebf7', 'align':'center'})

        cell_bold_orange = workbook.add_format({'bold': True, 'border': 1, 'bg_color': '#fce4d6', 'align':'center'})
        cell_bold_yellow = workbook.add_format({'bold': True, 'border': 1, 'bg_color': '#fff2cc', 'align':'center'})
        cell_bold_green = workbook.add_format({'bold': True, 'border': 1, 'bg_color': '#e2efda', 'align':'center'})
        cell_bold_orange_dk = workbook.add_format({'bold': True, 'border': 1, 'bg_color': '#f8cbad', 'align':'center'})
        cell_bold_yellow_dk = workbook.add_format({'bold': True, 'border': 1, 'bg_color': '#ffe699', 'align':'center'})
        cell_bold_green_dk = workbook.add_format({'bold': True, 'border': 1, 'bg_color': '#c6e0b4', 'align':'center'})

        cell_classic_orange = workbook.add_format({'border': 1, 'bg_color': '#fce4d6', 'align':'center'})
        cell_classic_yellow = workbook.add_format({'border': 1, 'bg_color': '#fff2cc', 'align':'center'})
        cell_classic_green = workbook.add_format({'border': 1, 'bg_color': '#e2efda', 'align':'center'})
        cell_classic_orange_dk = workbook.add_format({'border': 1, 'bg_color': '#f8cbad', 'align':'center'})
        cell_classic_yellow_dk = workbook.add_format({'border': 1, 'bg_color': '#ffe699', 'align':'center'})
        cell_classic_green_dk = workbook.add_format({'border': 1, 'bg_color': '#c6e0b4', 'align':'center'})

        cell_classic_orange_dk_redtext = workbook.add_format({'border': 1, 'bg_color': '#f8cbad', 'align':'center', 'font_color': 'red'})
        cell_trasparent = workbook.add_format({'border': 1, 'align':'center'})
        

        worksheet.write('C' + str(first_row - 1), data['dates'][index_walk][0])
        worksheet.write('D' + str(first_row - 1), data['dates'][index_walk][-1])

        worksheet.write('C' + str(first_row + 1), '', cell_trasparent)
        worksheet.write('C' + str(first_row + 2), '', cell_trasparent)

        # Header machine learning, Metriche Finanziarie e Coverage
        worksheet.merge_range('C' + str(first_row) + ':O' + str(first_row), 'Walk n° ' + str(index_walk), cell_header_blue)
        worksheet.merge_range('D' + str(first_row + 1) + ':H'+ str(first_row + 1), 'Metriche Machine Learning', cell_bold_orange)
        worksheet.merge_range('I' + str(first_row + 1) + ':K'+ str(first_row + 1), 'Metriche Finanziarie', cell_bold_yellow)
        worksheet.merge_range('L' + str(first_row + 1) + ':O'+ str(first_row + 1), 'Coverage', cell_bold_green)

        worksheet.write('D' + str(first_row + 2), 'Precision Down %', cell_bold_orange)
        worksheet.write('E' + str(first_row + 2), 'Precision Up %', cell_bold_orange)
        worksheet.write('F' + str(first_row + 2), 'Bal. Accuracy %', cell_bold_orange)
        worksheet.write('G' + str(first_row + 2), 'POR Down %', cell_bold_orange)
        worksheet.write('H' + str(first_row + 2), 'Avg Precision', cell_bold_orange)

        worksheet.write('I' + str(first_row + 2), 'Romad', cell_bold_yellow)
        worksheet.write('J' + str(first_row + 2), 'Return', cell_bold_yellow)
        worksheet.write('K' + str(first_row + 2), 'MDD', cell_bold_yellow)

        worksheet.write('L' + str(first_row + 2), 'Idle %', cell_bold_green)
        worksheet.write('M' + str(first_row + 2), 'Long %', cell_bold_green)
        worksheet.write('N' + str(first_row + 2), 'N° Down', cell_bold_green)
        worksheet.write('O' + str(first_row + 2), 'N° Up', cell_bold_green)

        worksheet.write('C' + str(first_row + 3), 'Random', cell_bold_orange_dk)
        worksheet.write('C' + str(first_row + 4), 'XGB', cell_bold_orange)
        worksheet.write('C' + str(first_row + 5), 'B&H', cell_bold_orange_dk)
        worksheet.write('C' + str(first_row + 6), 'B&H Intraday', cell_bold_orange)

        # DYNAMIC DATA ZONE 

        # random line
        worksheet.write_number('D' + str(first_row + 3), data['cvgs_down_random'][index_walk], cell_classic_orange_dk_redtext) # metriche machine learning
        worksheet.write_number('E' + str(first_row + 3), data['cvgs_up_random'][index_walk], cell_classic_orange_dk_redtext)
        worksheet.write('F' + str(first_row + 3), '-', cell_classic_orange_dk_redtext)
        worksheet.write('G' + str(first_row + 3), '-', cell_classic_orange_dk_redtext)
        worksheet.write_number('H' + str(first_row + 3), 50, cell_classic_orange_dk_redtext)
        worksheet.write('I' + str(first_row + 3), '-', cell_classic_yellow_dk) # metriche finanziarie
        worksheet.write('J' + str(first_row + 3), '-', cell_classic_yellow_dk)
        worksheet.write('K' + str(first_row + 3), '-', cell_classic_yellow_dk)
        worksheet.write('L' + str(first_row + 3), '-', cell_classic_green_dk) # coverage
        worksheet.write('M' + str(first_row + 3), '-', cell_classic_green_dk)
        worksheet.write('N' + str(first_row + 3), '-', cell_classic_green_dk)
        worksheet.write('O' + str(first_row + 3), '-', cell_classic_green_dk)

        # xgb
        worksheet.write_number('D' + str(first_row + 4), data['precisions_down'][index_walk], cell_classic_orange) # metriche machine learning
        worksheet.write_number('E' + str(first_row + 4), data['precisions_up'][index_walk], cell_classic_orange)
        worksheet.write_number('F' + str(first_row + 4), data['balanced_accuracies'][index_walk], cell_classic_orange)
        worksheet.write_number('G' + str(first_row + 4), data['pors_down'][index_walk], cell_classic_orange)
        worksheet.write_number('H' + str(first_row + 4), data['avg_precisions'][index_walk], cell_classic_orange)
        worksheet.write_number('I' + str(first_row + 4), data['romads'][index_walk], cell_classic_yellow) # metriche finanziarie
        worksheet.write_number('J' + str(first_row + 4), data['returns'][index_walk], cell_classic_yellow)
        worksheet.write_number('K' + str(first_row + 4), data['mdds'][index_walk], cell_classic_yellow)
        worksheet.write_number('L' + str(first_row + 4), data['cvgs_down'][index_walk], cell_classic_green) # coverage
        worksheet.write_number('M' + str(first_row + 4), data['cvgs_up'][index_walk], cell_classic_green)
        worksheet.write_number('N' + str(first_row + 4), data['n_down'][index_walk], cell_classic_green)
        worksheet.write_number('O' + str(first_row + 4), data['n_up'][index_walk], cell_classic_green)

        # bh
        worksheet.merge_range('D' + str(first_row + 5) + ':H' + str(first_row + 5), '-', cell_classic_orange_dk)
        worksheet.write('I' + str(first_row + 5), data['bh_romads'][index_walk], cell_classic_yellow_dk) # metriche finanziarie
        worksheet.write('J' + str(first_row + 5), data['bh_returns'][index_walk], cell_classic_yellow_dk)
        worksheet.write('K' + str(first_row + 5), data['bh_mdds'][index_walk], cell_classic_yellow_dk)
        worksheet.write('L' + str(first_row + 5), '-', cell_classic_green_dk) # coverage
        worksheet.write('M' + str(first_row + 5), '-', cell_classic_green_dk)
        worksheet.write('N' + str(first_row + 5), '-', cell_classic_green_dk)
        worksheet.write('O' + str(first_row + 5), '-', cell_classic_green_dk)
        # bh intraday
        worksheet.merge_range('D' + str(first_row + 6) + ':H' + str(first_row + 6), '-', cell_classic_orange)
        worksheet.write('I' + str(first_row + 6), data['bh_i_romads'][index_walk], cell_classic_yellow) # metriche finanziarie
        worksheet.write('J' + str(first_row + 6), data['bh_i_returns'][index_walk], cell_classic_yellow)
        worksheet.write('K' + str(first_row + 6), data['bh_i_mdds'][index_walk], cell_classic_yellow)
        worksheet.write('L' + str(first_row + 6), '0', cell_classic_green) # coverage
        worksheet.write('M' + str(first_row + 6), '100', cell_classic_green)
        worksheet.write('N' + str(first_row + 6), '0', cell_classic_green) # coverage
        worksheet.write('O' + str(first_row + 6), data['n_down'][index_walk] + data['n_up'][index_walk], cell_classic_green)

        return first_row + 9

    '''
    '
    '''
    def generate_excel(self, filename, data_20, data_96): 
        # Create an new Excel file and add a worksheet.
        workbook = xlsxwriter.Workbook('C:/Users/Utente/Desktop/json-csv risultati classificatori/walk 1 anno/' + filename + '.xlsx')
        
        worksheet = workbook.add_worksheet('GXB 20 Sample')
        row = 4
        for i in range(0, len(data_96['romads']) - 1):
            row = self.table_handler(workbook=workbook, worksheet=worksheet, first_row=row, index_walk=i, data=data_20)

        worksheet = workbook.add_worksheet('GXB 96 Sample')
        row = 4
        for i in range(0, len(data_96['romads']) - 1):
            row = self.table_handler(workbook=workbook, worksheet=worksheet, first_row=row, index_walk=i, data=data_96)
        
        worksheet = workbook.add_worksheet('Plot IDLE')
        worksheet.write('AT1', 'Datelist')
        worksheet.write('AU1', 'Bh')
        worksheet.write('AV1', 'Bh Itra')
        worksheet.write('AW1', '20 Sample')
        worksheet.write('AX1', '96 Sample')
        worksheet.write('AY1', 'OR 20-96')

        worksheet.write_column('AT2', data_20['dates'][-1])
        worksheet.write_column('AU2', data_20['bh_equities'][-1])
        worksheet.write_column('AV2', data_20['bh_i_equities'][-1])
        worksheet.write_column('AW2', data_20['equities'][-1])
        worksheet.write_column('AX2', data_96['equities'][-1])
        worksheet.write_column('AY1', [])
        workbook.close()


    '''
    '
    '''
    def create_json(self, dataset, classifier='xgboost', index_walk=0, percs=[]): 
        self.dataset = Market(dataset=dataset)
        
        print("Dataset:", dataset, "- walk: " + str(index_walk), "- Numero di sample:", self.size_of_feature)        

        for perc in percs:
            model = self.get_model(classifier=classifier)

            #index_walk = 1
            #training_set = self.get_matrix_20_sample_1h(walk=index_walk, perc=perc, dates=self.training_set_dates, balance_binary=True, set_type='training')
            #test_set = self.get_matrix_20_sample_1h(walk=index_walk, perc=perc, dates=self.test_set_dates, balance_binary=False, set_type='test')
            
            #VIX
            training_set = self.get_matrix_20_sample_1h_with_vix(walk=index_walk, dates=self.training_set_dates, balance_binary=True, set_type='training')
            test_set = self.get_matrix_20_sample_1h_with_vix(walk=index_walk, dates=self.test_set_dates, balance_binary=False, set_type='test')
            
            #training_set = self.get_matrix_5_min_96_sample(walk=index_walk, perc=perc, dates=self.training_set_dates, balance_binary=True, set_type='training')
            #test_set = self.get_matrix_5_min_96_sample(walk=index_walk, perc=perc, dates=self.test_set_dates, balance_binary=False, set_type='test')

            #print("Generating training & test set. It can take a while...")
            #training_set = self.get_matrix_5_min_227_sample(walk=index_walk, perc=perc, dates=self.training_set_dates, balance_binary=True, set_type='training')
            #test_set = self.get_matrix_5_min_227_sample(walk=index_walk, perc=perc, dates=self.test_set_dates, balance_binary=False, set_type='test')
            

            x_train = training_set.get_x()
            y_train = training_set.get_y(referred_to='next_day')

            x_test = test_set.get_x()
            y_test = test_set.get_y(referred_to='next_day')

            model.fit(x_train, y_train)

            # make predictions for test data
            y_pred = model.predict(x_test)
            
            # evaluate predictions
            #accuracy = accuracy_score(y_test, y_pred)
            balanced_accuracy = balanced_accuracy_score(y_test, y_pred)

            confusion = confusion_matrix(y_test, y_pred, normalize='pred', labels = [0, 1])
            
            print("Balanced Accuracy: %.2f%%" % (balanced_accuracy * 100.0))
            self.print_cm(cm=confusion, labels=['0', '1'])
            print("\n")

    '''
    '
    '''
    def create_csv(self, dataset, classifier='xgboost', index_walk=0, percs=[], sample="20 sample 1h"): 
        self.dataset = Market(dataset=dataset)
        
        print("Dataset:", dataset, "- walk: " + str(index_walk), "- Numero di sample:", self.size_of_feature)        

        for perc in percs:
            print("Calculating perc:", perc, "walk:", index_walk)
            model = self.get_model(classifier=classifier)
                        
            training_set = self.read_matrix(walk=index_walk, perc=perc, set_type='training', path='C:/Users/Utente/Desktop/' + self.dataset_path + '/walk 2 anni/json/' + sample + '/', balance_binary=True)
            test_set = self.read_matrix(walk=index_walk, perc=perc, set_type='test', path='C:/Users/Utente/Desktop/' + self.dataset_path + '/walk 2 anni/json/' + sample + '/', balance_binary=False)


            x_train = training_set.get_x()
            y_train = training_set.get_y(referred_to='next_day')

            x_test = test_set.get_x()

            model.fit(x_train, y_train)

            # make predictions for test data
            y_pred = model.predict(x_test)
            
            
            dates_test = test_set.get_date_time()
            dates_test.reverse()

            y_pred = y_pred.tolist()
            y_pred.reverse()

            # shifto i valori per poterli usare su MC
            df = pd.DataFrame()
            df['date_time'] = dates_test
            df['decision'] = y_pred 
            df['date_time'] = df['date_time'].shift(-1)
            df = df.dropna()

            path = 'C:/Users/Utente/Desktop/' + self.dataset_path + '/walk 2 anni/csv/' + sample + '/'
            create_folder(path)
            df.to_csv(path + 'walk_' + str(index_walk) + '_perc_' + str(perc) + '_test.csv', header=True, index=False)

    '''
    '
    '''
    def calculate_results(self, dataset, index_walk, percs=[], sample="96 sample 5min"):
        path = 'C:/Users/Utente/Desktop/' + self.dataset_path + '/walk 2 anni/csv/' + sample + '/'

        print("\nWalk:", index_walk, sample)
        #print("Perc\t Delta Down %\t\t Label Down %\t\t Balanced Accuracy %\t\t Delta Por Down\t\t Label Por Down\t\t Delta AVG Precision\t\t Label Avg Precision\t\tOperazioni % / N° operazioni")
        print("Perc\t Delta Down %\t\t Label Down %\t\t Balanced Accuracy %\t\t Delta AVG Precision\t\t Label Avg Precision\t\tOperazioni % / N° operazioni")
        for perc in percs:
            df = pd.read_csv(path + 'walk_' + str(index_walk) + '_perc_' + str(perc) + '_test.csv')

            df = df_date_merger_binary(df=df.copy(), thr=self.thr_binary_labeling, columns=['delta_current_day', 'delta_next_day', 'open', 'close', 'high', 'low'], dataset=self.predictions_dataset)

            results = Measures.get_equity_return_mdd_romad(df=df.copy(), multiplier=50, type='long_only', penalty=25, stop_loss=1000, delta_to_use='delta_current_day', compact_results=True)
            
            label_precisions = confusion_matrix(df['label_current_day'].tolist(), df['decision'].tolist(), normalize='pred', labels = [0, 1])
            label_random_precision = Measures.get_binary_coverage(y=df['label_current_day'].tolist())

            coverage = Measures.get_binary_coverage(y=df['decision'].tolist())

            # Precision delta + random delta
            delta_precision = Measures.get_binary_delta_precision(y=df['decision'].tolist(), delta=df['delta_current_day'].tolist(), delta_val=-25)

            balanced_accuracy = balanced_accuracy_score(df['label_current_day'].tolist(), df['decision'].tolist())

            print(perc, "\t", \
                    round(delta_precision['down'], 2), "/", round(delta_precision['random_down'], 2), "\t\t",
                    round(label_precisions[0][0] * 100, 2), "/", round(label_random_precision['down_perc'], 2), "\t\t",
                    round(balanced_accuracy * 100, 2), "\t\t\t\t", 
                    #round(((delta_precision['down'] / delta_precision['random_down']) - 1 ) * 100, 2), "\t\t",
                    #round((((label_precisions[0][0] * 100) / label_random_precision['down_perc']) - 1 ) * 100, 2), "\t\t",
                    round((delta_precision['down'] + delta_precision['up']) / 2, 2), "\t\t\t\t",
                    round(((label_precisions[0][0] + label_precisions[1][1]) / 2) * 100, 2), "\t\t\t\t",
                    coverage['down_perc'], "/", coverage['down_count'],
            )

    

    '''
    '
    '''
    def create_plot(self, scale_pos_weight_list=[], perc=0, sample="20 sample 1h", plot_name=""):      
        delta_down_precision_avg = []
        cvg_avg_precision = []
        random_delta_precision = []

        return_avg = []
        mdd_avg = []
        romad_avg = []
        #por_down_avg = []

        for scale_pos_weight in scale_pos_weight_list:
            walk_0 = self.get_csv_runtime(index_walk=0, perc=perc, scale_pos_weight=scale_pos_weight, sample=sample)         
            walk_1 = self.get_csv_runtime(index_walk=1, perc=perc, scale_pos_weight=scale_pos_weight, sample=sample)         
            walk_2 = self.get_csv_runtime(index_walk=2, perc=perc, scale_pos_weight=scale_pos_weight, sample=sample)         
            
            walk_0 = df_date_merger_binary(df=walk_0.copy(), thr=self.thr_binary_labeling, columns=['delta_current_day', 'delta_next_day', 'open', 'close', 'high', 'low'], dataset=self.predictions_dataset)
            walk_1 = df_date_merger_binary(df=walk_1.copy(), thr=self.thr_binary_labeling, columns=['delta_current_day', 'delta_next_day', 'open', 'close', 'high', 'low'], dataset=self.predictions_dataset)
            walk_2 = df_date_merger_binary(df=walk_2.copy(), thr=self.thr_binary_labeling, columns=['delta_current_day', 'delta_next_day', 'open', 'close', 'high', 'low'], dataset=self.predictions_dataset)

            walk_0_coverage = Measures.get_binary_coverage(y=walk_0['decision'].tolist())
            walk_0_delta_precision = Measures.get_binary_delta_precision(y=walk_0['decision'].tolist(), delta=walk_0['delta_current_day'].tolist(), delta_val=-25)

            walk_1_coverage = Measures.get_binary_coverage(y=walk_1['decision'].tolist())
            walk_1_delta_precision = Measures.get_binary_delta_precision(y=walk_1['decision'].tolist(), delta=walk_1['delta_current_day'].tolist(), delta_val=-25)

            walk_2_coverage = Measures.get_binary_coverage(y=walk_2['decision'].tolist())
            walk_2_delta_precision = Measures.get_binary_delta_precision(y=walk_2['decision'].tolist(), delta=walk_2['delta_current_day'].tolist(), delta_val=-25)

            walk_0_random_label = Measures.get_binary_coverage(y=walk_0['label_current_day'].tolist())
            walk_1_random_label = Measures.get_binary_coverage(y=walk_1['label_current_day'].tolist())
            walk_2_random_label = Measures.get_binary_coverage(y=walk_2['label_current_day'].tolist())

            walk_0_results = Measures.get_equity_return_mdd_romad(df=walk_0.copy(), multiplier=50, type='long_only', penalty=25, stop_loss=1000, delta_to_use='delta_current_day', compact_results=True)
            walk_1_results = Measures.get_equity_return_mdd_romad(df=walk_1.copy(), multiplier=50, type='long_only', penalty=25, stop_loss=1000, delta_to_use='delta_current_day', compact_results=True)
            walk_2_results = Measures.get_equity_return_mdd_romad(df=walk_2.copy(), multiplier=50, type='long_only', penalty=25, stop_loss=1000, delta_to_use='delta_current_day', compact_results=True)
            #distribuzione campione
            avg_all = (walk_0_random_label['down_perc'] + walk_1_random_label['down_perc'] + walk_2_random_label['down_perc']) / 3

            # APPEND ZONE
            delta_down_precision_avg.append(round( (walk_0_delta_precision['down'] + walk_1_delta_precision['down'] + walk_2_delta_precision['down'] ) / 3, 2))
            cvg_avg_precision.append(round( (walk_0_coverage['down_perc'] + walk_1_coverage['down_perc'] + walk_2_coverage['down_perc'] ) / 3, 2))
             
            random_delta_precision.append(round((walk_0_delta_precision['random_down'] + walk_1_delta_precision['random_down'] + walk_2_delta_precision['random_down']) / 3, 2))

            return_avg.append( round((walk_0_results['return'] + walk_1_results['return'] + walk_2_results['return']) / 3, 2))
            mdd_avg.append( round((walk_0_results['mdd'] + walk_1_results['mdd'] + walk_2_results['mdd']) / 3, 2))
            romad_avg.append( round((walk_0_results['romad'] + walk_1_results['romad'] + walk_2_results['romad']) / 3, 2))
            
            # la media dei 3 walk, non da il por corretto 
            #por_walk_0 = ((walk_0_delta_precision['down'] / walk_0_delta_precision['random_down']) - 1) * 100
            #por_walk_1 = ((walk_1_delta_precision['down'] / walk_1_delta_precision['random_down']) - 1) * 100
            #por_walk_2 = ((walk_2_delta_precision['down'] / walk_2_delta_precision['random_down']) - 1) * 100
            #por_down_avg.append(round( (por_walk_0 + por_walk_1 + por_walk_2) / 3, 2))

        # PORC CHART

        avg_por = [] 
        for i,x  in enumerate(random_delta_precision):
            por = (( delta_down_precision_avg[i] / random_delta_precision[i]) - 1) * 100
            avg_por.append(round(por, 2))
        
        #print("'random':", random_delta_precision)
        print("'precision':", delta_down_precision_avg)
        print("'por':", avg_por)
        print("'cvg':", cvg_avg_precision)
        print("")
        print("'return':", return_avg)
        print("'mdd':", mdd_avg)
        print("'romad':", romad_avg)

        '''
        fig, ax = plt.subplots(figsize=(22,9))
        ax.plot(cvg_avg_precision, avg_por, label="PoR Thr" + str(self.thr_binary_labeling)) # por_down_avg -> 
        for i,j in zip(cvg_avg_precision, avg_por):
            ax.annotate("(" + str(j) + ", " + str(i) + ")",xy=(i,j))

        ax.set(xlabel='CVG', ylabel='PoR', title='')
        ax.grid()
        plt.legend(loc='best')

        path = 'C:/Users/Utente/Desktop/' + self.dataset_path + '/'
        create_folder(path)
        plt.savefig(path + plot_name + '_por.png')
        
        # PRECISION PLOT
        fig, ax = plt.subplots(figsize=(22,9))
        ax.plot(cvg_avg_precision, delta_down_precision_avg, label="Precision Thr" + str(self.thr_binary_labeling))
        ax.plot(cvg_avg_precision, random_delta_precision, color="orange", linestyle='--')
        for i,j in zip(cvg_avg_precision, delta_down_precision_avg):
            ax.annotate("(" + str(j) + ", " + str(i) + ")",xy=(i,j))
        ax.set(xlabel='CVG', ylabel='Precision', title='')
        ax.grid()
        plt.legend(loc='best')

        path = 'C:/Users/Utente/Desktop/' + self.dataset_path + '/'
        create_folder(path)
        plt.savefig(path + plot_name + '_precision.png')
        '''

    '''
    ' Metodo d'appoggio, calcolo le baseline di bh e bh2
    ' sia avg che totali sui 3 walk
    '''
    def get_results_bh(self, scale_pos_weight=1, perc=0, sample="20 sample 1h"):   
        walk_all = pd.DataFrame() 

        walk_0 = self.get_csv_runtime(index_walk=0, perc=perc, scale_pos_weight=scale_pos_weight, sample=sample)         
        walk_1 = self.get_csv_runtime(index_walk=1, perc=perc, scale_pos_weight=scale_pos_weight, sample=sample)         
        walk_2 = self.get_csv_runtime(index_walk=2, perc=perc, scale_pos_weight=scale_pos_weight, sample=sample)         
        
        walk_0 = df_date_merger_binary(df=walk_0.copy(), thr=self.thr_binary_labeling, columns=['delta_current_day', 'delta_next_day', 'open', 'close', 'high', 'low'], dataset=self.predictions_dataset)
        walk_1 = df_date_merger_binary(df=walk_1.copy(), thr=self.thr_binary_labeling, columns=['delta_current_day', 'delta_next_day', 'open', 'close', 'high', 'low'], dataset=self.predictions_dataset)
        walk_2 = df_date_merger_binary(df=walk_2.copy(), thr=self.thr_binary_labeling, columns=['delta_current_day', 'delta_next_day', 'open', 'close', 'high', 'low'], dataset=self.predictions_dataset)

        walk_all = pd.concat([walk_0, walk_1, walk_2], ignore_index=True)

        # All walk 
        bh_result_all = Measures.get_return_mdd_romad_bh(close=walk_all['close'].tolist(), multiplier=50, compact_results=True)
        bh_2_result_all = Measures.get_equity_return_mdd_romad(df=walk_all.copy(), multiplier=50, type='bh_long', penalty=25, stop_loss=1000, delta_to_use='delta_current_day', compact_results=True)
        
        # AVg
        bh_result_walk_0 = Measures.get_return_mdd_romad_bh(close=walk_0['close'].tolist(), multiplier=50, compact_results=True)
        bh_2_result_walk_0 = Measures.get_equity_return_mdd_romad(df=walk_0.copy(), multiplier=50, type='bh_long', penalty=25, stop_loss=1000, delta_to_use='delta_current_day', compact_results=True)

        bh_result_walk_1 = Measures.get_return_mdd_romad_bh(close=walk_1['close'].tolist(), multiplier=50, compact_results=True)
        bh_2_result_walk_1 = Measures.get_equity_return_mdd_romad(df=walk_1.copy(), multiplier=50, type='bh_long', penalty=25, stop_loss=1000, delta_to_use='delta_current_day', compact_results=True)

        bh_result_walk_2 = Measures.get_return_mdd_romad_bh(close=walk_2['close'].tolist(), multiplier=50, compact_results=True)
        bh_2_result_walk_2 = Measures.get_equity_return_mdd_romad(df=walk_2.copy(), multiplier=50, type='bh_long', penalty=25, stop_loss=1000, delta_to_use='delta_current_day', compact_results=True)

        # results variable
        return_bh_avg = round( (bh_result_walk_0['return'] + bh_result_walk_1['return'] + bh_result_walk_2['return']) / 3, 2)
        mdd_bh_avg = round( (bh_result_walk_0['mdd'] + bh_result_walk_1['mdd'] + bh_result_walk_2['mdd']) / 3, 2)
        romad_bh_avg = round( (bh_result_walk_0['romad'] + bh_result_walk_1['romad'] + bh_result_walk_2['romad']) / 3, 2)

        return_bh_2_avg = round( (bh_2_result_walk_0['return'] + bh_2_result_walk_1['return'] + bh_2_result_walk_2['return']) / 3, 2)
        mdd_bh_2_avg = round( (bh_2_result_walk_0['mdd'] + bh_2_result_walk_1['mdd'] + bh_2_result_walk_2['mdd']) / 3, 2)
        romad_bh_2_avg = round( (bh_2_result_walk_0['romad'] + bh_2_result_walk_1['romad'] + bh_2_result_walk_2['romad']) / 3, 2)

        return_bh_all = bh_result_all['return']
        mdd_bh_all = bh_result_all['mdd']
        romad_bh_all = round(bh_result_all['romad'], 2)
        
        return_bh_2_all = bh_2_result_all['return']
        mdd_bh_2_all = bh_2_result_all['mdd']
        romad_bh_2_all = round(bh_2_result_all['romad'], 2)
            
        print("\n\n\n")
        print("'bh_return_all':", np.full(12, return_bh_all))
        print("'bh_mdd_all':", np.full(12, mdd_bh_all))
        print("'bh_romad_all':", np.full(12, romad_bh_all))
        print("")
        print("'bh_2_return_all':", np.full(12, return_bh_2_all))
        print("'bh_2_mdd_all':", np.full(12, mdd_bh_2_all))
        print("'bh_2_romad_all':", np.full(12, romad_bh_2_all))
        print("\n")
        print("'bh_return_avg':", np.full(12, return_bh_avg))
        print("'bh_mdd_avg':", np.full(12, mdd_bh_avg))
        print("'bh_romad_avg':", np.full(12, romad_bh_avg))
        print("")
        print("'bh_2_return_avg':", np.full(12, return_bh_2_avg))
        print("'bh_2_mdd_avg':", np.full(12, mdd_bh_2_avg))
        print("'bh_2_romad_avg':", np.full(12, romad_bh_2_avg))

    '''
    '
    '''
    def get_csv_runtime(self, index_walk, perc, scale_pos_weight, sample):
        model = self.get_model(classifier='xgboost', scale_pos_weight=scale_pos_weight)
        
        
        #training_set = self.read_matrix(walk=index_walk, perc=perc, set_type='training', path='C:/Users/Utente/Desktop/' + self.dataset_path + '/walk 2 anni/json/' + sample + '/', balance_binary=True)
        #test_set = self.read_matrix(walk=index_walk, perc=perc, set_type='test', path='C:/Users/Utente/Desktop/' + self.dataset_path + '/walk 2 anni/json/' + sample + '/', balance_binary=False)

        training_set = self.read_matrix(walk=index_walk, perc=perc, set_type='training', path='C:/Users/Utente/Desktop/Dataset Json Classifier Vix/' + self.dataset_path + '/walk 2 anni/json/' + sample + '/', balance_binary=True)
        test_set = self.read_matrix(walk=index_walk, perc=perc, set_type='test', path='C:/Users/Utente/Desktop/Dataset Json Classifier Vix/' + self.dataset_path + '/walk 2 anni/json/' + sample + '/', balance_binary=False)


        x_train = training_set.get_x()
        y_train = training_set.get_y(referred_to='next_day')

        x_test = test_set.get_x()

        model.fit(x_train, y_train)

        # make predictions for test data
        y_pred = model.predict(x_test)
        
        
        dates_test = test_set.get_date_time()
        dates_test.reverse()

        y_pred = y_pred.tolist()
        y_pred.reverse()

        # shifto i valori per poterli usare su MC
        df = pd.DataFrame()
        df['date_time'] = dates_test
        df['decision'] = y_pred 
        df['date_time'] = df['date_time'].shift(-1)
        df = df.dropna()

        return df












    '''
    '
    '''
    def run_walks(self, dataset, classifier='xgboost'): 
        self.dataset = Market(dataset=dataset)
        
        data_20 = self.get_data_for_excel(path='classifier_json_dataset_20_sample_1h_walk_1anno')
        data_96 = self.get_data_for_excel(path='classifier_json_dataset_96_sample_5min_walk_1anno')
        
        self.generate_excel('excel_prova_short', data_20=data_20, data_96=data_96)

    '''
    '
    '''
    def get_data_for_excel(self, path):
        dates = []
        equities = []
        romads = []
        returns = []
        mdds = []
        cvgs_down = []
        cvgs_up = []
        cvgs_down_random = []
        cvgs_up_random = []
        precisions_down = []
        precisions_up = []
        balanced_accuracies = []
        avg_precisions = []
        pors_down = []
        n_up = []
        n_down = []

        bh_equities = []
        bh_romads = []
        bh_returns = []
        bh_mdds = []
        bh_i_equities = []
        bh_i_romads = []
        bh_i_returns = []
        bh_i_mdds = []
        
        # all walk
        df_final = pd.DataFrame()

        for index_walk, walk in enumerate(self.training_set_dates):
            df = pd.read_csv('C:/Users/Utente/Documents/GitHub/PhD-Market-Nets/' + path + '/predictions_walk_' + str(index_walk) + '_perc_0.csv')

            df = df_date_merger_binary(df=df.copy(), thr=self.thr_binary_labeling, columns=['delta_current_day', 'open', 'close', 'high', 'low'], dataset='sp500_cet')
            df = df.rename(columns={"predictions": "decision"})
            df['date_time'] = pd.to_datetime(df['date_time'])
            df = df.sort_values(by=['date_time'])
            df['date_time'] = df['date_time'].astype(str)
            df_final = pd.concat([df_final, df])


            date_list = df['date_time'].tolist()
            y_pred = df['decision'].tolist()
            y_test = df['label_current_day'].tolist()
            close = df['close'].tolist()

            #y_pred = [1 for y in y_pred]
            info_cvg = self.count_cvg(y_pred)
            random_cvg = self.count_cvg(y_test)

            #equity_line, romad, global_return, mdd = self.get_romad_return_mdd(y_pred=y_pred, delta=delta_current_day)
            equity_line, global_return, mdd, romad, i, j = Measures.get_equity_return_mdd_romad(df=df.copy(), multiplier=50, type='long_short', penalty=25, stop_loss=1000, delta_to_use='delta_current_day')
            bh_close, bh_return, bh_mdd, bh_romad, bh_i, bh_j = Measures.get_return_mdd_romad_bh(close=close, multiplier=50)
            #bh_i_equity_line, bh_i_romad, bh_i_return, bh_i_mdd = self.get_romad_return_mdd(y_pred=y_pred_intra, delta=delta_current_day)
            bh_i_equity_line, bh_i_return, bh_i_mdd, bh_i_romad, bh_ii, bh_jj = Measures.get_equity_return_mdd_romad(df=df.copy(), multiplier=50, type='bh_long', penalty=25, stop_loss=1000, delta_to_use='delta_current_day')
 
            balanced_accuracy = balanced_accuracy_score(y_test, y_pred)
            confusion = confusion_matrix(y_test, y_pred, normalize='pred', labels = [0, 1])

            equities.append(equity_line)
            romads.append(round(romad, 1))
            returns.append(global_return)
            mdds.append(mdd)
            cvgs_down.append(round(info_cvg['down_cvg'] * 100, 1))
            cvgs_up.append(round(info_cvg['up_cvg'] * 100, 1))
            cvgs_down_random.append(round(random_cvg['down_cvg'] * 100, 1))
            cvgs_up_random.append(round(random_cvg['up_cvg'] * 100, 1))
            n_up.append(info_cvg['n_up'])
            n_down.append(info_cvg['n_down'])

            balanced_accuracies.append(round(balanced_accuracy * 100, 1))
            
            precisions_down.append(round(confusion[0][0] * 100, 1))
            precisions_up.append(round(confusion[1][1] * 100,1))
            avg_precisions.append(round(((confusion[0][0] + confusion[1][1]) / 2) * 100, 1))
            pors_down.append(round(((confusion[0][0] / random_cvg['down_cvg']) - 1 ) * 100, 1))

            bh_equities.append(bh_close)
            bh_romads.append(round(bh_romad, 1))
            bh_returns.append(bh_return)
            bh_mdds.append(bh_mdd)

            bh_i_equities.append(bh_i_equity_line)
            bh_i_romads.append(round(bh_i_romad, 1))
            bh_i_returns.append(bh_i_return)
            bh_i_mdds.append(bh_i_mdd)
            dates.append(date_list)

        equity_line, global_return, mdd, romad, i, j = Measures.get_equity_return_mdd_romad(df=df_final.copy(), multiplier=50, type='long_short', penalty=25, stop_loss=1000, delta_to_use='delta_current_day')
        bh_close, bh_return, bh_mdd, bh_romad, bh_i, bh_j = Measures.get_return_mdd_romad_bh(close=df_final['close'].tolist(), multiplier=50)
        first_close = bh_close[0]
        bh_close = [i - first_close for i in bh_close]
        bh_i_equity_line, bh_i_return, bh_i_mdd, bh_i_romad, bh_ii, bh_jj = Measures.get_equity_return_mdd_romad(df=df_final.copy(), multiplier=50, type='bh_long', penalty=25, stop_loss=1000, delta_to_use='delta_current_day')
        
        date_list = df_final['date_time'].tolist()
        equities.append(equity_line)
        romads.append(round(romad, 1))
        returns.append(global_return)
        mdds.append(mdd)
        cvgs_down.append(round(info_cvg['down_cvg'] * 100, 1))
        cvgs_up.append(round(info_cvg['up_cvg'] * 100, 1))
        cvgs_down_random.append(round(random_cvg['down_cvg'] * 100, 1))
        cvgs_up_random.append(round(random_cvg['up_cvg'] * 100, 1))
        n_up.append(info_cvg['n_up'])
        n_down.append(info_cvg['n_down'])

        balanced_accuracies.append(round(balanced_accuracy * 100, 1))
        
        precisions_down.append(round(confusion[0][0] * 100, 1))
        precisions_up.append(round(confusion[1][1] * 100,1))
        avg_precisions.append(round(((confusion[0][0] + confusion[1][1]) / 2) * 100, 1))
        pors_down.append(round(((confusion[0][0] / random_cvg['down_cvg']) - 1 ) * 100, 1))

        bh_equities.append(bh_close)
        bh_romads.append(round(bh_romad, 1))
        bh_returns.append(bh_return)
        bh_mdds.append(bh_mdd)

        bh_i_equities.append(bh_i_equity_line)
        bh_i_romads.append(round(bh_i_romad, 1))
        bh_i_returns.append(bh_i_return)
        bh_i_mdds.append(bh_i_mdd)
        dates.append(date_list)

        data = {
            'dates': dates,
            'equities': equities, 
            'romads': romads,
            'returns': returns,
            'mdds': mdds,
            'cvgs_down': cvgs_down,
            'cvgs_up': cvgs_up,
            'n_down': n_down,
            'n_up': n_up,
            'cvgs_down_random': cvgs_down_random,
            'cvgs_up_random': cvgs_up_random,
            'precisions_down': precisions_down,
            'precisions_up': precisions_up,
            'avg_precisions': avg_precisions,
            'pors_down': pors_down,
            'balanced_accuracies': balanced_accuracies,

            'bh_equities': bh_equities,
            'bh_romads': bh_romads,
            'bh_returns': bh_returns,
            'bh_mdds': bh_mdds,
            'bh_i_equities': bh_i_equities,
            'bh_i_romads': bh_i_romads,
            'bh_i_returns': bh_i_returns,
            'bh_i_mdds': bh_i_mdds,
        }

        df_final.to_csv('C:/Users/Utente/Documents/GitHub/PhD-Market-Nets/' + path + '/predictions_walk_all_perc_0.csv', header=True, index=False)
        return data


