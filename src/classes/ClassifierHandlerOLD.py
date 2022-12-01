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
from classes.Utils import create_folder
from cycler import cycler
import sklearn.metrics as sklm  
from sklearn.metrics import confusion_matrix
import json
import platform

# classification moment
from sklearn import model_selection
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score
from sklearn.preprocessing import LabelEncoder
import xgboost
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm

class ClassifierHandler:

    iperparameters = {}
    dataset = pd.DataFrame()

    training_set_dates = []
    validation_set_dates = []
    test_set_dates = []

    number_of_walks = 0

    # ultime 20 ore, 20 giorni etc
    size_of_feature = 20
    # il numero di risoluzioni: 1h, 4h, 8h etc
    resolutions = 1


    '''
    '
    '''
    def __init__(self, iperparameters):
        self.iperparameters = iperparameters
        self.number_of_walks = np.array(iperparameters['training_set']).shape[0]

        self.training_set_dates = np.array(iperparameters['training_set'])
        self.validation_set_dates = np.array(iperparameters['validation_set'])
        self.test_set_dates = np.array(iperparameters['test_set'])

    '''
    ' Creo la matrice di un dataset, inserendo le 4 risoluzioni dentro la matrice
    '''
    def get_matrix(self, walk=0, dates=[], balance_binary=False):
        market = Market(dataset='sp500_cet')
        market = market.get_binary_labels(freq='1d', columns=['delta_current_day', 'delta_next_day', 'close'], thr=-0.5).reset_index()
        market = Market.get_df_by_data_range(df=market.copy(), start_date=dates[walk][0], end_date=dates[walk][1])

        # self.dataset contiene un dataset modificato a monte
        one_h = self.dataset.group(freq='1h', nan=False)
        #four_h = self.dataset.group(freq='4h', nan=False)
        #eight_h = self.dataset.group(freq='8h', nan=False)
        #one_d = self.dataset.group(freq='1d', nan=False)

        one_h = Market.get_df_by_data_range(df=one_h.copy(), start_date=dates[walk][0], end_date=dates[walk][1])
        #four_h = Market.get_df_by_data_range(df=four_h.copy(), start_date=dates[walk][0], end_date=dates[walk][1])
        #eight_h = Market.get_df_by_data_range(df=eight_h.copy(), start_date=dates[walk][0], end_date=dates[walk][1])
        #one_d = Market.get_df_by_data_range(df=one_d.copy(), start_date=dates[walk][0], end_date=dates[walk][1])


        #x = np.zeros(shape=(one_d.shape[0] - self.size_of_feature, 4, 20), dtype=np.uint8)
        x = np.zeros(shape=(market.shape[0] - self.size_of_feature, self.resolutions * self.size_of_feature), dtype=np.uint8)


        # tolgo i primi TOT elementi della lista
        date_list       = market['date_time'].tolist()[self.size_of_feature:]
        y_current_day   = market['label_current_day'].tolist()[self.size_of_feature:]
        y_next_day      = market['label_next_day'].tolist()[self.size_of_feature:]
        delta_next_day  = market['delta_next_day'].tolist()[self.size_of_feature:]
        delta_curr_day  = market['delta_current_day'].tolist()[self.size_of_feature:]
        close           = market['close'].tolist()[self.size_of_feature:]

        date_list.reverse()
        y_current_day.reverse()
        y_next_day.reverse()
        delta_next_day.reverse()
        delta_curr_day.reverse()
        close.reverse()

        for i, date in enumerate(date_list): 
            
            #print(date.strftime('%Y-%m-%d'))
            subset_one_h    = one_h.loc[one_h['date_time'] <= date + timedelta(hours=23)]
            #subset_four_h   = four_h.loc[four_h['date_time'] <= date + timedelta(hours=23)]
            #subset_eight_h  = eight_h.loc[eight_h['date_time'] <= date + timedelta(hours=23)]
            #subset_one_d    = one_d.loc[one_d['date_time'] <= date + timedelta(hours=23)]
            
            feature = []
            '''
            if y_next_day[i] == 0:
                #print("Sono dentro 0")
                feature = subset_one_h['delta_current_day_percentage'].tolist()[-self.size_of_feature - 1: -1] #+ \

                print(feature)
                print(date, y_next_day[i] )
                input()
                #feature = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0] 
                ''
                feature = subset_one_h['delta_current_day'].tolist()[-self.size_of_feature:] + \
                        subset_four_h['delta_current_day'].tolist()[-self.size_of_feature:] + \
                        subset_eight_h['delta_current_day'].tolist()[-self.size_of_feature:] + \
                        subset_one_d['delta_current_day'].tolist()[-self.size_of_feature:]
                ''
            if y_next_day[i] == 1: 
                #print("Sono dentro 1")
                feature = subset_one_h['delta_current_day_percentage'].tolist()[-self.size_of_feature - 1 : -1]
                    
                print(feature)
                print(date, y_next_day[i] )
                input()
                ''
                feature = subset_one_h['delta_current_day'].tolist()[-self.size_of_feature:] + \
                        subset_four_h['delta_current_day'].tolist()[-self.size_of_feature:] + \
                        subset_eight_h['delta_current_day'].tolist()[-self.size_of_feature:] + \
                        subset_one_d['delta_current_day'].tolist()[-self.size_of_feature:] #subset_one_h['delta_current_day'].tolist()[-self.size_of_feature:] #+ \ 
                ''
            #print(feature)
            #print(len(feature))
            #print(x.shape)
            #input()
            #feature = subset_one_h['delta_current_day'].tolist()[-self.size_of_feature:]
            '''
            
            feature = subset_one_h['delta_current_day_percentage'].tolist()[-self.size_of_feature - 1: -1]
            #print(feature)
            #print(date, y_next_day[i] )
            #input()
            x[i] = feature
            '''
            x[i] = [
                subset_one_h['delta_current_day'].tolist()[-self.size_of_feature:] +
                subset_four_h['delta_current_day'].tolist()[-self.size_of_feature:] + 
                subset_eight_h['delta_current_day'].tolist()[-self.size_of_feature:] + 
                subset_one_d['delta_current_day'].tolist()[-self.size_of_feature:]
            ]'''

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
    def get_model(self, classifier):

        if classifier == 'xgboost': 
            learning_rate = 0.1
            estimators = 500
            max_depth = 4
            seed = 1
            print("Using XGBClassifier. Learning_rate", learning_rate, "- N_estimators", estimators, "- Max_depth", max_depth, "- Seed", seed)
            return xgboost.XGBClassifier(learning_rate=learning_rate, n_estimators=estimators, max_depth=max_depth, seed=seed)

        if classifier == 'svm': 
            print("Using SVM." )
            return svm.SVC()

        if classifier == 'random_forest': 
            max_depth = 10
            random_state = 0
            print("Using Random Forest. Max Depth", max_depth, "- Random State", random_state)
            return RandomForestClassifier(max_depth=max_depth, random_state=random_state)
   
    '''
    '
    '''
    def run(self, dataset, classifier='xgboost'): 
        self.dataset = Market(dataset=dataset)
        
        
        
        ''' DEBUG
        dataset_prova = self.dataset.get()
        dataset_prova = Market.get_df_by_data_range(df=dataset_prova, start_date='2020-07-27', end_date='2020-07-28')
        print(dataset_prova[['date', 'open', 'close']])
        input()
        '''

        print("Dataset:", dataset)

        model = self.get_model(classifier=classifier)

        
        #for index_walk in range(self.number_of_walks):3
        index_walk = 0
        training_set = self.get_matrix(walk=index_walk, dates=self.training_set_dates, balance_binary=True)
        #validation_set = self.get_matrix(walk=index_walk, dates=self.validation_set_dates, balance_binary=False)
        test_set = self.get_matrix(walk=index_walk, dates=self.test_set_dates, balance_binary=False)
        
        
        x_train = training_set.get_x()
        y_train = training_set.get_y()

        x_test = test_set.get_x()
        y_test = test_set.get_y()

        
        model.fit(x_train, y_train)

        # make predictions for test data
        y_pred = model.predict(x_test)


        # evaluate predictions
        accuracy = accuracy_score(y_test, y_pred)

        confusion = confusion_matrix(y_test, y_pred, normalize='pred', labels = [0, 1])

        print("Walk n°", index_walk, "- Accuracy: %.2f%%" % (accuracy * 100.0))
        self.print_cm(cm=confusion, labels=['0', '1'])

        print("\n")
        
            