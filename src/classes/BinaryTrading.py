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



class BinaryTrading:

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

    long = 1
    hold = 0

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

    
    def run(self, stop_loss=1000, penalty=25, multiplier=50):
        print("Inizio ad eseguire il trading binario")
        epoch = "400"

        df = pd.read_csv(self.original_predictions_test_folder + 'walk_4/net_0.csv')
        df = df[['date_time', 'epoch_' + epoch]]
        df = df.rename({'epoch_' + epoch: 'decision'}, axis=1)
        df = df_date_merger_binary(df=df.copy(), thr=self.iperparameters['thr_labeling'], columns=['open', 'close', 'high', 'low'])



        
        stop_loss_point = stop_loss / multiplier
        penalty_points = penalty / multiplier if penalty > 0 else 0

        position_open = False
        last_open = 0
        last_close = 0

        cumulative_profit = []

        print(df)
        for index, row in df.iterrows():
            #print("Data:", row['date_time'], "\t", "Decision:", row['decision'], "\t", "Position open:", position_open)
            
            if position_open == False and row['decision'] == self.long: 
                print("[ENTRY LONG] Data:", row['date_time'], "Open:", row['open'])
                position_open = True
                last_open = row['open']

            if position_open == True:
                if row['low'] - row['open'] > stop_loss_point:
                    last_close = row['low']
                    position_open = False
                    cumulative_profit.append(last_close - last_open)
                    print("[EXIT STOP LOSS] Data:", row['date_time'], "\tProfit:", str(last_close - last_open), "\n")
                    last_open = 0
                    last_close = 0
                else: 
                    last_close = row['close']
                    
                if index == df.shape[0] -1 :
                    print("[EXIT] Data:", row['date_time'], "\tProfit:", str(last_close - last_open))
                    position_open = False
                    cumulative_profit.append(last_close - last_open)