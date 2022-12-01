import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import sys
import cv2
import time
import pickle
import datetime
import matplotlib
matplotlib.use("Agg")
import numpy as np
import pandas as pd
from datetime import timedelta
from keras.callbacks import Callback
import sklearn.metrics as sklm  
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

from classes.Measures import Measures
from classes.ResultsHandler import ResultsHandler
from classes.Utils import convert_probabilities, natural_keys, df_date_merger, do_plot, revert_probabilities, create_folder

import json


class Metrics(Callback):

    training_set = []
    validation_set = []
    test_set = []

    df_train_predictions = ''
    df_val_predictions = ''
    df_test_predictions = ''

    iperparameters = {}
    base_path = ''
    output_folder = ''
    output_training_folder = ""
    output_validation_folder = ""
    output_test_folder = ""
    output_selection_folder = ''

    index_walk = 0
    index_net = 0
    number_of_epochs = 0
    number_of_nets = 0

    return_multiplier = 50

    use_probabilities = False
    
    rh = ''

    '''
    '
    '''
    #def __init__(self, validation_set, test_set, output_folder, index_walk, index_net, number_of_epochs, number_of_nets):
    def __init__(self, training_set, validation_set, test_set, iperparameters, base_path, output_folder, index_walk, index_net, number_of_epochs, number_of_nets, return_multiplier, use_probabilities=False):
        self.training_set = training_set
        self.validation_set = validation_set
        self.test_set = test_set

        self.iperparameters = iperparameters
        self.base_path = base_path
        self.output_folder = output_folder

        self.output_training_folder = self.output_folder + 'predictions/predictions_during_training/training/'
        self.output_validation_folder = self.output_folder + 'predictions/predictions_during_training/validation/'
        self.output_test_folder = self.output_folder + 'predictions/predictions_during_training/test/'
        self.output_selection_folder = self.output_folder + 'selection/'

        self.index_walk = index_walk
        self.index_net = index_net
        self.number_of_epochs = number_of_epochs
        self.number_of_nets = number_of_nets

        self.df_train_predictions = pd.DataFrame() 
        self.df_val_predictions = pd.DataFrame() 
        self.df_test_predictions = pd.DataFrame() 

        self.df_train_predictions['date_time'] = training_set.get_date_time()
        self.df_val_predictions['date_time'] = validation_set.get_date_time()
        self.df_test_predictions['date_time'] = test_set.get_date_time()

        self.return_multiplier = return_multiplier
        self.use_probabilities = use_probabilities

        
    
    '''
    '
    '''
    def on_epoch_end(self, epoch, logs={}):
        

        #score = np.asarray(self.model.predict(self.validation_data[0]))
        y_true = np.array(self.validation_data[1]).flatten()
        
        y_pred_train = np.round(np.asarray(self.model.predict(self.training_set.get_x(normalized=True, multivariate=self.iperparameters['multivariate']))), 2)
        y_pred_val = np.round(np.asarray(self.model.predict(self.validation_set.get_x(normalized=False, multivariate=self.iperparameters['multivariate']))), 2)
        y_pred_test = np.round(np.asarray(self.model.predict( self.test_set.get_x(normalized=False, multivariate=self.iperparameters['multivariate']))), 2)

        loss_test_set = self.model.evaluate(x=self.test_set.get_x(normalized=False, multivariate=self.iperparameters['multivariate']), y=self.test_set.get_y(type='categorical', referred_to='next_day'))

        y_true_test = self.test_set.get_y(type='classic', referred_to='next_day')

        accuracy_test_set = accuracy_score(y_true_test, np.argmax(y_pred_test, axis=1), normalize=True)

        if 'val_loss' not in self.model.history.history:
            self.model.history.history["test_loss"] = []
            self.model.history.history["test_accuracy"] = []


        self.model.history.history['test_loss'].append(loss_test_set[0])
        self.model.history.history['test_accuracy'].append(accuracy_test_set)

        if self.use_probabilities is False:
            y_pred_train = np.argmax(y_pred_train, axis=1)
            y_pred_val = np.argmax(y_pred_val, axis=1)
            y_pred_test  = np.argmax(y_pred_test, axis=1)

        if self.use_probabilities is True:
            y_pred_train = np.apply_along_axis(convert_probabilities, 1, y_pred_train)
            y_pred_val = np.apply_along_axis(convert_probabilities, 1, y_pred_val)
            y_pred_test  = np.apply_along_axis(convert_probabilities, 1, y_pred_test)         
        
        
        self.df_train_predictions['epoch_' + str(epoch + 1)] = y_pred_train
        self.df_val_predictions['epoch_' + str(epoch + 1)] = y_pred_val        
        self.df_test_predictions['epoch_' + str(epoch + 1)] = y_pred_test

        # sono nell'if dell'ultima epoca
        if(self.number_of_epochs == epoch + 1):
            train_path = self.output_training_folder + 'walk_' + str(self.index_walk) + '/'
            val_path = self.output_validation_folder + 'walk_' + str(self.index_walk) + '/'
            test_path = self.output_test_folder + 'walk_' + str(self.index_walk) + '/'

            bl_path = self.output_selection_folder + 'best_LONG/'
            bs_path = self.output_selection_folder + 'best_SHORT/'
            bls_path = self.output_selection_folder + 'best_LONG_SHORT/'
            
            train_filename = train_path + 'net_' + str(self.index_net) + '.csv'
            val_filename = val_path + 'net_' + str(self.index_net) + '.csv'
            test_filename = test_path + 'net_' + str(self.index_net) + '.csv'

            #if not os.path.isdir(bls_path):
            #    os.makedirs(bls_path)

            if not os.path.isdir(train_path):
                os.makedirs(train_path)

            if not os.path.isdir(val_path):
                os.makedirs(val_path)

            if not os.path.isdir(test_path):
                os.makedirs(test_path)

            #self.df_val_predictions = self.df_val_predictions.reset_index()
            #self.df_train_predictions.to_csv(train_filename, header=True, index=True)

            self.df_val_predictions.to_csv(val_filename, header=True, index=True)
            self.df_test_predictions.to_csv(test_filename, header=True, index=True)

            if 'experiment_path' in self.iperparameters:
                self.rh = ResultsHandler(experiment_name=self.iperparameters['experiment_name'], experiment_path=self.iperparameters['experiment_path'])
            else: 
                self.rh = ResultsHandler(experiment_name=self.iperparameters['experiment_name'])
                
            val_json = self.rh.generate_single_net_json(type='validation', index_walk=self.index_walk, index_net=self.index_net, penalty=0, stop_loss=0)
            test_json = self.rh.generate_single_net_json(type='test', index_walk=self.index_walk, index_net=self.index_net, penalty=0, stop_loss=0)

            if self.index_net == (self.iperparameters["number_of_nets"] - 1):
                # Genero gli avg per questo walk
                self.rh.generate_avg_json(type='validation', id_walk=self.index_walk)
                self.rh.generate_avg_json(type='test', id_walk=self.index_walk)
            
            # Long-only selection
            #self.rh.get_best_epoch(path=bl_path, rule_type='long', min_thr=2, step=2, net=self.index_net, valid=val_json, test=test_json)
            #self.rh.get_best_epoch(path=bs_path, rule_type='short', min_thr=3, step=3, net=self.index_net, valid=val_json, test=test_json)

        return