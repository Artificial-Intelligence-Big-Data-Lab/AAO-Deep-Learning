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

from classes.Measures import Measures
from classes.Utils import convert_probabilities


class Metrics(Callback):

    training_set = []
    validation_set = []
    test_set = []

    df_train_predictions = ''
    df_val_predictions = ''
    df_test_predictions = ''

    output_folder = ''
    output_training_folder = ""
    output_validation_folder = ""
    output_test_folder = ""

    index_walk = 0
    index_net = 0
    number_of_epochs = 0
    number_of_nets = 0

    return_multiplier = 50

    use_probabilities = False
    '''
    '
    '''
    #def __init__(self, validation_set, test_set, output_folder, index_walk, index_net, number_of_epochs, number_of_nets):
    def __init__(self, training_set, validation_set, test_set, output_folder, index_walk, index_net, number_of_epochs, number_of_nets, return_multiplier, use_probabilities=False):
        self.training_set = training_set
        self.validation_set = validation_set
        self.test_set = test_set
        
        self.output_folder = output_folder

        self.output_training_folder = self.output_folder + 'predictions/predictions_during_training/training/'
        self.output_validation_folder = self.output_folder + 'predictions/predictions_during_training/validation/'
        self.output_test_folder = self.output_folder + 'predictions/predictions_during_training/test/'

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
        
        y_pred_train = np.round(np.asarray(self.model.predict(self.training_set.get_x(normalized=True))), 2)
        y_pred_val = np.round(np.asarray(self.model.predict(self.validation_set.get_x(normalized=False))), 2)
        y_pred_test = np.round(np.asarray(self.model.predict( self.test_set.get_x(normalized=False))), 2)

        loss_test_set = self.model.evaluate(x=self.test_set.get_x(normalized=False), y=self.test_set.get_y(type='categorical', referred_to='next_day'))
        
        if 'val_loss' not in self.model.history.history:
            self.model.history.history["test_loss"] = []

        self.model.history.history['test_loss'].append(loss_test_set[0])

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
            
            train_filename = train_path + 'net_' + str(self.index_net) + '.csv'
            val_filename = val_path + 'net_' + str(self.index_net) + '.csv'
            test_filename = test_path + 'net_' + str(self.index_net) + '.csv'


            # Se non esiste la cartella, la creo
            if not os.path.isdir(train_path):
                os.makedirs(train_path)

            # Se non esiste la cartella, la creo
            if not os.path.isdir(val_path):
                os.makedirs(val_path)

            # Se non esiste la cartella, la creo
            if not os.path.isdir(test_path):
                os.makedirs(test_path)

            #self.df_val_predictions = self.df_val_predictions.reset_index()
            self.df_train_predictions.to_csv(train_filename, header=True, index=True)
            self.df_val_predictions.to_csv(val_filename, header=True, index=True)
            self.df_test_predictions.to_csv(test_filename, header=True, index=True)

        return

        '''
        if 'val_long_acc' not in self.model.history.history:
  
            val_coverage = Measures.get_delta_coverage(delta=self.validation_set.get_delta_next_day())
            test_coverage = Measures.get_delta_coverage(delta=self.test_set.get_delta_next_day())


            # validation history
            self.model.history.history["val_long_acc"] = []
            self.model.history.history["val_short_acc"] = []
            self.model.history.history["val_hold_acc"] = []
            self.model.history.history["val_romad_ls"] = []
            self.model.history.history["val_romad_lh"] = []
            self.model.history.history["val_romad_sh"] = []
            self.model.history.history["val_return_ls"] = []
            self.model.history.history["val_return_lh"] = []
            self.model.history.history["val_return_sh"] = []
            self.model.history.history["val_mdd_ls"] = []
            self.model.history.history["val_mdd_lh"] = []
            self.model.history.history["val_mdd_sh"] = []
            self.model.history.history["val_return_bh"] = []
            self.model.history.history["val_long_perc"] = []
            self.model.history.history["val_short_perc"] = []
            self.model.history.history["val_hold_perc"] = []
            # coverage delle label originali del dataset
            self.model.history.history["val_set_label_coverage"] = val_coverage

            self.model.history.history["test_long_acc"] = []
            self.model.history.history["test_short_acc"] = []
            self.model.history.history["test_hold_acc"] = []
            self.model.history.history["test_romad_ls"] = []
            self.model.history.history["test_romad_lh"] = []
            self.model.history.history["test_romad_sh"] = []
            self.model.history.history["test_return_ls"] = []
            self.model.history.history["test_return_lh"] = []
            self.model.history.history["test_return_sh"] = []
            self.model.history.history["test_mdd_ls"] = []
            self.model.history.history["test_mdd_lh"] = []
            self.model.history.history["test_mdd_sh"] = []
            self.model.history.history["test_return_bh"] = []
            self.model.history.history["test_long_perc"] = []
            self.model.history.history["test_short_perc"] = []
            self.model.history.history["test_hold_perc"] = []
            # coverage delle label originali del dataset
            self.model.history.history["test_set_label_coverage"] = test_coverage

            if (epoch > 0):
                print("ERROR IN AVERAGES: history already exists but epoch is ", epoch)
                input()

        validation_values = self.calculator(y_pred=y_pred_val, delta=self.validation_set.get_delta_next_day(), close=self.validation_set.get_close(), type='Validation')
        test_values = self.calculator(y_pred=y_pred_test, delta=self.test_set.get_delta_next_day(), close=self.test_set.get_close(), type='Test')

        self.model.history.history["val_short_acc"].append(validation_values['short_acc'])
        self.model.history.history["val_hold_acc"].append(validation_values['hold_acc'])
        self.model.history.history["val_long_acc"].append(validation_values['long_acc'])
        self.model.history.history["val_long_perc"].append(validation_values['long_perc'])
        self.model.history.history["val_short_perc"].append(validation_values['short_perc'])
        self.model.history.history["val_hold_perc"].append(validation_values['hold_perc'])
        self.model.history.history["val_return_bh"].append(validation_values['bh_return'])
        self.model.history.history["val_romad_ls"].append(validation_values['romad_ls'])
        self.model.history.history["val_romad_lh"].append(validation_values['romad_lh'])
        self.model.history.history["val_romad_sh"].append(validation_values['romad_sh'])
        self.model.history.history["val_return_ls"].append(validation_values['return_ls'])
        self.model.history.history["val_return_lh"].append(validation_values['return_lh'])
        self.model.history.history["val_return_sh"].append(validation_values['return_sh'])
        self.model.history.history["val_mdd_ls"].append(validation_values['mdd_ls'])
        self.model.history.history["val_mdd_lh"].append(validation_values['mdd_lh'])
        self.model.history.history["val_mdd_sh"].append(validation_values['mdd_sh'])

        self.model.history.history["test_short_acc"].append(test_values['short_acc'])
        self.model.history.history["test_hold_acc"].append(test_values['hold_acc'])
        self.model.history.history["test_long_acc"].append(test_values['long_acc'])
        self.model.history.history["test_long_perc"].append(test_values['long_perc'])
        self.model.history.history["test_short_perc"].append(test_values['short_perc'])
        self.model.history.history["test_hold_perc"].append(test_values['hold_perc'])
        self.model.history.history["test_return_bh"].append(test_values['bh_return'])
        self.model.history.history["test_romad_ls"].append(test_values['romad_ls'])
        self.model.history.history["test_romad_lh"].append(test_values['romad_lh'])
        self.model.history.history["test_romad_sh"].append(test_values['romad_sh'])
        self.model.history.history["test_return_ls"].append(test_values['return_ls'])
        self.model.history.history["test_return_lh"].append(test_values['return_lh'])
        self.model.history.history["test_return_sh"].append(test_values['return_sh'])
        self.model.history.history["test_mdd_ls"].append(test_values['mdd_ls'])
        self.model.history.history["test_mdd_lh"].append(test_values['mdd_lh'])
        self.model.history.history["test_mdd_sh"].append(test_values['mdd_sh'])
        '''
        

    '''
    '
    ''
    def calculator(self, y_pred, delta, close, type="Validation"): 
        # LONG + SHORT
        ls_equity_line, ls_global_return, ls_mdd, ls_romad, ls_i, ls_j = Measures.get_equity_return_mdd_romad(y_pred=y_pred, delta=delta, multiplier=self.return_multiplier, type='long_short')
        long, short, hold, general = Measures.get_precision_count_coverage(y_pred=y_pred, delta=delta)

        # LONG ONLY
        lh_equity_line, lh_global_return, lh_mdd, lh_romad, lh_i, lh_j = Measures.get_equity_return_mdd_romad(y_pred=y_pred, delta=delta, multiplier=self.return_multiplier, type='long_only')

        # SHORT ONLY
        sh_equity_line, sh_global_return, sh_mdd, sh_romad, sh_i, sh_j = Measures.get_equity_return_mdd_romad(y_pred=y_pred, delta=delta, multiplier=self.return_multiplier, type='short_only')

        bh_equity_line, bh_return, bh_mdd, bh_romad, bh_i, bh_j = Measures.get_return_mdd_romad_bh(close=close, multiplier=self.return_multiplier) 
        
        print(type + '- LONGs: ' + str(long['guessed']) + '/' + str(long['count']) + '(' + str(long['precision']) + ')')
        print(type + '- SHORTs: ' + str(short['guessed']) + '/' + str(short['count']) + '(' + str(short['precision']) + ')')
        print(type + '- HOLDs: ' + str(hold['count']))

        print(type + " - LONG SHORT:  (return: " + str(ls_global_return) + " | mdd: " + str(ls_mdd) + ")")
        print(type + " - LONG ONLY:  (return: " + str(lh_global_return) + " | mdd: " + str(lh_mdd) + ")")
        print(type + " - SHORT ONLY:  (return: " + str(sh_global_return) + " | mdd: " + str(sh_mdd) + ")")

        values = {
            'short_acc': short['precision'], 
            'hold_acc': 0, 
            'long_acc': long['precision'], 
            'short_perc': short['coverage'], 
            'hold_perc': hold['coverage'], 
            'long_perc': long['coverage'], 

            'bh_return': bh_return, 

            'romad_ls': ls_romad, 
            'romad_lh': lh_romad, 
            'romad_sh': sh_romad, 

            'return_ls': ls_global_return, 
            'return_lh': lh_global_return, 
            'return_sh': sh_global_return, 

            'mdd_ls': ls_mdd, 
            'mdd_lh': lh_mdd, 
            'mdd_sh': sh_mdd, 
        }

        return values
    '''