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
from classes.CustomLoss import w_categorical_crossentropy
from classes.Utils import create_folder
from cycler import cycler
from keras.models import load_model
from vgg16.SmallerVGGNet import SmallerVGGNet
from sklearn.preprocessing import LabelBinarizer, MultiLabelBinarizer
from keras.optimizers import Adam, RMSprop, SGD, Nadam
from keras_radam import RAdam
from keras import backend as K
from keras.callbacks import ModelCheckpoint
from keras.callbacks import Callback
import tensorflow as tf
import sklearn.metrics as sklm  
from sklearn.metrics import confusion_matrix
from keras.utils import to_categorical
import json
import platform

'''
' Questa classe serve per lanciare e gestire 
' la configurazione della CNN
' @author Andrea corriga
'''
class VggBinaryHandler:
    platform = platform.platform()
    
    # path iniziale da dove leggere le immagini da passare alla rete
    images_base_path = '../images/'

    # qui verrà inserito il path completo una volta passato il parametro in input
    images_full_paths = []

    # i datasets usati input per il training ed il validation
    input_datasets = []

    # il datasets che voglio predire
    predictions_dataset = ''

    # cartella con le immagini per il dataset su cui voglio fare le predizioni
    predictions_images_folder = ''

    # path dove verranno salvati gli esperimenti
    if platform == 'Linux-4.15.0-45-generic-x86_64-with-Ubuntu-16.04-xenial': 
        output_base_path = '/media/unica/HDD 9TB Raid0 - 1/experiments/'
    else: 
        output_base_path = 'D:/PhD-Market-Nets/experiments/' # locale 
        #output_base_path = '/var/www/html/experiments' # server hawkeye
     

    base_path = ''
    output_folder = ''

    # dataframe con label del giorno
    label_df = pd.DataFrame()

    # dataframe con date_time e label del giorno dopo [la uso per allenare la rete]
    date_label_df = pd.DataFrame()

    # configurazione base della rete
    number_of_nets = 20
    epochs = 200
    init_lr = 0.001
    bs = 32

    # parametri passati alla rete
    training_set = []
    validation_set = []
    test_set = []
    number_of_walks = 0
    input_shape = (40, 40, 3)

    save_pkl = False
    save_model_history = False
    model_history_period = epochs

    history_array = []

    acc_name = "accuracy"
    val_acc_name = "val_accuracy"

    # Moltiplicatore per i grafici del return
    return_multiplier = 50
    loss_function = 'sparse_categorical_crossentropy'
    custom_loss_weight = np.ones((3,3))

    use_probabilities = False
    
    iperparameters = {}

    '''
    '
    '''
    def __init__(self, iperparameters):
        if not isinstance(iperparameters['number_of_nets'], int):
            sys.exit("VggHandler.net_config number_of_nets must be a int value")

        if not isinstance(iperparameters['epochs'], int):
            sys.exit("VggHandler.net_config epochs must be a int value")

        if not isinstance(iperparameters['init_lr'], float):
            sys.exit("VggHandler.net_config init_lr must be a float")

        if not isinstance(iperparameters['bs'], int):
            sys.exit("VggHandler.net_config bs must be a int value")

        if not isinstance(iperparameters['save_pkl'], bool): 
            sys.exit("VggHandler.net_config save_pkl must be a bool value")
        
        if not isinstance(iperparameters['save_model_history'], bool): 
            sys.exit("VggHandler.net_config save_model_history must be a bool value")

        if iperparameters['save_model_history'] is True and not isinstance(iperparameters['model_history_period'], int): 
            sys.exit("VggHandler.net_config model_history_period must be a bool value")

        if iperparameters['model_history_period'] > iperparameters['epochs']: 
            sys.exit("VggHandler.net_config model_history_period must be lesser dan epochs parameter")

        self.iperparameters = iperparameters

        self.number_of_nets = iperparameters['number_of_nets']
        self.epochs = iperparameters['epochs']
        self.init_lr = iperparameters['init_lr']
        self.bs = iperparameters['bs']

        self.save_pkl = iperparameters['save_pkl']
        self.save_model_history = iperparameters['save_model_history']
        
        self.return_multiplier = iperparameters['return_multiplier']

        if iperparameters['save_model_history'] == True: 
            if iperparameters['model_history_period'] == None: 
                self.model_history_period = iperparameters['epochs']
            else: 
                self.model_history_period = iperparameters['model_history_period']
        
        if iperparameters['save_model_history'] == False:
            self.model_history_period = None

        self.loss_function = iperparameters['loss_function']
        self.custom_loss_weight = iperparameters['loss_weight']

        if 'experiment_path' in iperparameters:
            self.output_base_path = iperparameters['experiment_path']

        self.base_path = self.output_base_path + iperparameters['experiment_name']

        ############ run initialize ################

        # il dataset in cui voglio fare le predizioni
        self.predictions_dataset = iperparameters['predictions_dataset']
        # la cartella con le immagini presenti per il dataset su cui voglio fare le predizioni
        self.predictions_images_folder = iperparameters['predictions_images_folder']

        # setto la variabile con i datasets su cui fare il training
        self.input_datasets = iperparameters['input_datasets']
        
        
        if len(iperparameters['input_images_folders']) != len(iperparameters['input_datasets']):
            sys.exit('VggHandler.init: input_images_folders and input_datasets must have same len. input_images_folders: ' + str(len(iperparameters['input_images_folders'])) + ' input_datasets: ' + str(len(iperparameters['input_datasets'])))

        # La path delle immagini su cui voglio fare il training
        for iperparameters['input_folder'] in iperparameters['input_images_folders']:
            self.images_full_paths.append(self.images_base_path + iperparameters['input_folder'])

        # genero il df con la label
        datasets = Market(dataset=iperparameters['predictions_dataset']) 
        #label_df = datasets.get_binary_labels(freq='1d', columns=['delta_next_day', 'delta_current_day', 'close'], thr=self.iperparameters['thr_binary_labeling']).reset_index()
        #label_df = datasets.get_binary_labels_volatility(freq='1d', columns=['delta_next_day', 'delta_current_day', 'close'], thr=self.iperparameters['thr_binary_labeling']).reset_index()
        label_df = datasets.get_binary_labels_volatility_7_days(freq='1d', columns=['delta_next_day', 'delta_current_day', 'close'], thr=self.iperparameters['thr_binary_labeling'], days=7).reset_index()

        # converto in np.array validation, traning e test set
        training_set = np.array(iperparameters['training_set'])
        validation_set = np.array(iperparameters['validation_set'])
        test_set = np.array(iperparameters['test_set'])

        # training, validation e test devono essere di dimensione uguale
        if training_set.shape != validation_set.shape or training_set.shape != test_set.shape or validation_set.shape != test_set.shape:
            debug_string_shape = "Training set shape " + str(training_set.shape) + " - Validation set shape " + str(validation_set.shape) + " - Test set shape " + str(test_set.shape)
            sys.exit("VggHandler.run_initialize: traning_set, validation_set and test_set must have same shape. " + debug_string_shape)

        # setto gli attributi interni della classe
        self.training_set = training_set
        self.validation_set = validation_set
        self.test_set = test_set
        self.number_of_walks = training_set.shape[0]
        self.output_folder = self.output_base_path + iperparameters['experiment_name'] + '/'
        self.input_shape = iperparameters['input_shape']

        # creo la cartella di output per gli esperimenti
        if not os.path.isdir(self.output_folder):
            os.makedirs(self.output_folder)
        
        # Creo un DF date_time: nome_immagine (poiché vengono sempre salvate con il nome del giorno
        date_df = pd.DataFrame({'date_time': pd.to_datetime(self.images_filename_reader(self.images_full_paths[0]))}).sort_values(by=['date_time'])
        date_df = date_df.reset_index().drop(['index'], axis=1)

        # Unisco i DF con i giorni e la label del giorno successivo. Qui dentro quindi ci sono le label e le date presenti nelle gadf
        self.date_label_df = pd.merge(date_df, label_df, how='inner', on='date_time')
        
        self.use_probabilities = iperparameters['use_probabilities']

        iperparameters['loss_weight'] = iperparameters['loss_weight'].tolist()

        with open(self.output_folder + '/log.json', 'w') as json_file:
            json.dump(iperparameters, json_file, indent=4)

        output_selection_folder = self.output_folder + 'selection/'

        bl_path = output_selection_folder + 'best_LONG/'
        bs_path = output_selection_folder + 'best_SHORT/'
        bls_path = output_selection_folder + 'best_LONG_SHORT/'

        # Se non esiste la cartella, la creo
        if not os.path.isdir(bl_path):
            os.makedirs(bl_path)

        if not os.path.isdir(bs_path):
            os.makedirs(bs_path)

        #headers
        header_long = 'net;epoch;por;valid_romad;valid_return;valid_mdd;valid_cove;test_romad;test_return;test_mdd;test_cove;por15_grp20;por15_grp50;por15_grp100;por10_grp20;por10_grp50;por10_grp100;por5_grp20;por5_grp50;por5_grp100;\n'
        header_short = 'net;epoch;por;valid_romad;valid_return;valid_mdd;valid_cove;test_romad;test_return;test_mdd;test_cove;por30_grp20;por30_grp50;por30_grp100;por20_grp20;por20_grp50;por20_grp100;por10_grp20;por10_grp50;por10_grp100;\n'

        # Long-only selection
        f=open(bl_path + 'long_selected.csv', "a+")
        f.write(header_long)

        # short
        f=open(bs_path + 'short_selected.csv', "a+")
        f.write(header_short)

    '''
    ' Funzione di appoggio, leggo tutte le
    ' immagini dentro una cartella. La uso per 
    ' prendere i nomi delle immagini che unirò successivamente
    '''
    def images_filename_reader(self, path):
        filename_list = []

        for file in os.listdir(path):
            elements = file.split('.')
            filename_list.append(elements[0])

        return filename_list

    '''
    ' Metodo di appoggio, preso da Market
    ' Lo uso per ottenere una porzione di un dataframe per data range
    ' Lo uso per ottenere i subset di training, validation e test set
    '''
    def get_df_by_data_range(self, df, start_date, end_date):
        # Search mask
        mask = (df['date_time'] >= start_date) & (df['date_time'] < end_date)
        # Get the subset of sp500
        return df.loc[mask]

    '''
    ' @author Silvio Barra
    ' @edited_by Andrea Corriga
    ' Salvo il modello e la history generata nella loro rispettiva cartella
    '''
    def save_models(self, model, H, index_walk, index_net):
        full_models_path = self.output_folder + 'models/'
        full_histories_path = self.output_folder + 'histories/'

        # Se non esiste la cartella, la creo
        if not os.path.isdir(full_models_path):
            os.makedirs(full_models_path)

        if self.save_pkl == True: 
            if not os.path.isdir(full_histories_path):
                os.makedirs(full_histories_path)
        
        if self.save_pkl == True:
            # Salvo la history
            f = open(full_histories_path + 'walk_' + str(index_walk) + '_net_' + str(index_net) + '.pkl', "wb")
            f.write(pickle.dumps(H))
            f.close()

        # Salvo il modello
        model.save(full_models_path + 'walk_' + str(index_walk) + '_net_' + str(index_net) + '.model')
    
    

    '''
    ' Salvo un file di log.txt in cui inserisco
    ' data di inizio esecuzione e tutta la configurazione della rete
    ' salvo anche i dati relativi al training, validation e test set
    '''
    def start_log(self):
        f=open(self.output_folder + 'log.txt','a')
        
        f.write("Experiment name: " + self.output_folder + "\n")

        f.write("Input_datasets: ")
        for inpt_dts in self.input_datasets:
            f.write(inpt_dts + " ")
        f.write("\n")

        f.write("Input images folder: ")
        for inpt_img in self.images_full_paths:
            f.write(inpt_img + " ")
        f.write("\n")

        f.write("Predictions dataset: " + self.predictions_dataset + "\n")
        f.write("\n")

        f.write("Net Start: " + str(datetime.datetime.now()) + "\n")

        f.write("\n")

        f.write("Number of nets: " + str(self.number_of_nets) + "\n")
        f.write("Epochs: " + str(self.epochs) + "\n")
        f.write("Init lr: " + str(self.init_lr) + "\n")
        f.write("Bs: " + str(self.bs) + "\n")

        f.write("\n")

        f.write("Number of walks: " + str(self.number_of_walks) + "\n")
        
        f.write("Walk \t\t Training \t\t\t Validation \t\t\t Test\n")
        
        for index_walk in range(self.number_of_walks):
            f.write("Walk " + str(index_walk) + ": \t ")
            f.write("[" + self.training_set[index_walk][0] + " - " + self.training_set[index_walk][1] + "] \t ")
            f.write("[" + self.validation_set[index_walk][0] + " - " + self.validation_set[index_walk][1] + "] \t ")
            f.write("[" + self.test_set[index_walk][0] + " - " + self.test_set[index_walk][1] + "] \n")
        
        f.write("\n")
        f.close()

    '''
    ' Aggiunge a fine file di log quando si 
    ' è conclusa l'esecuzione della rete
    '''
    def end_log(self):
        f=open(self.output_folder + 'log.txt','a')
        
        f.write("\nNet Stop: " + str(datetime.datetime.now()) + "\n")
        f.write("\n")
        f.close()
    
    '''
    '
    '''
    def add_line_log(self, line): 
        f=open(self.output_folder + 'log.txt','a')
        f.write("["+ str(datetime.datetime.now()) +"] " + str(line) + "\n")
        f.close()

    '''
    ' Per recuperare l'esecuzione in caso di crash
    '''
    def save_last_walk_log(self, index_walk, is_last, gpu_id=0): 
        f=open(self.output_base_path + 'last_walk_log_gpu_' + str(gpu_id) + '.txt','w')
        f.write(str(index_walk) + "\n" + str(is_last))
        f.close()
    
    '''
    ' @author Silvio Barra
    ' @edited_by Andrea Corriga
    ' Effettuo il training della rete
    '''
    def train_2D(self, training_set, validation_set, test_set, index_net, index_walk):  
        # [INFO] compiling model...
        #model = SmallerVGGNet.build_vgg16_2d_smaller(height=self.input_shape[0], width=self.input_shape[1], depth=self.input_shape[2], classes=3, init_var=index_net)
        #model = SmallerVGGNet.build_vgg16(height=self.input_shape[0], width=self.input_shape[1], depth=self.input_shape[2], classes=len(lb.classes_))
        #model = SmallerVGGNet.build_anse_2d(height=self.input_shape[0], width=self.input_shape[1], depth=self.input_shape[2], init_var=index_net)
        #model = SmallerVGGNet.build_anse_v2(height=40, width=40, depth=3, init_var=index_net)
        model = ''

        # univariate
        if self.iperparameters['multivariate'] == False:
            #model = SmallerVGGNet.build_small_2d_binary(width=self.input_shape[0], height=self.input_shape[1], depth=self.input_shape[2], init_var=index_net)
            #model = SmallerVGGNet.build_vgg16_2d_smaller(width=self.input_shape[0], height=self.input_shape[1], depth=self.input_shape[2], init_var=index_net, classes=2)
            #model = SmallerVGGNet.build_small_2d_deeper_binary(width=self.input_shape[0], height=self.input_shape[1], depth=self.input_shape[2], init_var=index_net)
            #model = SmallerVGGNet.build_small_2d_deeper_2_binary(width=self.input_shape[0], height=self.input_shape[1], depth=self.input_shape[2], init_var=index_net)
            #model = SmallerVGGNet.build_small_2d_v4_binary(width=self.input_shape[0], height=self.input_shape[1], depth=self.input_shape[2], init_var=index_net)
                
            model = SmallerVGGNet.build_alexNet(width=self.input_shape[0], height=self.input_shape[1], depth=self.input_shape[2])
            
        #opt = RAdam(total_steps=5000, warmup_proportion=0.1, min_lr=1e-5)
        opt = Adam(lr=self.init_lr, decay=(self.init_lr/ self.epochs))
        #opt = SGD(lr=self.init_lr)
        
        if self.loss_function == 'sparse_categorical_crossentropy': 
            model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
        
        if self.loss_function == 'w_categorical_crossentropy': 
            ncce = functools.partial(w_categorical_crossentropy, weights=self.custom_loss_weight)
            ncce.__name__ ='w_categorical_crossentropy'
            model.compile(loss=ncce, optimizer=opt, metrics=['accuracy'])
        
        # Salvo per ogni epoca il modello
        filepath_foreach_epoch = self.output_folder + 'models/model_foreach_epoch/' + "walk_" + str(index_walk) + "/net_" + str(index_net) + "/"
        
        #filename = self.output_folder + 'models/walk_' + str(index_walk) + '_net_' + str(index_net) + '.model'

        # creo la cartella di output per gli esperimenti
        if not os.path.isdir(filepath_foreach_epoch):
            os.makedirs(filepath_foreach_epoch)

        if not os.path.isdir(self.output_folder + 'models/'):
            os.makedirs(self.output_folder + 'models/')

        other_metrics = Metrics(training_set=training_set, validation_set=validation_set, test_set=test_set, 
                                base_path = self.base_path,
                                output_folder=self.output_folder, 
                                iperparameters=self.iperparameters, 
                                index_walk=index_walk, 
                                index_net=index_net, 
                                number_of_epochs=self.epochs, 
                                number_of_nets=self.number_of_nets,
                                return_multiplier=self.return_multiplier,
                                use_probabilities=self.use_probabilities
                            )

        
        # Se voglio salvare la storia dei modelli creo la callback 
        if self.save_model_history == True: 
            filename_epoch =  filepath_foreach_epoch + "epoch_{epoch:02d}.model"
            # period indica ogni quanto salvare il modello 
            checkpoint = ModelCheckpoint(filename_epoch, monitor='val_acc', verbose=1, save_best_only=False, save_weights_only=False, period=self.model_history_period)
            #earlystop = EarlyStopping(monitor='val_acc', mode='max', min_delta=5, patience=1000)
            
            callbacks_list = [checkpoint, other_metrics]
            #callbacks_list = [other_metrics]#, earlystop]

            # BUG TRASCENDENTALE -> validation_data -> get_x(normalized=False)
            H = model.fit(  x=training_set.get_x(normalized=True, multivariate=self.iperparameters['multivariate']), 
                            y=training_set.get_y(type="categorical", referred_to='next_day'), 
                            batch_size=self.bs, 
                            validation_data=(validation_set.get_x(normalized=True, multivariate=self.iperparameters['multivariate']), validation_set.get_y(type="categorical", referred_to='next_day')), 
                            epochs=self.epochs, 
                            verbose=1, 
                            callbacks=callbacks_list)
       
        # Non voglio salvare la storia dei modelli ogni tot epoche, salvo solo l'ultimo
        if self.save_model_history == False:
            callbacks_list = [other_metrics]
            # BUG TRASCENDENTALE -> validation_data -> get_x(normalized=False)
            H = model.fit(  x=training_set.get_x(normalized=True, multivariate=self.iperparameters['multivariate']), 
                            y=training_set.get_y(type="categorical", referred_to='next_day'), 
                            batch_size=self.bs, 
                            validation_data=(validation_set.get_x(normalized=True, multivariate=self.iperparameters['multivariate']), validation_set.get_y(type="categorical")), 
                            epochs=self.epochs, 
                            verbose=1, 
                            callbacks=callbacks_list)#,
                            #class_weight={0:1,1:1,2:1})
             
            #H = model.fit(a, training_set['y'], batch_size=self.bs, validation_data=(validation_set['x'], validation_set['y']), epochs=self.epochs, verbose=1,callbacks=callbacks_list)#, sample_weight=weights)

        # Salvo le loss
        df_acc_loss = pd.DataFrame()
        df_acc_loss['epoch'] = range(1, len(H.history['loss']) + 1)

        df_acc_loss['training_loss'] = H.history['loss']
        df_acc_loss['validation_loss'] = H.history['val_loss']
        df_acc_loss['test_loss'] = H.history['test_loss']
        
        df_acc_loss['training_accuracy'] = H.history['accuracy']
        df_acc_loss['validation_accuracy'] = H.history['val_accuracy']
        df_acc_loss['test_accuracy'] = H.history['test_accuracy']

        path = self.output_folder + 'predictions/loss_during_training/' + 'walk_' + str(index_walk) + '/' 

        filename = path + 'net_' + str(index_net) + '.csv'
        create_folder(path)
        df_acc_loss.to_csv(filename, header=True, index=False)

        return model, H



    def train_1D(self, training_set, validation_set, test_set, index_net, index_walk):  
        model = ''

        # univariate
        #model = SmallerVGGNet.build_small_1d(width=self.input_shape[0], height=self.input_shape[1], init_var=index_net)
        model = SmallerVGGNet.build_alexNet_1D(width=self.input_shape[0], height=self.input_shape[1])

        #opt = RAdam(total_steps=5000, warmup_proportion=0.1, min_lr=1e-5)
        opt = Adam(lr=self.init_lr, decay=(self.init_lr/ self.epochs))
        #opt = SGD(lr=self.init_lr)
        
        if self.loss_function == 'sparse_categorical_crossentropy': 
            model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
        
        if self.loss_function == 'w_categorical_crossentropy': 
            ncce = functools.partial(w_categorical_crossentropy, weights=self.custom_loss_weight)
            ncce.__name__ ='w_categorical_crossentropy'
            model.compile(loss=ncce, optimizer=opt, metrics=['accuracy'])
        
        # Salvo per ogni epoca il modello
        filepath_foreach_epoch = self.output_folder + 'models/model_foreach_epoch/' + "walk_" + str(index_walk) + "/net_" + str(index_net) + "/"
        
        #filename = self.output_folder + 'models/walk_' + str(index_walk) + '_net_' + str(index_net) + '.model'

        # creo la cartella di output per gli esperimenti
        if not os.path.isdir(filepath_foreach_epoch):
            os.makedirs(filepath_foreach_epoch)

        if not os.path.isdir(self.output_folder + 'models/'):
            os.makedirs(self.output_folder + 'models/')

        other_metrics = Metrics(training_set=training_set, validation_set=validation_set, test_set=test_set, 
                                base_path = self.base_path,
                                output_folder=self.output_folder, 
                                iperparameters=self.iperparameters, 
                                index_walk=index_walk, 
                                index_net=index_net, 
                                number_of_epochs=self.epochs, 
                                number_of_nets=self.number_of_nets,
                                return_multiplier=self.return_multiplier,
                                use_probabilities=self.use_probabilities
                            )

        
        # Se voglio salvare la storia dei modelli creo la callback 
        if self.save_model_history == True: 
            filename_epoch =  filepath_foreach_epoch + "epoch_{epoch:02d}.model"
            # period indica ogni quanto salvare il modello 
            checkpoint = ModelCheckpoint(filename_epoch, monitor='val_acc', verbose=1, save_best_only=False, save_weights_only=False, period=self.model_history_period)
            #earlystop = EarlyStopping(monitor='val_acc', mode='max', min_delta=5, patience=1000)
            
            callbacks_list = [checkpoint, other_metrics]
            #callbacks_list = [other_metrics]#, earlystop]

            # BUG TRASCENDENTALE -> validation_data -> get_x(normalized=False)
            H = model.fit(  x=training_set.get_x(normalized=True, multivariate=self.iperparameters['multivariate']), 
                            y=training_set.get_y(type="categorical", referred_to='next_day'), 
                            batch_size=self.bs, 
                            validation_data=(validation_set.get_x(normalized=True, multivariate=self.iperparameters['multivariate']), validation_set.get_y(type="categorical", referred_to='next_day')), 
                            epochs=self.epochs, 
                            verbose=1, 
                            callbacks=callbacks_list)
       
        # Non voglio salvare la storia dei modelli ogni tot epoche, salvo solo l'ultimo
        if self.save_model_history == False:
            callbacks_list = [other_metrics]
            # BUG TRASCENDENTALE -> validation_data -> get_x(normalized=False)
            H = model.fit(  x=training_set.get_x(normalized=True, multivariate=self.iperparameters['multivariate']), 
                            y=training_set.get_y(type="categorical", referred_to='next_day'), 
                            batch_size=self.bs, 
                            validation_data=(validation_set.get_x(normalized=True, multivariate=self.iperparameters['multivariate']), validation_set.get_y(type="categorical")), 
                            epochs=self.epochs, 
                            verbose=1, 
                            callbacks=callbacks_list)#,
                            #class_weight={0:1,1:1,2:1})
             
            #H = model.fit(a, training_set['y'], batch_size=self.bs, validation_data=(validation_set['x'], validation_set['y']), epochs=self.epochs, verbose=1,callbacks=callbacks_list)#, sample_weight=weights)

        # Salvo le loss
        df_acc_loss = pd.DataFrame()
        df_acc_loss['epoch'] = range(1, len(H.history['loss']) + 1)

        df_acc_loss['training_loss'] = H.history['loss']
        df_acc_loss['validation_loss'] = H.history['val_loss']
        df_acc_loss['test_loss'] = H.history['test_loss']
        
        df_acc_loss['training_accuracy'] = H.history['accuracy']
        df_acc_loss['validation_accuracy'] = H.history['val_accuracy']
        df_acc_loss['test_accuracy'] = H.history['test_accuracy']

        path = self.output_folder + 'predictions/loss_during_training/' + 'walk_' + str(index_walk) + '/' 

        filename = path + 'net_' + str(index_net) + '.csv'
        create_folder(path)
        df_acc_loss.to_csv(filename, header=True, index=False)

        return model, H

    def get_set_1D(self, perc, dataset, walk=0, dates=[], balance_binary=False):
        # parametri hardcoded
        size_of_feature = 227 #227
        resolutions = 1

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

        market = Market(dataset=dataset)
        
        # dataframe che userò per i sample
        five_min_df = market.get()

        #market = market.get_binary_labels(freq='1d', columns=['delta_current_day', 'delta_next_day', 'close'], thr=self.iperparameters['thr_binary_labeling']).reset_index()
        #market = market.get_binary_labels_volatility(freq='1d', columns=['delta_current_day', 'delta_next_day', 'close'], thr=self.iperparameters['thr_binary_labeling']).reset_index()
        market = market.get_binary_labels_volatility_7_days(freq='1d', columns=['delta_current_day', 'delta_next_day', 'close'], thr=self.iperparameters['thr_binary_labeling'], days=7).reset_index()
        
        market = Market.get_df_by_data_range(df=market.copy(), start_date=dates[walk][0], end_date=dates[walk][1])


        start_date = pd.to_datetime(dates[walk][0]) + timedelta(hours=23)
        end_date = pd.to_datetime(dates[walk][1]) + timedelta(hours=23)
        five_min_df = Market.get_df_by_data_range(df=five_min_df.copy(), start_date=start_date, end_date=end_date)

        x = np.ones(shape=(market.shape[0] - size_of_feature, resolutions * size_of_feature))

        # tolgo i primi TOT elementi della lista
        date_list       = market['date_time'].tolist()[size_of_feature:]
        y_current_day   = market['label_current_day'].tolist()[size_of_feature:]
        y_next_day      = market['label_next_day'].tolist()[size_of_feature:]
        delta_next_day  = market['delta_next_day'].tolist()[size_of_feature:]
        delta_curr_day  = market['delta_current_day'].tolist()[size_of_feature:]
        close           = market['close'].tolist()[size_of_feature:]

        date_list.reverse()
        y_current_day.reverse()
        y_next_day.reverse()
        delta_next_day.reverse()
        delta_curr_day.reverse()
        close.reverse()
        
        for i, date in enumerate(date_list): 
            subset_one_h = five_min_df.loc[five_min_df['date_time'] <= date + timedelta(hours=23)]
            
            feature = np.array(subset_one_h['delta_current_day_percentage'].tolist()[-size_of_feature:])


            if y_next_day[i] == 0:
                feature = np.multiply(feature, pattern)

            x[i] = feature

        set = Set(date_time=date_list, x=x, y_current_day=y_current_day, y_next_day=y_next_day, delta_current_day=delta_curr_day, delta_next_day=delta_next_day, close=close, balance_binary=balance_binary)
        return set


    def get_set_1D_debug(self, dataset, walk=0, dates=[], balance_binary=False):
        market = Market(dataset=self.iperparameters['predictions_dataset'])
        #market = market.get_binary_labels(freq='1d', columns=['delta_current_day', 'delta_next_day', 'close'], thr=self.iperparameters['thr_binary_labeling']).reset_index()
        #market = market.get_binary_labels_volatility(freq='1d', columns=['delta_current_day', 'delta_next_day', 'close'], thr=self.iperparameters['thr_binary_labeling']).reset_index()
        market = market.get_binary_labels_volatility_7_days(freq='1d', columns=['delta_current_day', 'delta_next_day', 'close'], thr=self.iperparameters['thr_binary_labeling'], days=7).reset_index()
        
        market = Market.get_df_by_data_range(df=market.copy(), start_date=dates[walk][0], end_date=dates[walk][1])

        dataset = Market(dataset=dataset)
        # self.dataset contiene un dataset modificato a monte
        one_h = dataset.group(freq='1h', nan=False)
        one_h = Market.get_df_by_data_range(df=one_h.copy(), start_date=dates[walk][0], end_date=dates[walk][1])
        x = np.zeros(shape=(market.shape[0] - self.input_shape[1], self.input_shape[0], self.input_shape[1]))

        date_list       = market['date_time'].tolist()[self.input_shape[1]:]

        y_current_day   = market['label_current_day'].tolist()[self.input_shape[1]:]
        y_next_day      = market['label_next_day'].tolist()[self.input_shape[1]:]
        delta_next_day  = market['delta_next_day'].tolist()[self.input_shape[1]:]
        delta_curr_day  = market['delta_current_day'].tolist()[self.input_shape[1]:]
        close           = market['close'].tolist()[self.input_shape[1]:]

        date_list.reverse()

        for i, date in enumerate(date_list): 
            
            #print(date.strftime('%Y-%m-%d'))
            subset_one_h    = one_h.loc[one_h['date_time'] < date]
            
            x[i] = [
                subset_one_h['delta_current_day'].tolist()[-self.input_shape[1]:],
                subset_one_h['delta_current_day'].tolist()[-self.input_shape[1]:], 
                subset_one_h['delta_current_day'].tolist()[-self.input_shape[1]:], 
                subset_one_h['delta_current_day'].tolist()[-self.input_shape[1]:]
            ]

        set = Set(date_time=date_list, x=x, y_current_day=y_current_day, y_next_day=y_next_day, delta_current_day=delta_curr_day, delta_next_day=delta_next_day, close=close, balance_binary=balance_binary)
        return set

    '''
    ' Ottengo il training set, 
    ' leggendo tutte le immagini con rispettiva label sia dal mercato di base
    ' che da tutte le companies associate. 
    ' Il metodo è stato pensato per usare le immagini di sp500 + le companies
    ' corrispondenti del mercato sp500
    '''
    def get_train(self, training_set):
        # dataset completo, in cui metterò tutti mercati + label 
        df = pd.DataFrame()

        # per ogni datasets passato in input alla rete, leggo le immagini e label ad esso associati
        # e poi concateno tutto dentro df
        for index, input_dataset in enumerate(self.input_datasets):

            # e' importante che le path delle immagini siano passate nello stesso ordine dei datasets
            # per evitare che ci siano incompatibilità tra data_range e immagini generate
            # leggo prima la lista delle immagini e credo un df associato. 
            images_list = self.images_filename_reader(self.images_full_paths[index])
            # creo il dataframe associato
            images_list_df = pd.DataFrame(images_list, columns=['date_time'])
            # aggiungo una colonna contenente la path dell'immagine completa 
            images_list_df['images_path'] = self.images_full_paths[index] + images_list_df['date_time'] + '.png' # .png
            # converto in pd.to_datetime il campo data_time così posso mergiarlo successivamente con il df delle label
            images_list_df['date_time'] = pd.to_datetime(images_list_df['date_time'])

            # leggo ora il dataframe associato a quel datasets
            market = Market(dataset=input_dataset)

            # prendo la label del giorno successivo
            #label_df = market.get_binary_labels(freq='1d', columns=['delta_next_day', 'delta_current_day', 'close'], thr=self.iperparameters['thr_binary_labeling']).reset_index()
            #label_df = market.get_binary_labels_volatility(freq='1d', columns=['delta_next_day', 'delta_current_day', 'close'], thr=self.iperparameters['thr_binary_labeling']).reset_index()
            label_df = market.get_binary_labels_volatility_7_days(freq='1d', columns=['delta_current_day', 'delta_next_day', 'close'], thr=self.iperparameters['thr_binary_labeling'], days=7).reset_index()
            
            # effettuo il merge tra il df del mercato e quello generato con le immagini 
            # in questo modo avrò solo le date disponibili effettivamente per leggere le immagini
            # e non avrò errori in fase di lettura
            df_merge = pd.merge(images_list_df, label_df, on="date_time")
            # concateno tutto dentro il df globale su cui farò poi le operazioni di data range e lettura immagini
            df = pd.concat([df, df_merge])

        # prendo solo i giorni che sono compresi nel training set
        df = self.get_df_by_data_range(df, training_set[0], training_set[1])

        # credo il vettore di input
        x = np.zeros(shape=(df.shape[0], self.input_shape[0], self.input_shape[1], self.input_shape[2]))
        
        # creo il vettore di label prendendole direttamente dal dataframe
        y_next_day = df['label_next_day'].tolist()
        y_current_day = df['label_current_day'].tolist()
        delta_current_day = df['delta_current_day'].tolist()
        delta_next_day = df['delta_next_day'].tolist()
        close = df['close'].tolist()
        date_list = df['date_time'].tolist()

        # Resetto gli indici altrimenti salvo su x_train le stesse cose nelle prime 5k posizioni
        df = df.reset_index()

        # leggo le immagini e le metto dentro il vettore
        #for index, row in df.iterrows(): 
        for index, row in enumerate(df.itertuples(index=False)):  
            x[index] = cv2.imread(row.images_path)

       
        set = Set(date_time=date_list, x=x, y_current_day=y_current_day, y_next_day=y_next_day, delta_current_day=delta_current_day, delta_next_day=delta_next_day, close=close, balance_binary=True)
        
        return set
            

    '''
    ' Calcolo un x, y set per il mercato su cui si vogliono fare le predizioni
    ' Può essere utilizzato sia per calcolare il training (solo su sp500 ad esempio)
    ' che per calcolare validation e test set. date_set è il data range di riferimento
    ' passato come parametro alla classe. 
    '''
    def get_set_predictions_dataset(self, date_set):   
        # leggo ora il dataframe associato a quel datasets
        market = Market(dataset=self.predictions_dataset)

        #df = market.get_binary_labels(freq='1d', columns=['delta_next_day', 'delta_current_day', 'open', 'close'], thr=self.iperparameters['thr_binary_labeling']).reset_index()
        #df = market.get_binary_labels_volatility(freq='1d', columns=['delta_next_day', 'delta_current_day', 'open', 'close'], thr=self.iperparameters['thr_binary_labeling']).reset_index()
        df = market.get_binary_labels_volatility_7_days(freq='1d', columns=['delta_current_day', 'delta_next_day', 'open', 'close'], thr=self.iperparameters['thr_binary_labeling'], days=7).reset_index()

        # prendo solo i giorni che mi servono
        df = self.get_df_by_data_range(df, date_set[0], date_set[1])

        #print(df)
        #input()
        # Uso __get_df_by_data_range per ottenere la lista dei nomi delle immagini per quei specifici range di date 
        # uso un data_range con i df e non un semplice for per generare tutti i nomi immagini per semplicità, in alcuni
        # giorni il mercato potrebbe essere chiuso e potrebbero non esserci i nomi delle immagini
        # uso la colonna ['date_time'] per avere la data (nome della pic) e converto tutto con .tolist() per avere una lista
        date_list = df['date_time'].tolist()

        x = np.zeros(shape=(len(date_list), self.input_shape[0], self.input_shape[1], self.input_shape[2]))
        
        y_next_day = df['label_next_day'].tolist()
        y_current_day = df['label_current_day'].tolist()
        delta_current_day = df['delta_current_day'].tolist()
        delta_next_day = df['delta_next_day'].tolist()
        close = df['close'].tolist()
        date_list = df['date_time'].tolist()

        #delta = []
        # ora devo riempire gli array x con le immagini vere e proprie che devo leggere
        # per fare ciò creo preventivamente delle matrici di 0 con la lunghezza della lista e la shape delle immagini
        # quindi se sono 100 immagini, per 2 input_folder (magari due companies)
        # 40x40 rbg avrò  [100 * 2][40][40][3]
        for i in range(len(date_list)):
            if os.path.isfile(self.images_base_path + self.predictions_images_folder + date_list[i].strftime('%Y-%m-%d') + '.png'): # .png
                x[i] = cv2.imread(self.images_base_path + self.predictions_images_folder + date_list[i].strftime('%Y-%m-%d') + '.png') # .png
        
        set = Set(date_time=date_list, x=x, y_current_day=y_current_day, y_next_day=y_next_day, delta_current_day=delta_current_day, delta_next_day=delta_next_day, close=close, balance_binary=False)

        #return x, y, delta, close, date_list
        return set

    '''
    ' Con questo metodo ottengo il training, validation e test set da passare al metodo train()
    ' Di base alla classe viene passata una matrice con [ [data_inizio, data_fine], [data_inizio, data_fine]]
    ' dove ogni elemento del'array è una coppia di inizio fine set per walk. 
    ' In questo metodo viene passato una coppia di ciascuno (tranining, validation, test), leggo 
    ' le immagini a seconda del data range e li restituisco come lista
    ' uso la colonna ['label'] per i valori y 
    '''
    def get_train_val_test(self, training_set, validation_set, test_set):

        # calcolo il training prendendo il mercato di riferimento + altre compagnie
        training = self.get_train(training_set=training_set)
        
        # calcolo validation set prendendo come samples solo il mercato di riferimento
        validation = self.get_set_predictions_dataset(date_set=validation_set)

        # calcolo il test set 
        test = self.get_set_predictions_dataset(date_set=test_set)
        
        return training, validation, test        
    
  
    '''
    ' Lancio l'esecuzione della rete
    ' prima di tutto convalido i dati passati come parametri, dopodichè 
    ' ciclo per ogni walk e per ogni rete e effettuo il fit del modello e calcolo l'acc in test
    '''
    def run_2D(self, start_index_walk=None, gpu_id=0):
        # salvo la prima parte dei log
        self.start_log()
        
        # Per ogni walk calcolo train, val e test set (x,y)
        for index_walk in range(self.number_of_walks):
            self.add_line_log("Inizio walk n°" + str(index_walk)) 

            self.history_array = []

            if start_index_walk is not None:                                                                         
                if index_walk <= start_index_walk: 
                    print("Skip walk number: ", index_walk)
                    continue
            
            # [x, y, date_time (only val e test), delta, close]
            
            training, validation, test = self.get_train_val_test(training_set=self.training_set[index_walk],
                                                                                        validation_set=self.validation_set[index_walk],
                                                                                        test_set=self.test_set[index_walk])

            
            
            # per ogni rete effettuo il fit del modello e salvo .model, .pkl ed i plots
            for index_net in range(self.number_of_nets):
                
                # Debug
                print("TRAINING - INDEX NET: ", str(index_net), " INDEX WALK: " + str(index_walk))

                # Ottengo dimensione del test, e inizializzo una variabile per calcolarmi l'accuracy in test
                #test_size = x_test.shape[0]
                #test_accuracy = 0
                #results = np.zeros(shape=(test_size))

                # effettuo il training del modello
                model, H = self.train_2D(training_set=training, validation_set=validation, test_set=test, index_net=index_net, index_walk=index_walk)
                #model, H = self.train_2D(training_set=training_set, validation_set=validation_set, test_set=test_set, index_net=index_net, index_walk=index_walk)

                
                # Salvo il modello ed i grafici della rete
                #self.save_plots(H=H, index_walk=index_walk, index_net=index_net, type="val")
                #self.save_plots(H=H, index_walk=index_walk, index_net=index_net, type="test")

                self.save_models(model=model, H=H, index_walk=index_walk, index_net=index_net)
                
                K.clear_session()

            #self.get_avg_history(walk=index_walk, close=validation.get_close(), type="val")
            #self.get_avg_history(walk=index_walk, close=test.get_close(), type="test")

            if index_walk < (self.number_of_walks - 1):
                is_last = 0
            else:
                is_last = 1

            self.add_line_log("Fine walk n°" + str(index_walk))
            self.save_last_walk_log(index_walk=index_walk, is_last=is_last, gpu_id=gpu_id)   

        # salvo l'ultima parte del log
        self.end_log()
    
    
    
    
    
    '''
    '
    '''
    def run_1D(self, dataset, start_index_walk=None, gpu_id=0, perc=0):
        # salvo la prima parte dei log
        self.start_log()
        
        # Per ogni walk calcolo train, val e test set (x,y)
        for index_walk in range(self.number_of_walks):
            self.add_line_log("Inizio walk n°" + str(index_walk)) 

            self.history_array = []

            if start_index_walk is not None:                                                                         
                if index_walk <= start_index_walk: 
                    print("Skip walk number: ", index_walk)
                    continue
            print("Loading datasets....")
            training_set = self.get_set_1D(dataset=dataset, perc=perc, walk=index_walk, dates=self.training_set, balance_binary=True)
            validation_set = self.get_set_1D(dataset=dataset, perc=perc, walk=index_walk, dates=self.validation_set, balance_binary=False)
            test_set = self.get_set_1D(dataset=dataset, perc=perc, walk=index_walk, dates=self.test_set, balance_binary=False)

            
            
            # per ogni rete effettuo il fit del modello e salvo .model, .pkl ed i plots
            for index_net in range(self.number_of_nets):
                
                # Debug
                print("TRAINING - INDEX NET: ", str(index_net), " INDEX WALK: " + str(index_walk))
                # effettuo il training del modello
                model, H = self.train_1D(training_set=training_set, validation_set=validation_set, test_set=test_set, index_net=index_net, index_walk=index_walk)

                self.save_models(model=model, H=H, index_walk=index_walk, index_net=index_net)
                
                K.clear_session()

            if index_walk < (self.number_of_walks - 1):
                is_last = 0
            else:
                is_last = 1

            self.add_line_log("Fine walk n°" + str(index_walk))
            self.save_last_walk_log(index_walk=index_walk, is_last=is_last, gpu_id=gpu_id)   

        # salvo l'ultima parte del log
        self.end_log()
    