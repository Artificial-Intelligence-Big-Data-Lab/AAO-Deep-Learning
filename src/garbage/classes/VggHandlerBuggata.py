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

#matplotlib.use("Agg")
import matplotlib.pyplot as plt

from classes.Market import Market
from classes.StopLossCustom import stop_loss_custom
from classes.Set import Set

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

from classes.Metrics import Metrics

'''
' Questa classe serve per lanciare e gestire 
' la configurazione della VGG16-Small creata da
' Silvio Barra 
' @author Andrea corriga
'''
class VggHandler:
    
    # path iniziale da dove leggere le immagini da passare alla rete
    __images_base_path = '../images/'

    # qui verrà inserito il path completo una volta passato il parametro in input
    __images_full_paths = []

    # i datasets usati input per il training ed il validation
    input_datasets = []

    # il datasets che voglio predire
    predictions_dataset = ''

    # cartella con le immagini per il dataset su cui voglio fare le predizioni
    predictions_images_folder = ''

    # path dove verranno salvati gli esperimenti
    __output_base_path = '../experiments/'
    __output_folder = ''

    # dataframe con label del giorno
    __label_df = pd.DataFrame()

    # dataframe con date_time e label del giorno dopo [la uso per allenare la rete]
    __date_next_day_label_df = pd.DataFrame()

    # dataframe con date_time e label del giorno corrente [la uso per stampare le predizioni per ogni singola rete]
    __date_current_day_label_df = pd.DataFrame()

    # configurazione base della rete
    __number_of_nets = 20
    __epochs = 200
    __init_lr = 0.001
    __bs = 32

    # parametri passati alla rete
    __training_set = []
    __validation_set = []
    __test_set = []
    __number_of_walks = 0
    __input_shape = (40, 40, 3)

    __save_pkl = False
    __save_model_history = False
    __model_history_period = __epochs

    __history_array = []

    __acc_name = "accuracy"
    __val_acc_name = "val_accuracy"

    def get_avg_history(self, walk, close):

        array = self.__history_array
        denom = len(array)

        tot_return = np.zeros(len(array[0].history["return"]))
        tot_return2 = np.zeros(len(array[0].history["return_bis"]))
        long_val_acc = np.zeros(len(array[0].history["long_val_acc"]))
        short_val_acc = np.zeros(len(array[0].history["short_val_acc"]))
        hold_val_acc = np.zeros(len(array[0].history["hold_val_acc"]))
        acc = np.zeros(len(array[0].history[self.__acc_name]))
        val_acc = np.zeros(len(array[0].history[self.__val_acc_name]))
        loss = np.zeros(len(array[0].history["loss"]))
        val_loss = np.zeros(len(array[0].history["val_loss"]))

        long_perc = np.zeros(len(array[0].history["long_perc"]))
        short_perc = np.zeros(len(array[0].history["short_perc"]))
        hold_perc = np.zeros(len(array[0].history["hold_perc"]))

        for i, val in enumerate(array):
            tot_return = np.add(val.history["return"], tot_return)
            tot_return2 = np.add(val.history["return_bis"], tot_return2)
            long_val_acc = np.add(val.history["long_val_acc"], long_val_acc)
            short_val_acc = np.add(val.history["short_val_acc"], short_val_acc)
            #hold_val_acc = np.add(val.history["hold_val_acc"], hold_val_acc)
            acc = np.add(val.history[self.__acc_name], acc)
            val_acc = np.add(val.history[self.__val_acc_name], val_acc)
            loss = np.add(val.history["loss"], loss)
            val_loss = np.add(val.history["val_loss"], val_loss)

            long_perc = np.add(val.history["long_perc"], long_perc)
            short_perc = np.add(val.history["short_perc"], short_perc)
            hold_perc= np.add(val.history["hold_perc"], hold_perc)
            
        tot_return = np.true_divide(tot_return, denom)
        tot_return2 = np.true_divide(tot_return2, denom)
        long_val_acc = np.true_divide(long_val_acc, denom)
        short_val_acc = np.true_divide(short_val_acc, denom)
        #hold_val_acc = np.true_divide(hold_val_acc, denom)
        acc = np.true_divide(acc, denom)
        val_acc = np.true_divide(val_acc, denom)
        loss = np.true_divide(loss, denom)
        val_loss = np.true_divide(val_loss, denom)

        long_perc = np.true_divide(long_perc, denom)
        short_perc = np.true_divide(short_perc, denom)
        hold_perc = np.true_divide(hold_perc, denom)
        
        full_plots_path = self.__output_folder + 'accuracy_loss_plots/average/'

        # Se non esiste la cartella, la creo
        if not os.path.isdir(full_plots_path):
            os.makedirs(full_plots_path)

        plt.figure(figsize=(15,12))
        # plot the training loss and accuracy
        plt.style.use("ggplot")
        plt.rcParams['axes.prop_cycle'] = cycler(color='bgrcmyk')
        plt.subplot(2, 2, 1)
        plt.xlabel("Epoch #")
        plt.ylabel("Dollars")
        
        plt.plot(np.arange(0, self.__epochs), tot_return, label="Return Long+Short (USD)")
        #plt.plot(np.arange(0, self.__epochs), tot_return2, label="Return Long Only(USD)")
        plt.plot(np.arange(0, self.__epochs), np.full((self.__epochs, 1), (close[-1] - close[0]) *50 ), label="Return Buy&Hold (USD)")
        plt.plot(np.arange(0, self.__epochs), np.zeros(self.__epochs), color="black")
        plt.legend(loc="upper left")

        plt.subplot(2, 2, 2)
        plt.xlabel("Epoch #")
        plt.ylabel("Per-class val. precision")

        plt.plot(np.arange(0, self.__epochs), long_val_acc, label="Long precision")
        plt.plot(np.arange(0, self.__epochs), short_val_acc, label="Short precision")
        #plt.plot(np.arange(0, self.__epochs), hold_val_acc, label="Hold precision")
        plt.plot(np.arange(0, self.__epochs), np.full((self.__epochs, 1), 0.5), color="black")
        plt.legend(loc="upper left")

        plt.subplot(2, 2, 3)
        plt.xlabel("Epoch #")
        plt.ylabel("Global accuracy")

        plt.plot(np.arange(0, self.__epochs), acc, label="Training accuracy")
        #plt.plot(np.arange(0, self.__epochs), H.history["accuracy"], label="train_acc")
        plt.plot(np.arange(0, self.__epochs), val_acc, label="Validation accuracy")
        #plt.title("Training Loss")
        plt.legend(loc="upper left")

        plt.subplot(2, 2, 4)
        #plt.xlabel("Epoch #")
        #plt.ylabel("Global loss")
        #plt.plot(np.arange(0, self.__epochs), loss, label="Training loss")
        #plt.plot(np.arange(0, self.__epochs), val_loss, label="Validation loss")
        plt.xlabel("Epoch #")
        plt.ylabel("Perc Operations")
        plt.plot(np.arange(0, self.__epochs),long_perc, label='% of long operations')
        plt.plot(np.arange(0, self.__epochs), short_perc, label="% of short operations")
        plt.plot(np.arange(0, self.__epochs), hold_perc, label="% of hold operations")
        #plt.title("Training Loss")
        plt.legend(loc="upper left")

        plt.savefig(full_plots_path + 'walk_' + str(walk) + '_average.png')

        plt.close('all')

        return

    '''
    ' Funzione di appoggio, leggo tutte le
    ' immagini dentro una cartella. La uso per 
    ' prendere i nomi delle immagini che unirò successivamente
    '''
    def __images_filename_reader(self, path):
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
    def __get_df_by_data_range(self, df, start_date, end_date):
        # Search mask
        mask = (df['date_time'] >= start_date) & (df['date_time'] < end_date)
        # Get the subset of sp500
        return df.loc[mask]

    '''
    ' @author Silvio Barra
    ' @edited_by Andrea Corriga
    ' Salvo il modello e la history generata nella loro rispettiva cartella
    '''
    def __save_models(self, model, H, index_walk, index_net):
        full_models_path = self.__output_folder + 'models/'
        full_histories_path = self.__output_folder + 'histories/'

        # Se non esiste la cartella, la creo
        if not os.path.isdir(full_models_path):
            os.makedirs(full_models_path)

        if self.__save_pkl == True: 
            if not os.path.isdir(full_histories_path):
                os.makedirs(full_histories_path)
        
        if self.__save_pkl == True:
            # Salvo la history
            f = open(full_histories_path + 'walk_' + str(index_walk) + '_net_' + str(index_net) + '.pkl', "wb")
            f.write(pickle.dumps(H))
            f.close()

        # Salvo il modello
        model.save(full_models_path + 'walk_' + str(index_walk) + '_net_' + str(index_net) + '.model')
    
    '''
    ' Salvo i plot di accuracy e loss di training e validation
    ' Li salvo in una sottocartella per ogni walk, in modo da avere 
    ' dentro la stessa cartella tutti i risultati delle reti per walk
    '''
    def __save_plots(self, H, index_walk, index_net):

        self.__history_array.append(H)

        full_plots_path = self.__output_folder + 'accuracy_loss_plots/walk_' + str(index_walk) + '/'

        # Se non esiste la cartella, la creo
        if not os.path.isdir(full_plots_path):
            os.makedirs(full_plots_path)

        plt.figure(figsize=(15,12))
        # plot the training loss and accuracy
        plt.style.use("ggplot")
        plt.rcParams['axes.prop_cycle'] = cycler(color='bgrcmyk')
        plt.subplot(2, 2, 1)
        plt.xlabel("Epoch #")
        plt.ylabel("Global return")
        
        plt.plot(np.arange(0, self.__epochs), H.history["return"], label="Return Long+Short (USD)")
        #plt.plot(np.arange(0, self.__epochs), H.history["return_bis"], label="Return Long Only (USD)")
        plt.plot(np.arange(0, self.__epochs), np.full((self.__epochs, 1), H.history["return_bh"][0]), label="Return Buy&Hold (USD)")
        plt.plot(np.arange(0, self.__epochs), np.zeros(self.__epochs), color="black")
        plt.legend(loc="upper left")

        plt.subplot(2, 2, 2)
        plt.xlabel("Epoch #")
        plt.ylabel("Per-class val. precision")

        plt.plot(np.arange(0, self.__epochs), H.history["long_val_acc"], label="Long precision")
        plt.plot(np.arange(0, self.__epochs), H.history["short_val_acc"], label="Short precision")
        #plt.plot(np.arange(0, self.__epochs), H.history["hold_val_acc"], label="Hold precision")
        plt.plot(np.arange(0, self.__epochs), np.full((self.__epochs, 1), 0.5), color="black")
        plt.legend(loc="upper left")

        plt.subplot(2, 2, 3)
        plt.xlabel("Epoch #")
        plt.ylabel("Global accuracy")

        plt.plot(np.arange(0, self.__epochs), H.history[self.__acc_name], label="Training accuracy")
        #plt.plot(np.arange(0, self.__epochs), H.history["accuracy"], label="train_acc")
        plt.plot(np.arange(0, self.__epochs), H.history[self.__val_acc_name], label="Validation accuracy")
        #plt.title("Training Loss")
        plt.legend(loc="upper left")

        plt.subplot(2, 2, 4)
        #plt.xlabel("Epoch #")
        #plt.ylabel("Global loss")
        #plt.plot(np.arange(0, self.__epochs), H.history["loss"], label="Training loss")
        #plt.plot(np.arange(0, self.__epochs), H.history["val_loss"], label="Validation loss")

        plt.xlabel("Epoch #")
        plt.ylabel("Perc Operations")
        plt.plot(np.arange(0, self.__epochs), H.history["long_perc"], label="% of long operations")
        plt.plot(np.arange(0, self.__epochs), H.history["short_perc"], label="% of short operations")
        plt.plot(np.arange(0, self.__epochs), H.history["hold_perc"], label="% of hold operations")

        #plt.title("Training Loss")
        plt.legend(loc="upper left")

        plt.savefig(full_plots_path + 'net_' + str(index_net) + '.png')

        plt.close('all')

    '''
    ' Salvo un file di log.txt in cui inserisco
    ' data di inizio esecuzione e tutta la configurazione della rete
    ' salvo anche i dati relativi al training, validation e test set
    '''
    def __start_log(self):
        f=open(self.__output_folder + 'log.txt','a')
        
        f.write("Net Start: " + str(datetime.datetime.now()) + "\n")

        f.write("\n")

        f.write("Number of nets: " + str(self.__number_of_nets) + "\n")
        f.write("Epochs: " + str(self.__epochs) + "\n")
        f.write("Init lr: " + str(self.__init_lr) + "\n")
        f.write("Bs: " + str(self.__bs) + "\n")

        f.write("\n")

        f.write("Number of walks: " + str(self.__number_of_walks) + "\n")
        
        f.write("Walk \t\t Training \t\t\t Validation \t\t\t Test\n")
        
        for index_walk in range(self.__number_of_walks):
            f.write("Walk " + str(index_walk) + ": \t ")
            f.write("[" + self.__training_set[index_walk][0] + " - " + self.__training_set[index_walk][1] + "] \t ")
            f.write("[" + self.__validation_set[index_walk][0] + " - " + self.__validation_set[index_walk][1] + "] \t ")
            f.write("[" + self.__test_set[index_walk][0] + " - " + self.__test_set[index_walk][1] + "] \n")
        
        f.write("\n")
        f.close()

    '''
    ' Aggiunge a fine file di log quando si 
    ' è conclusa l'esecuzione della rete
    '''
    def __end_log(self):
        f=open(self.__output_folder + 'log.txt','a')
        
        f.write("\nNet Stop: " + str(datetime.datetime.now()) + "\n")
        f.write("\n")
        f.close()
    
    '''
    '
    '''
    def __add_line_log(self, line): 
        f=open(self.__output_folder + 'log.txt','a')
        f.write("["+ str(datetime.datetime.now()) +"] " + str(line) + "\n")
        f.close()

    '''
    ' Per recuperare l'esecuzione in caso di crash
    '''
    def __save_last_walk_log(self, index_walk, is_last): 
        f=open(self.__output_base_path + 'last_walk_log.txt','w')
        f.write(str(index_walk) + "\n" + str(is_last))
        f.close()
    '''
    ' @author Silvio Barra
    ' @edited_by Andrea Corriga
    ' Effettuo il training della rete
    '''
    #def __train_2D(self, training_set, validation_set, test_set, index_net, index_walk):
    def __train_2D(self, training_set, validation_set, test_set, index_net, index_walk):
        # binarizing labels
        #lb = LabelBinarizer()
        #y_train = lb.fit_transform(y_train)
        #y_val = lb.fit_transform(y_val)

        # [INFO] compiling model...
        #model = SmallerVGGNet.build_vgg16_2d_smaller(height=self.__input_shape[0], width=self.__input_shape[1], depth=self.__input_shape[2], classes=3, init_var=index_net)
        #model = SmallerVGGNet.build_vgg16(height=self.__input_shape[0], width=self.__input_shape[1], depth=self.__input_shape[2], classes=len(lb.classes_))
        #model = SmallerVGGNet.build_anse_2d(height=self.__input_shape[0], width=self.__input_shape[1], depth=self.__input_shape[2], init_var=index_net)
        #model = SmallerVGGNet.build_anse_v2(height=40, width=40, depth=3, init_var=index_net)
        model = SmallerVGGNet.build_small_2d(height=self.__input_shape[0], width=self.__input_shape[1], depth=self.__input_shape[2], init_var=index_net)

        #opt = RAdam(total_steps=5000, warmup_proportion=0.1, min_lr=1e-5)
        #opt = Adam(lr=self.__init_lr, decay=(self.__init_lr/ self.__epochs))
        opt = SGD(lr=self.__init_lr, momentum=0.9)
        
        #weights = np.abs(np.random.randn(len(y_train)) + 5)
        #class_weight = {0: 3, 1: 1, 2: 3}
        #weights = np.array(range(0, len(y_train)))
        #weights = np.concatenate(weights1, weights2)
        #weights = weights * 10 
        #weights = np.array(range(len(y_train), 0))
        #sess = K.get_session()
        #sl = stop_loss_custom(weights)
        #metrica = metrica(weights)
        
        #model.compile(loss=stop_loss_custom(weights), optimizer=opt, metrics=["accuracy"])
        model.compile(optimizer=opt, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        
        #model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])
        
        # model info
        #model.summary()
        #print(opt)

        #[INFO] training network...
        #filepath = self.__output_folder + 'models/epochs/' +"weights-improvement-{epoch:02d}-{val_acc:.2f}.hdf5"
        
        # Salvo per ogni epoca il modello
        filepath_foreach_epoch = self.__output_folder + 'models/model_foreach_epoch/' + "walk_" + str(index_walk) + "/net_" + str(index_net) + "/"
        
        #filename = self.__output_folder + 'models/walk_' + str(index_walk) + '_net_' + str(index_net) + '.model'

        # creo la cartella di output per gli esperimenti
        if not os.path.isdir(filepath_foreach_epoch):
            os.makedirs(filepath_foreach_epoch)

        if not os.path.isdir(self.__output_folder + 'models/'):
            os.makedirs(self.__output_folder + 'models/')

        #other_metrics = Metrics(validation_set=validation_set, test_set=test_set, output_folder=self.__output_folder, index_walk=index_walk, index_net=index_net, number_of_epochs=self.__epochs, number_of_nets=self.__number_of_nets)
        #other_metrics = Metrics(delta_val, output_folder=self.__output_folder, index_walk=index_walk, index_net=index_net, number_of_epochs=self.__epochs, number_of_nets=self.__number_of_nets)
        other_metrics = Metrics(validation_set=validation_set, test_set=test_set, output_folder=self.__output_folder, index_walk=index_walk, index_net=index_net, number_of_epochs=self.__epochs, number_of_nets=self.__number_of_nets)

        # Se voglio salvare la storia dei modelli creo la callback 
        if self.__save_model_history == True: 
            filename_epoch =  filepath_foreach_epoch + "epoch_{epoch:02d}.model"
            # period indica ogni quanto salvare il modello 
            checkpoint = ModelCheckpoint(filename_epoch, monitor='val_acc', verbose=1, save_best_only=False, save_weights_only=False, period=self.__model_history_period)
            
            callbacks_list = [checkpoint, other_metrics]
            H = model.fit(training_set.x, training_set.y, batch_size=self.__bs, validation_data=(validation_set.x, validation_set.y), epochs=self.__epochs, verbose=1, callbacks=callbacks_list)#, class_weight=class_weight)
       
        # Non voglio salvare la storia dei modelli ogni tot epoche, salvo solo l'ultimo
        if self.__save_model_history == False:

            callbacks_list = [other_metrics]
             
            H = model.fit(x=training_set.x, y=training_set.y, batch_size=self.__bs, validation_data=(validation_set.x, validation_set.y), epochs=self.__epochs, verbose=1,callbacks=callbacks_list)#, sample_weight=weights)
       
        return model, H

    
    def __train_1D(self, x_train, y_train, x_val, y_val, index_net, index_walk):
        # binarizing labels
        lb = LabelBinarizer()
        y_train = lb.fit_transform(y_train)
        y_val = lb.fit_transform(y_val)

        # [INFO] compiling model...
        model = SmallerVGGNet.build_vgg16_1d_smaller(height=x_train.shape[0], width=x_train.shape[1],
                                    classes=len(lb.classes_), init_var=index_net)

        opt = Adam(lr=self.__init_lr, decay=(self.__init_lr/ self.__epochs))
        model.compile(loss="sparse_categorical_crossentropy", optimizer=opt, metrics=["accuracy"])
        
        # model info
        #model.summary()
        #print(opt)

        #[INFO] training network...
        #filepath = self.__output_folder + 'models/epochs/' +"weights-improvement-{epoch:02d}-{val_acc:.2f}.hdf5"
        
        # Salvo per ogni epoca il modello
        filepath = self.__output_folder + 'models/model_foreach_epoch/' + "walk_" + str(index_walk) + "/net_" + str(index_net) + "/"
        # creo la cartella di output per gli esperimenti
        if not os.path.isdir(filepath):
            os.makedirs(filepath)

        # Se voglio salvare la storia dei modelli creo la callback 
        if self.__save_model_history == True: 
            filename =  "epoch_{epoch:02d}.model"
            # period indica ogni quanto salvare il modello 
            checkpoint = ModelCheckpoint(filepath + filename, monitor='val_loss', verbose=0, save_best_only=False, save_weights_only=False, mode='auto', period=self.__model_history_period)
            callbacks_list = [checkpoint]
            
            H = model.fit(x_train, y_train, batch_size=self.__bs, validation_data=(x_val, y_val), epochs=self.__epochs, verbose=1, callbacks=callbacks_list)
       
        # Non voglio salvare la storia dei modelli ogni tot epoche, salvo solo l'ultimo
        if self.__save_model_history == False:
            H = model.fit(x_train, y_train, batch_size=self.__bs, validation_data=(x_val, y_val), epochs=self.__epochs, verbose=1)
       
        return model, H

    '''
    '
    '''
    def __train_again(self, x_train, y_train, x_val, y_val, index_net, index_walk, model_filename):
        # Carico le immagini
        X_train = x_train.astype('float32') / 255
        X_val = x_val.astype('float32') / 255

        # binarizing labels
        lb = LabelBinarizer()
        y_train = lb.fit_transform(y_train)
        y_val = lb.fit_transform(y_val)

        # [INFO] compiling model...
        #model = SmallerVGGNet.build(height=self.__input_shape[0], width=self.__input_shape[1], depth=self.__input_shape[2],
        #                            classes=len(lb.classes_), init_var=index_net)

        model = load_model(model_filename)
        opt = Adam(lr=self.__init_lr, decay=(self.__init_lr/ self.__epochs))
        model.compile(loss="sparse_categorical_crossentropy", optimizer=opt, metrics=["accuracy"])

        # model info
        #model.summary()
        #print(opt)

        #[INFO] training network...
        #filepath = self.__output_folder + 'models/epochs/' +"weights-improvement-{epoch:02d}-{val_acc:.2f}.hdf5"
        
        # Salvo per ogni epoca il modello
        filepath = self.__output_folder + 'models/model_foreach_epoch/' + "walk_" + str(index_walk) + "/net_" + str(index_net) + "/"
        # creo la cartella di output per gli esperimenti
        if not os.path.isdir(filepath):
            os.makedirs(filepath)

        # Se voglio salvare la storia dei modelli creo la callback 
        if self.__save_model_history == True: 
            filename =  "epoch_{epoch:02d}.model"
            # period indica ogni quanto salvare il modello 
            checkpoint = ModelCheckpoint(filepath + filename, monitor='val_loss', verbose=0, save_best_only=False, save_weights_only=False, mode='auto', period=self.__model_history_period)
            callbacks_list = [checkpoint]
            
            H = model.fit(X_train, y_train, batch_size=self.__bs, validation_data=(x_val, y_val), epochs=self.__epochs, verbose=1, callbacks=callbacks_list)
       
        # Non voglio salvare la storia dei modelli ogni tot epoche, salvo solo l'ultimo
        if self.__save_model_history == False:
            H = model.fit(X_train, y_train, batch_size=self.__bs, validation_data=(x_val, y_val), epochs=self.__epochs, verbose=1)
       
        return model, H


    '''
    '
    '''
    def __train_small(self, x_train, y_train, x_val, y_val):

        # Carico le immagini
        X_train = x_train.astype('float32') / 255
        X_val = x_val.astype('float32') / 255

        # binarizing labels
        lb = LabelBinarizer()
        y_train = lb.fit_transform(y_train)
        y_val = lb.fit_transform(y_val)

        # [INFO] compiling model...
        model = SmallerVGGNet.build_small(height=40, width=40, depth=3)

        # compile model
        opt = SGD(lr=0.001, momentum=0.9)
        model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])

        H = model.fit(X_train, y_train, batch_size=self.__bs, validation_data=(x_val, y_val), epochs=self.__epochs, verbose=1)
        
        return model, H
        
    '''
    ' Ottengo il training set, 
    ' leggendo tutte le immagini con rispettiva label sia dal mercato di base
    ' che da tutte le companies associate. 
    ' Il metodo è stato pensato per usare le immagini di sp500 + le companies
    ' corrispondenti del mercato sp500
    '''
    def __get_train(self, training_set, df_label_type):
        # dataset completo, in cui metterò tutti mercati + label 
        df = pd.DataFrame()

        # per ogni datasets passato in input alla rete, leggo le immagini e label ad esso associati
        # e poi concateno tutto dentro df
        for index, input_dataset in enumerate(self.input_datasets):
            # e' importante che le path delle immagini siano passate nello stesso ordine dei datasets
            # per evitare che ci siano incompatibilità tra data_range e immagini generate
            # leggo prima la lista delle immagini e credo un df associato. 
            images_list = self.__images_filename_reader(self.__images_full_paths[index])
            # creo il dataframe associato
            images_list_df = pd.DataFrame(images_list, columns=['date_time'])
            # aggiungo una colonna contenente la path dell'immagine completa 
            images_list_df['images_path'] = self.__images_full_paths[index] + images_list_df['date_time'] + '.png'
            # converto in pd.to_datetime il campo data_time così posso mergiarlo successivamente con il df delle label
            images_list_df['date_time'] = pd.to_datetime(images_list_df['date_time'])

            # leggo ora il dataframe associato a quel datasets
            market = Market(dataset=input_dataset)
            # prendo la label del giorno successivo
            if df_label_type == 'next_day':
                label_df = market.get_label_next_day(freq='1d', columns=['delta', 'close']).reset_index()
            # prendo la label del giorno corente
            if df_label_type == 'current_day':
                label_df = market.get_label_current_day(freq='1d', columns=['delta', 'close']).reset_index()

            # effettuo il merge tra il df del mercato e quello generato con le immagini 
            # in questo modo avrò solo le date disponibili effettivamente per leggere le immagini
            # e non avrò errori in fase di lettura
            df_merge = pd.merge(images_list_df, label_df, on="date_time")
            # concateno tutto dentro il df globale su cui farò poi le operazioni di data range e lettura immagini
            df = pd.concat([df, df_merge])

        # prendo solo i giorni che sono compresi nel training set
        df = self.__get_df_by_data_range(df, training_set[0], training_set[1])
        

        # credo il vettore di input
        x_train = np.zeros(shape=(df.shape[0], self.__input_shape[0], self.__input_shape[1], self.__input_shape[2]), dtype=np.uint8)
        # creo il vettore di label prendendole direttamente dal dataframe
        y_train = self.__get_df_by_data_range(df, training_set[0], training_set[1])['label'].tolist()

        delta_train = self.__get_df_by_data_range(df, training_set[0], training_set[1])['delta'].tolist()
        close_train = self.__get_df_by_data_range(df, training_set[0], training_set[1])['close'].tolist()

        ''' Middle test
        true_label = len(np.where(df['label'] == 1)[0])
        false_label = len(np.where(df['label'] == -1)[0])
        all_samples = true_label + false_label

        print("Training samples starting by " + str(training_set[0]) + " to " + str(training_set[1]) + " with " + str(all_samples) + " samples.")
        print("Training True Label: " + str(true_label) + " [" + str((true_label*100)/all_samples) + " %]")
        print("Training False Label: " + str(false_label) + " [" + str((false_label*100)/all_samples) + " %]")
        '''

        # Resetto gli indici altrimenti salvo su x_train le stesse cose nelle prime 5k posizioni
        df = df.reset_index()

        # leggo le immagini e le metto dentro il vettore
        #for index, row in df.iterrows(): 
        for index, row in enumerate(df.itertuples(index=False)):  
            x_train[index] = cv2.imread(row.images_path)

        '''
        training_set = { 
                        'x': x_train.astype('float32') / 255, 
                        'y': [int(i) for i in y_train],
                        'delta': delta_train, 
                        'close': close_train
                }
        '''
        #return x_train, y_train, delta_train, close_train
        training_set = Set(x=x_train, y=y_train, delta=delta_train, close=close_train)
        return training_set
            

    '''
    ' Calcolo un x, y set per il mercato su cui si vogliono fare le predizioni
    ' Può essere utilizzato sia per calcolare il training (solo su sp500 ad esempio)
    ' che per calcolare validation e test set. date_set è il data range di riferimento
    ' passato come parametro alla classe. 
    '''
    def __get_set_predictions_dataset(self, date_set, df_label_type):   
        if df_label_type == 'next_day':
            market = Market(dataset=self.predictions_dataset)
            df = market.get_label_next_day(freq='1d', columns=['delta', 'close']).reset_index()

        if df_label_type == 'current_day':
            market = Market(dataset=self.predictions_dataset)
            df = market.get_label_current_day(freq='1d', columns=['delta', 'close']).reset_index()

        # Uso __get_df_by_data_range per ottenere la lista dei nomi delle immagini per quei specifici range di date 
        # uso un data_range con i df e non un semplice for per generare tutti i nomi immagini per semplicità, in alcuni
        # giorni il mercato potrebbe essere chiuso e potrebbero non esserci i nomi delle immagini
        # uso la colonna ['date_time'] per avere la data (nome della pic) e converto tutto con .tolist() per avere una lista
        date_list = self.__get_df_by_data_range(df, date_set[0], date_set[1])['date_time'].tolist()

        x = np.zeros(shape=(len(date_list), self.__input_shape[0], self.__input_shape[1], self.__input_shape[2]), dtype=np.uint8)
        y = self.__get_df_by_data_range(df, date_set[0], date_set[1])['label'].tolist() 
        
        delta = self.__get_df_by_data_range(df, date_set[0], date_set[1])['delta'].tolist() 
        close = self.__get_df_by_data_range(df, date_set[0], date_set[1])['close'].tolist() 

        #delta = []
        # ora devo riempire gli array x con le immagini vere e proprie che devo leggere
        # per fare ciò creo preventivamente delle matrici di 0 con la lunghezza della lista e la shape delle immagini
        # quindi se sono 100 immagini, per 2 input_folder (magari due companies)
        # 40x40 rbg avrò  [100 * 2][40][40][3]
        for i in range(len(date_list)):
            x[i] = cv2.imread(self.__images_base_path + self.predictions_images_folder + date_list[i].strftime('%Y-%m-%d') + '.png')
        
        '''
        set = { 
                'x': x.astype('float32') / 255, 
                'y': [int(i) for i in y],
                'date_time': date_list, 
                'delta': delta, 
                'close': close
                }
        '''
        set = Set(x=x, y=y, date_time=date_list, delta=delta, close=close)
        #return x, y, delta, close
        return set

    '''
    ' Con questo metodo ottengo il training, validation e test set da passare al metodo train()
    ' Di base alla classe viene passata una matrice con [ [data_inizio, data_fine], [data_inizio, data_fine]]
    ' dove ogni elemento del'array è una coppia di inizio fine set per walk. 
    ' In questo metodo viene passato una coppia di ciascuno (tranining, validation, test), leggo 
    ' le immagini a seconda del data range e li restituisco come lista
    ' uso la colonna ['label'] per i valori y 
    '''
    def __get_train_val_test(self, training_set, validation_set, test_set, df_label_type):
        '''
        # calcolo il training prendendo il mercato di riferimento + altre compagnie
        training_set = self.__get_train(training_set=training_set, df_label_type=df_label_type)
        
        # calcolo validation set prendendo come samples solo il mercato di riferimento
        #x_val, y_val, delta_val, close_val = self.__get_set_predictions_dataset(date_set=validation_set, df_label_type=df_label_type)
        validation_set = self.__get_set_predictions_dataset(date_set=validation_set, df_label_type=df_label_type)

        # calcolo il test set 
        #x_test, y_test, delta_test, close_test = self.__get_set_predictions_dataset(date_set=test_set, df_label_type=df_label_type)
        test_set = self.__get_set_predictions_dataset(date_set=test_set, df_label_type=df_label_type)
        
        return training_set, validation_set, test_set
        '''
        # calcolo il training prendendo il mercato di riferimento + altre compagnie
        #x_train, y_train, delta_train, close_train = self.__get_train(training_set=training_set, df_label_type=df_label_type)
        training_set = self.__get_train(training_set=training_set, df_label_type=df_label_type)

        # calcolo validation set prendendo come samples solo il mercato di riferimento
        #x_val, y_val, delta_val, close_val = self.__get_set_predictions_dataset(date_set=validation_set, df_label_type=df_label_type)
        validation_set = self.__get_set_predictions_dataset(date_set=validation_set, df_label_type=df_label_type)
        
        # calcolo il test set 
        #x_test, y_test, delta_test, close_test = self.__get_set_predictions_dataset(date_set=test_set, df_label_type=df_label_type)
        test_set = self.__get_set_predictions_dataset(date_set=test_set, df_label_type=df_label_type)

        #return x_train, y_train, x_val, y_val, delta_val, close_val, x_test, y_test, delta_test, close_test
        #return training_set, x_val, y_val, delta_val, close_val, x_test, y_test, delta_test, close_test
        return training_set, validation_set, test_set
        
    '''
    ' Configuro la rete prima di eseguirla   
    ' Effettuo anche controlli sui parametri passati al metoodo
    '''
    def net_config(self, number_of_nets=20, epochs=50, init_lr=0.001, bs=32, save_pkl=False, save_model_history=True, model_history_period=0):

        #if number_of_nets < 1 or number_of_nets > 20:
         #   sys.exit("VggHandler.net_config number_of_nets must be a value between 1 and 20")

        if not isinstance(number_of_nets, int):
            sys.exit("VggHandler.net_config number_of_nets must be a int value")

        if not isinstance(epochs, int):
            sys.exit("VggHandler.net_config epochs must be a int value")

        if not isinstance(init_lr, float):
            sys.exit("VggHandler.net_config init_lr must be a float")

        if not isinstance(bs, int):
            sys.exit("VggHandler.net_config bs must be a int value")

        if not isinstance(save_pkl, bool): 
            sys.exit("VggHandler.net_config save_pkl must be a bool value")
        
        if not isinstance(save_model_history, bool): 
            sys.exit("VggHandler.net_config save_model_history must be a bool value")

        if save_model_history is True and not isinstance(model_history_period, int): 
            sys.exit("VggHandler.net_config model_history_period must be a bool value")

        if model_history_period > epochs: 
            sys.exit("VggHandler.net_config model_history_period must be lesser dan epochs parameter")

        
        self.__number_of_nets = number_of_nets
        self.__epochs = epochs
        self.__init_lr = init_lr
        self.__bs = bs

        self.__save_pkl = save_pkl
        self.__save_model_history = save_model_history

        if save_model_history == True: 
            if model_history_period == None: 
                self.__model_history_period = epochs
            else: 
                self.__model_history_period = model_history_period
        
        if save_model_history == False:
            self.__model_history_period = None

    '''
    ' Configuro l'esperimento prima della run run()
    ' controllo la validità di alcuni parametri, ad esempio training, validation e test devono essere di dimensione uguale
    ' Converto le matrici in np.array per poterle usare in futuro
    ' Genero anche il dataset con le label e creo la cartella di output se non esiste
    '''
    def run_initialize(self, predictions_dataset, predictions_images_folder, input_images_folders, input_datasets, training_set, validation_set, test_set, input_shape, output_folder):
        
        # il dataset in cui voglio fare le predizioni
        self.predictions_dataset = predictions_dataset
        # la cartella con le immagini presenti per il dataset su cui voglio fare le predizioni
        self.predictions_images_folder = predictions_images_folder

        # setto la variabile con i datasets su cui fare il training
        self.input_datasets = input_datasets
        
        
        if len(input_images_folders) != len(input_datasets):
            sys.exit('VggHandler.__run_initialize: input_images_folders and input_datasets must have same len. input_images_folders: ' + str(len(input_images_folders)) + ' input_datasets: ' + str(len(input_datasets)))

        # La path delle immagini su cui voglio fare il training
        for input_folder in input_images_folders:
            self.__images_full_paths.append(self.__images_base_path + input_folder)

        # genero il df con la label
        datasets = Market(dataset=predictions_dataset) 
        next_day_label_df = datasets.get_label_next_day(freq='1d').reset_index()
        current_day_label_df = datasets.get_label_current_day(freq='1d').reset_index()

        # converto in np.array validation, traning e test set
        training_set = np.array(training_set)
        validation_set = np.array(validation_set)
        test_set = np.array(test_set)

        # training, validation e test devono essere di dimensione uguale
        if training_set.shape != validation_set.shape or training_set.shape != test_set.shape or validation_set.shape != test_set.shape:
            debug_string_shape = "Training set shape " + str(training_set.shape) + " - Validation set shape " + str(validation_set.shape) + " - Test set shape " + str(test_set.shape)
            sys.exit("VggHandler.__run_initialize: traning_set, validation_set and test_set must have same shape. " + debug_string_shape)

        # setto gli attributi interni della classe
        self.__training_set = training_set
        self.__validation_set = validation_set
        self.__test_set = test_set
        self.__number_of_walks = training_set.shape[0]
        self.__output_folder = self.__output_base_path + output_folder + '/'
        self.__input_shape = input_shape

        # creo la cartella di output per gli esperimenti
        if not os.path.isdir(self.__output_folder):
            os.makedirs(self.__output_folder)
        
        # Creo un DF date_time: nome_immagine (poiché vengono sempre salvate con il nome del giorno
        date_df = pd.DataFrame({'date_time': pd.to_datetime(self.__images_filename_reader(self.__images_full_paths[0]))}).sort_values(by=['date_time'])
        date_df = date_df.reset_index().drop(['index'], axis=1)

        # Unisco i DF con i giorni e la label del giorno successivo. Qui dentro quindi ci sono le label e le date presenti nelle gadf
        self.__date_next_day_label_df = pd.merge(date_df, next_day_label_df, how='inner', on='date_time')
        # Unisco i DF con i giorni e la label del giorno corrente. 
        self.__date_current_day_label_df = pd.merge(date_df, current_day_label_df, how='inner', on='date_time')

    '''
    ' Lancio l'esecuzione della rete
    ' prima di tutto convalido i dati passati come parametri, dopodichè 
    ' ciclo per ogni walk e per ogni rete e effettuo il fit del modello e calcolo l'acc in test
    '''
    def run_2D(self, start_index_walk=None):
        # salvo la prima parte dei log
        self.__start_log()
        
        # Per ogni walk calcolo train, val e test set (x,y)
        for index_walk in range(self.__number_of_walks):
            self.__add_line_log("Inizio walk n°" + str(index_walk)) 

            if start_index_walk is not None:                                                                         
                if index_walk <= start_index_walk: 
                    print("Skip walk number: ", index_walk)
                    continue
            
            # [x, y, date_time (only val e test), delta, close]
            training_set, validation_set, test_set = self.__get_train_val_test(training_set=self.__training_set[index_walk],
                                                                                        validation_set=self.__validation_set[index_walk],
                                                                                        test_set=self.__test_set[index_walk],
                                                                                        df_label_type='next_day')

            # per ogni rete effettuo il fit del modello e salvo .model, .pkl ed i plots
            for index_net in range(self.__number_of_nets):
                
                # Debug
                print("TRAINING - INDEX NET: ", str(index_net), " INDEX WALK: " + str(index_walk))

                # effettuo il training del modello
                model, H = self.__train_2D(training_set=training_set, validation_set=validation_set, test_set=test_set, index_net=index_net, index_walk=index_walk)
                #model, H = self.__train_2D(training_set=training_set, delta_val=delta_val, x_val=x_val, y_val=y_val, index_net=index_net, index_walk=index_walk)

                # Salvo il modello ed i grafici della rete
                self.__save_plots(H=H, index_walk=index_walk, index_net=index_net)
                self.__save_models(model=model, H=H, index_walk=index_walk, index_net=index_net)
                
                K.clear_session()

            self.get_avg_history(walk=index_walk, close=validation_set.close)
            #self.get_avg_history(walk=index_walk, close=close_val)

            if index_walk < (self.__number_of_walks - 1):
                is_last = 0
            else:
                is_last = 1

            self.__add_line_log("Fine walk n°" + str(index_walk))
            self.__save_last_walk_log(index_walk=index_walk, is_last=is_last)   

        # salvo l'ultima parte del log
        self.__end_log()
    
    def get_1d_x_vector(self, index_walk, vector_len, start_date, end_date):
        print("Walk " + str(index_walk) + " - Calcolo il vettore x per le date " + str(start_date) + " - " + str(end_date))

        dataset = Market(dataset=self.predictions_dataset)

        d1_df = dataset.group(freq='1d', nan=False)
        h1_df = dataset.group(freq='1h', nan=False)
        h4_df = dataset.group(freq='4h', nan=False)
        h8_df = dataset.group(freq='8h', nan=False)

        x = np.zeros(shape=(vector_len, 20), dtype=np.uint8)

        # TRAINING
        start_date = pd.to_datetime(start_date)
        end_date = pd.to_datetime(end_date)

        d1_df_selected = Market.get_df_until_data(d1_df, end_date)
        h1_df_selected = Market.get_df_until_data(h1_df, end_date)
        h4_df_selected = Market.get_df_until_data(h4_df, end_date)
        h8_df_selected = Market.get_df_until_data(h8_df, end_date)
        
        index_x = 0

        for i, (idx, row) in enumerate(d1_df_selected.iterrows()):

            if row['date_time'] < start_date:
                continue

            d1_subset = Market.get_df_until_data(d1_df_selected, row['date_time'])
            h1_subset = Market.get_df_until_data(h1_df_selected, row['date_time'])
            h4_subset = Market.get_df_until_data(h4_df_selected, row['date_time'])
            h8_subset = Market.get_df_until_data(h8_df_selected, row['date_time'])
            
            
            d1_df_range = d1_subset.tail(20)
            h1_df_range = h1_subset.tail(20)
            h4_df_range = h4_subset.tail(20)
            h8_df_range = h8_subset.tail(20)

            
            #x[index_x] = d1_df_range['delta'].to_list() + h1_df_range['delta'].to_list() + h4_df_range#['delta'].to_list() + h8_df_range['delta'].to_list() 
            x[index_x] = d1_df_range['delta'].to_list() 
            
            index_x = index_x + 1

        return x

        
    '''
    '
    '''
    def run_1D(self):
        # salvo la prima parte dei log
        self.__start_log()
        
        # Per ogni walk calcolo train, val e test set (x,y)
        for index_walk in range(self.__number_of_walks):

            dataset = Market(dataset=self.predictions_dataset)

            day_df = dataset.get_label_next_day(freq='1d')

            y_train =  Market.get_df_by_data_range(day_df, self.__training_set[index_walk][0], self.__training_set[index_walk][1])['label'].to_list()
            y_val =  Market.get_df_by_data_range(day_df, self.__validation_set[index_walk][0], self.__validation_set[index_walk][1])['label'].to_list()
            y_test =  Market.get_df_by_data_range(day_df, self.__test_set[index_walk][0], self.__test_set[index_walk][1])['label'].to_list()

           
            x_train = self.get_1d_x_vector(index_walk=index_walk, vector_len=len(y_train), start_date=self.__training_set[index_walk][0], end_date=self.__training_set[index_walk][1])
            x_val = self.get_1d_x_vector(index_walk=index_walk, vector_len=len(y_val), start_date=self.__validation_set[index_walk][0], end_date=self.__validation_set[index_walk][1])
            x_test = self.get_1d_x_vector(index_walk=index_walk, vector_len=len(y_test), start_date=self.__test_set[index_walk][0], end_date=self.__test_set[index_walk][1])
           

            x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], 1)
            x_val = x_val.reshape(x_val.shape[0], x_val.shape[1], 1)
            x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], 1)

            # per ogni rete effettuo il fit del modello e salvo .model, .pkl ed i plots
            for index_net in range(self.__number_of_nets):
                
                # Debug
                print("TRAINING - INDEX NET: ", str(index_net), " INDEX WALK: " + str(index_walk))

                # effettuo il training del modello
                model, H = self.__train_1D(x_train=x_train, y_train=y_train, x_val=x_val, y_val=y_val, index_net=index_net, index_walk=index_walk)

                # Salvo il modello ed i grafici della rete
                self.__save_plots(H=H, index_walk=index_walk, index_net=index_net)
                self.__save_models(model=model, H=H, index_walk=index_walk, index_net=index_net)
                
                K.clear_session()
                
                
        # salvo l'ultima parte del log
        self.__end_log()
        

    '''
    '
    '''
    def run_small(self):
        # salvo la prima parte dei log
        self.__start_log()
        
        # Per ogni walk calcolo train, val e test set (x,y)
        for index_walk in range(self.__number_of_walks):

            x_train, y_train, delta_train, x_val, y_val, delta_val, x_test, y_test, delta_test = self.__get_train_val_test(training_set=self.__training_set[index_walk],
                                                                                        validation_set=self.__validation_set[index_walk],
                                                                                        test_set=self.__test_set[index_walk],
                                                                                        df_label_type='next_day')
            
            # Debug
            print("TRAINING - INDEX WALK: " + str(index_walk) + " with small CNN")

            # effettuo il training del modello
            model, H = self.__train_small(x_train=x_train, y_train=y_train, x_val=x_val, y_val=y_val)

            # Salvo il modello ed i grafici della rete
            self.__save_plots(H=H, index_walk=index_walk, index_net=1)
            self.__save_models(model=model, H=H, index_walk=index_walk, index_net=1)
            
            K.clear_session()
                
                
        # salvo l'ultima parte del log
        self.__end_log()

    '''
    '
    '''
    def run_again(self, model_input_folder):
        # salvo la prima parte dei log
        self.__start_log()
        
        # Per ogni walk calcolo train, val e test set (x,y)
        for index_walk in range(self.__number_of_walks):

            x_train, y_train, delta_train, x_val, y_val, delta_val, x_test, y_test, delta_test = self.__get_train_val_test(training_set=self.__training_set[index_walk],
                                                                                        validation_set=self.__validation_set[index_walk],
                                                                                        test_set=self.__test_set[index_walk],
                                                                                        df_label_type='next_day')
            # per ogni rete effettuo il fit del modello e salvo .model, .pkl ed i plots
            for index_net in range(self.__number_of_nets):
                # Debug
                print("TRAINING - INDEX NET: ", str(index_net), " INDEX WALK: " + str(index_walk))

                # carico il modello per utilizzarlo successivamente per calcolare l'a classe di 'output
                model_filename = self.__output_base_path + model_input_folder + '/' + 'models/walk_' + str(index_walk) + '_net_'+ str(index_net) + '.model'

                # effettuo il training del modello
                model, H = self.__train_again(x_train=x_train, y_train=y_train, x_val=x_val, y_val=y_val, index_net=index_net, index_walk=index_walk, model_filename=model_filename)

                # Salvo il modello ed i grafici della rete
                self.__save_plots(H=H, index_walk=index_walk, index_net=index_net)
                self.__save_models(model=model, H=H, index_walk=index_walk, index_net=index_net)
                
                K.clear_session()
                
                
        # salvo l'ultima parte del log
        self.__end_log()

    '''
    ' Questo metodo stampa all'interno della cartella predictions l'output della rete allenata
    ' E' possibile stampare l'output sia per il training, validation e test. set_type è appunto il parametro
    ' di ingresso per stabilire quale degli output stampati.
    ' Il csv stampato viene generato usando la label del giorno corrente. 
    '''
    def get_predictions_2D(self, set_type):

        # Controllo che il parametro sia valido
        if set_type  is not 'validation' and set_type is not 'test':
            sys.exit('VggHandler.get_predictions: set_type must be validation or test')

        # Dove andrò a salvare il file, dentro la cartella predictions
        folder = self.__output_folder + 'predictions/' + set_type + '/'
            
        # Se le cartella del dataset non esistono, la creo a runtime
        if not os.path.isdir(folder):
            os.makedirs(folder)

        # Per ogni walk calcolo train, val e test set (x,y)
        for index_walk in range(0, self.__number_of_walks):

            # Calcolo tutti i vari set per questo walk
            x_train, y_train, delta_train, x_val, y_val, delta_val, x_test, y_test, delta_test = self.__get_train_val_test(
                                                                                    training_set=self.__training_set[index_walk],
                                                                                    validation_set=self.__validation_set[index_walk],
                                                                                    test_set=self.__test_set[index_walk],
                                                                                    df_label_type='next_day')

            # A seconda del parametro passato al metodo setto dentro walk_df il set desiderato
            # x_set e y_set li uso come variabile generale, settati con una delle variabili generate sopra.
            # lo faccio per rendere il metodo più generale, così successivamente uso solo x/y_set
            if set_type == "validation":
                walk_df = self.__get_df_by_data_range(df=self.__date_next_day_label_df, 
                                                        start_date=self.__validation_set[index_walk][0], 
                                                        end_date=self.__validation_set[index_walk][1])
                x_set = x_val
                y_set = y_val

            if set_type == "test":
                walk_df = self.__get_df_by_data_range(df=self.__date_next_day_label_df, 
                                                        start_date=self.__test_set[index_walk][0], 
                                                        end_date=self.__test_set[index_walk][1])
                x_set = x_test
                y_set = y_test

            # imposto come sempre date_time come indice del dataframe
            walk_df = walk_df.set_index('date_time')
            
            # imposto il nome del file di output aggiungendogli la path 
            walkname = folder + 'GADF_walk_' + str(index_walk) + ".csv"
            
            # per ogni rete carico modello preallenato e genero le predizioni per ogni giorno
            for index_net in range(0, self.__number_of_nets):
                print("WALK:" + str(index_walk) + ' - NET: ' + str(index_net))

                # Calcolo la grandezza del x_set in modo da crearmi poi un array vuoto di quella dimensione (per metterci dentro le predizioni)
                set_size = x_set.shape[0]
                preds = np.zeros(shape=set_size, dtype=np.uint8)
                
                # carico il modello per utilizzarlo successivamente per calcolare l'a classe di 'output
                model_filename = self.__output_folder + 'models/walk_' + str(index_walk) + '_net_'+ str(index_net) + '.model'
                #model = load_model(model_filename)
                model = load_model(model_filename, custom_objects={'RAdam': RAdam})


                ''' Utilizzo le probabilità, al momento non utilizzato
                probs = np.zeros(shape=set_size, dtype=np.uint8)
                probs = model.predict(x_set)
                print(probs)
                preds = np.argmax(probs, axis=1)
                preds[preds == 0] = -1
                '''

                # Calcolo la classe e genero l'array delle predizioni
                preds = model.predict_classes(x_set)
                
                # Cambio la classe 0 con -1, in quanto le short in tutto il framework sono indicate con -1
                preds[preds == 0] = -1
                
                # aggiungo la colonna net_N con le predizioni appena calcolate 
                walk_df['net_' + str(index_net)] = preds

                # Pulisco la sessione per evitare rallentamenti ad ogni load
                K.clear_session()
            # Salvo le predizioni con tutte le reti incolonnate
            walk_df.to_csv(walkname, header=True, index=True)

    '''
    '
    '''
    def get_predictions_1D(self, set_type):

        # Controllo che il parametro sia valido
        if set_type  is not 'validation' and set_type is not 'test':
            sys.exit('VggHandler.get_predictions: set_type must be validation or test')

        # Dove andrò a salvare il file, dentro la cartella predictions
        folder = self.__output_folder + 'predictions/' + set_type + '/'
            
        # Se le cartella del dataset non esistono, la creo a runtime
        if not os.path.isdir(folder):
            os.makedirs(folder)

        # Per ogni walk calcolo train, val e test set (x,y)
        for index_walk in range(0, self.__number_of_walks):
            
            dataset = Market(dataset=self.predictions_dataset)

            day_df = dataset.get_label_next_day(freq='1d')

            y_train =  Market.get_df_by_data_range(day_df, self.__training_set[index_walk][0], self.__training_set[index_walk][1])['label'].to_list()
            y_val =  Market.get_df_by_data_range(day_df, self.__validation_set[index_walk][0], self.__validation_set[index_walk][1])['label'].to_list()
            y_test =  Market.get_df_by_data_range(day_df, self.__test_set[index_walk][0], self.__test_set[index_walk][1])['label'].to_list()

           
            x_train = self.get_1d_x_vector(index_walk=index_walk, vector_len=len(y_train), start_date=self.__training_set[index_walk][0], end_date=self.__training_set[index_walk][1])
            x_val = self.get_1d_x_vector(index_walk=index_walk, vector_len=len(y_val), start_date=self.__validation_set[index_walk][0], end_date=self.__validation_set[index_walk][1])
            x_test = self.get_1d_x_vector(index_walk=index_walk, vector_len=len(y_test), start_date=self.__test_set[index_walk][0], end_date=self.__test_set[index_walk][1])

            x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], 1)
            x_val = x_val.reshape(x_val.shape[0], x_val.shape[1], 1)
            x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], 1)

            # A seconda del parametro passato al metodo setto dentro walk_df il set desiderato
            # x_set e y_set li uso come variabile generale, settati con una delle variabili generate sopra.
            # lo faccio per rendere il metodo più generale, così successivamente uso solo x/y_set
            if set_type == "validation":
                walk_df = self.__get_df_by_data_range(df=self.__date_next_day_label_df, 
                                                        start_date=self.__validation_set[index_walk][0], 
                                                        end_date=self.__validation_set[index_walk][1])
                x_set = x_val
                y_set = y_val

            if set_type == "test":
                walk_df = self.__get_df_by_data_range(df=self.__date_next_day_label_df, 
                                                        start_date=self.__test_set[index_walk][0], 
                                                        end_date=self.__test_set[index_walk][1])
                x_set = x_test
                y_set = y_test

            # imposto come sempre date_time come indice del dataframe
            walk_df = walk_df.set_index('date_time')

            # imposto il nome del file di output aggiungendogli la path 
            walkname = folder + 'GADF_walk_' + str(index_walk) + ".csv"
            
            # per ogni rete carico modello preallenato e genero le predizioni per ogni giorno
            for index_net in range(0, self.__number_of_nets):
                print("WALK:" + str(index_walk) + ' - NET: ' + str(index_net))
                
                # Calcolo la grandezza del x_set in modo da crearmi poi un array vuoto di quella dimensione (per metterci dentro le predizioni)
                set_size = x_set.shape[0]
                preds = np.zeros(shape=set_size, dtype=np.uint8)
                
                # carico il modello per utilizzarlo successivamente per calcolare l'a classe di 'output
                model_filename = self.__output_folder + 'models/walk_' + str(index_walk) + '_net_'+ str(index_net) + '.model'
                model = load_model(model_filename)

                # Calcolo la classe e genero l'array delle predizioni
                preds = model.predict_classes(x_set)
                
                # Cambio la classe 0 con -1, in quanto le short in tutto il framework sono indicate con -1
                preds[preds == 0] = -1
                
                # aggiungo la colonna net_N con le predizioni appena calcolate 
                walk_df['net_' + str(index_net)] = preds

                # Pulisco la sessione per evitare rallentamenti ad ogni load
                K.clear_session()
            # Salvo le predizioni con tutte le reti incolonnate
            walk_df.to_csv(walkname, header=True, index=True)
    '''
    '
    '''
    def get_predictions_foreach_epoch(self, set_type):

        # Controllo che il parametro sia valido
        if set_type  is not 'validation' and set_type is not 'test':
            sys.exit('VggHandler.get_predictions: set_type must be validation or test')

        # Dove andrò a salvare il file, dentro la cartella predictions
        folder = self.__output_folder + 'predictions_foreach_model/' + set_type + '/'
            
        # Se le cartella del dataset non esistono, la creo a runtime
        if not os.path.isdir(folder):
            os.makedirs(folder)

        # Per ogni walk calcolo train, val e test set (x,y)
        for index_walk in range(0, self.__number_of_walks):

            # Calcolo tutti i vari set per questo walk
            x_train, y_train, delta_train, x_val, y_val, delta_val, x_test, y_test, delta_test = self.__get_train_val_test(
                                                                                    training_set=self.__training_set[index_walk],
                                                                                    validation_set=self.__validation_set[index_walk],
                                                                                    test_set=self.__test_set[index_walk],
                                                                                    df_label_type='next_day')

            # A seconda del parametro passato al metodo setto dentro walk_df il set desiderato
            # x_set e y_set li uso come variabile generale, settati con una delle variabili generate sopra.
            # lo faccio per rendere il metodo più generale, così successivamente uso solo x/y_set
            if set_type == "validation":
                walk_df = self.__get_df_by_data_range(df=self.__date_current_day_label_df, 
                                                        start_date=self.__validation_set[index_walk][0], 
                                                        end_date=self.__validation_set[index_walk][1])
                x_set = x_val
                y_set = y_val

            if set_type == "test":
                walk_df = self.__get_df_by_data_range(df=self.__date_current_day_label_df, 
                                                        start_date=self.__test_set[index_walk][0], 
                                                        end_date=self.__test_set[index_walk][1])
                x_set = x_test
                y_set = y_test

            # imposto come sempre date_time come indice del dataframe
            walk_df = walk_df.set_index('date_time')
            
            # imposto il nome del file di output aggiungendogli la path 
            walkname = folder + 'GADF_walk_' + str(index_walk) + ".csv"
            
            index_net = 0
                
            starting = self.__model_history_period
            acc = self.__model_history_period

            while starting <= self.__epochs:
                    # imposto il nome del file di output aggiungendogli la path 
                walkname = folder + 'GADF_epochs_' + str(starting) + ".csv"
                print("WALK:" + str(index_walk) + ' - NET: ' + str(index_net) + ' EPOCH: ' +  str(starting))

                # Calcolo la grandezza del x_set in modo da crearmi poi un array vuoto di quella dimensione (per metterci dentro le predizioni)
                set_size = x_set.shape[0]
                preds = np.zeros(shape=set_size, dtype=np.uint8)
                
                # carico il modello per utilizzarlo successivamente per calcolare l'a classe di 'output
                model_filename = self.__output_folder + 'models/model_foreach_epoch/walk_' + str(index_walk) + '/net_'+ str(index_net) + '/epoch_' + str(starting) + '.model'
                model = load_model(model_filename)


                # Calcolo la classe e genero l'array delle predizioni
                preds = model.predict_classes(x_set)
                
                # Cambio la classe 0 con -1, in quanto le short in tutto il framework sono indicate con -1
                preds[preds == 0] = -1
                
                # aggiungo la colonna net_N con le predizioni appena calcolate 
                walk_df['net_' + str(index_net)] = preds

                # Pulisco la sessione per evitare rallentamenti ad ogni load
                K.clear_session()

                starting += acc
                # Salvo le predizioni con tutte le reti incolonnate
                walk_df.to_csv(walkname, header=True, index=True)

    
