import os
import sys
import cv2
import time
import pickle
import datetime
import matplotlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# VISIOSCIENTIAE FRAMEWORK
from classes.Market import Market
from classes.Gaf import Gaf
from classes.VggHandlerTemperature import VggHandlerTemperature
from classes.ResultsHandlerTemperature import ResultsHandlerTemperature


# KERAS FRAMEWORK
from keras.models import load_model
from vgg16.SmallerVGGNet import SmallerVGGNet
from sklearn.preprocessing import LabelBinarizer
from keras.optimizers import Adam, RMSprop, SGD, Nadam
from keras import backend as K
from keras.callbacks import ModelCheckpoint


# TEST CLASSIFICATORI
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn.metrics import accuracy_score

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID";
os.environ["CUDA_VISIBLE_DEVICES"]="1";

'''

def group(df, freq):

    # Get first Open, Last Close, Highest value of "High" and Lower value of "Low", and sum(Volume)
    grouped = df.groupby(pd.Grouper(key='date_time', freq=freq), sort=True).agg({
        'temperature': 'mean'
    })
    
    return grouped.dropna()

# calcolo la label per ogni riga, controllando la successiva
def get_label_next_day_using_close(df, columns=[]):
    
    df['label'] = 0

    # Calcolo per ogni riga la Label, ovvero se il giorno dopo il mercato
    # salira' o scendera'
    # quindi se oggi la temperatura è 10 e domani 12, la label sarà 1 altrimenti 0
    df['label'] = (df['temperature'].diff().shift(-1) > 0).astype(int)
    
    # Invece di mettere la label sul giorno precedente, la mette sul giorno corrente. 
    #df['label'] = (df['close'].diff().bfill() > 0).astype(int)
    
    if len(columns) > 0:
        columns.append('label')
        df = df.filter(columns, axis=1)
    else:
        df = df.filter(['label'], axis=1) 

    return df


def save_plots(output_folder, H, epochs, walk_index, approach):

    full_plots_path = output_folder + 'accuracy_loss/'

    # Se non esiste la cartella, la creo
    if not os.path.isdir(full_plots_path):
        os.makedirs(full_plots_path)

    plt.figure(figsize=(15,12))
    # plot the training loss and accuracy
    plt.style.use("ggplot")
    plt.subplot(2, 1, 1)
    plt.plot(np.arange(0, epochs), H.history["acc"], label="train_acc")
    plt.plot(np.arange(0, epochs), H.history["val_acc"], label="val_acc")
    #plt.title("Training Accuracy")
    plt.xlabel("Epoch #")
    plt.ylabel("Accuracy")
    plt.legend(loc="upper left")

    plt.subplot(2, 1, 2)
    plt.plot(np.arange(0, epochs), H.history["loss"], label="train_loss")
    plt.plot(np.arange(0, epochs), H.history["val_loss"], label="val_loss")
    #plt.title("Training Loss")
    plt.xlabel("Epoch #")
    plt.ylabel("Loss")
    plt.legend(loc="upper left")

    plt.savefig(full_plots_path + 'results_walk_' + str(walk_index) + '_' + approach + '.png')

    plt.close('all')

# fit del modello 
def train_1D(x_train, y_train, x_val, y_val, epochs=10):
    __epochs = epochs
    __init_lr = 0.001
    __bs = 6

    # binarizing labels
    lb = LabelBinarizer()
    y_train = lb.fit_transform(y_train)
    y_val = lb.fit_transform(y_val)

    # [INFO] compiling model...
    #model = SmallerVGGNet.build_small_1d(height=x_train.shape[0], width=x_train.shape[1], classes=len(lb.classes_))
    #opt = SGD(lr=0.001, momentum=0.9)
    #model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])

    model = SmallerVGGNet.build_vgg16_1d_smaller(height=x_train.shape[0], width=x_train.shape[1], classes=len(lb.classes_), init_var=0)
    opt = Adam(lr=__init_lr, decay=(__init_lr/ __epochs))
    model.compile(loss="sparse_categorical_crossentropy", optimizer=opt, metrics=["accuracy"])

    

    H = model.fit(x_train, y_train, batch_size=__bs, validation_data=(x_val, y_val), epochs=__epochs, verbose=1)
       
    return model, H

#
def get_1d_x_vector(df, vector_len, feature_size, start_date, end_date):

    #print("Calculating 1D x vector for data range: " + str(start_date) + " - " + str(end_date))
    x = np.zeros(shape=(vector_len, feature_size), dtype=np.uint8)

    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)

    df_selected = Market.get_df_until_data(df, end_date)
    
    index_x = 0

    for i, (idx, row) in enumerate(df_selected.iterrows()):

        if row['date_time'] < start_date:
            continue

        d1_subset = Market.get_df_until_data(df_selected, row['date_time'])
        
        df_range = d1_subset.tail(feature_size)
        x[index_x] = df_range['temperature'].to_list() 
        
        index_x = index_x + 1

    return x

#
def get_1d_x_vector_multi_resolution(df_1d, df_1h, df_4h, df_8h, vector_len, feature_size, start_date, end_date):

    #print("Calculating 1D x vector for data range: " + str(start_date) + " - " + str(end_date))
    x = np.zeros(shape=(vector_len, feature_size), dtype=np.uint8)

    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)

    d1_df_selected = Market.get_df_until_data(df_1d, end_date)
    h1_df_selected = Market.get_df_until_data(df_1h, end_date)
    h4_df_selected = Market.get_df_until_data(df_4h, end_date)
    h8_df_selected = Market.get_df_until_data(df_8h, end_date)

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

        x[index_x] = d1_df_range['temperature'].to_list() + h1_df_range['temperature'].to_list() + h4_df_range['temperature'].to_list() + h8_df_range['temperature'].to_list() 
        
        index_x = index_x + 1

    return x
    
#
def run_CNN_1D(df, start_date_train, end_date_train, start_date_validation, end_date_validation, epochs, output_folder, walk_index, approach):
    
    y_train =  Market.get_df_by_data_range(df, start_date_train, end_date_train)['label'].to_list()
    y_val =  Market.get_df_by_data_range(df, start_date_validation, end_date_validation)['label'].to_list()

    
    x_train = get_1d_x_vector(df=df, vector_len=len(y_train), feature_size=20, start_date=start_date_train, end_date=end_date_train)
    x_val = get_1d_x_vector(df=df, vector_len=len(y_val), feature_size=20, start_date=start_date_validation, end_date=end_date_validation)
    
    x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], 1)
    x_val = x_val.reshape(x_val.shape[0], x_val.shape[1], 1)

    model, H = train_1D(x_train=x_train, y_train=y_train, x_val=x_val, y_val=y_val, epochs=epochs)

    save_plots(output_folder=output_folder, H=H, epochs=epochs, walk_index=walk_index, approach=approach)
    
#
def run_CNN_1D_multi_resolution(df_1d, df_1h, df_4h, df_8h, start_date_train, end_date_train, start_date_validation, end_date_validation, epochs, output_folder, walk_index, approach):
    
    y_train =  Market.get_df_by_data_range(df_1d, start_date_train, end_date_train)['label'].to_list()
    y_val =  Market.get_df_by_data_range(df_1d, start_date_validation, end_date_validation)['label'].to_list()

    
    x_train = get_1d_x_vector_multi_resolution(df_1d=df_1d, df_1h=df_1h, df_4h=df_4h, df_8h=df_8h, vector_len=len(y_train), feature_size=80, start_date=start_date_train, end_date=end_date_train)
    x_val = get_1d_x_vector_multi_resolution(df_1d=df_1d, df_1h=df_1h, df_4h=df_4h, df_8h=df_8h, vector_len=len(y_val), feature_size=80, start_date=start_date_validation, end_date=end_date_validation)
    
    x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], 1)
    x_val = x_val.reshape(x_val.shape[0], x_val.shape[1], 1)

    model, H = train_1D(x_train=x_train, y_train=y_train, x_val=x_val, y_val=y_val, epochs=epochs)

    save_plots(output_folder=output_folder, H=H, epochs=epochs, walk_index=walk_index, approach=approach)

#
def run_classifier(df, start_date_train, end_date_train, start_date_validation, end_date_validation, epochs, output_folder):
    
    y_train =  Market.get_df_by_data_range(df, start_date_train, end_date_train)['label'].to_list()
    y_val =  Market.get_df_by_data_range(df, start_date_validation, end_date_validation)['label'].to_list()
    y_val=  np.asarray(y_val)
    print("Label originali: ")
    print("[run_classifier] Vettore delle label, numero di 0 presenti: " + str(np.count_nonzero(y_val == 0)))
    print("[run_classifier] Vettore delle label, numero di 1 presenti: : " + str(np.count_nonzero(y_val == 1)))

    x_train = get_1d_x_vector(df=df, vector_len=len(y_train), feature_size=20, start_date=start_date_train, end_date=end_date_train)
    x_val = get_1d_x_vector(df=df, vector_len=len(y_val), feature_size=20, start_date=start_date_validation, end_date=end_date_validation)
    
    
    #clf = RandomForestClassifier(n_estimators=1000, max_depth=300, random_state=0)
    #clf.fit(x_train, y_train)  
    #y_pred = clf.predict(x_val)
    #accuracy = accuracy_score(y_val, y_pred)
    #print("Single Resolution Random Forest Accuracy: ", accuracy)
    
    clf = svm.SVC(gamma='scale')
    clf.fit(x_train, y_train)  
    y_pred = clf.predict(x_val)
    accuracy_non_norm = accuracy_score(y_val, y_pred, normalize=False)
    accuracy_norm = accuracy_score(y_val, y_pred, normalize=True)
    print("Label predette: ")
    print("[run_classifier] Accuracy NON normalizzata: ", accuracy_non_norm)
    print("[run_classifier] Accuracy normalizzata: ", accuracy_norm)
    print("[run_classifier] Vettore delle label, numero di 0 presenti: " + str(np.count_nonzero(y_pred == 0)))
    print("[run_classifier] Vettore delle label, numero di 1 presenti: : " + str(np.count_nonzero(y_pred == 1)))
    
#
def run_classifier_multi_resolution(df_1d, df_1h, df_4h, df_8h, start_date_train, end_date_train, start_date_validation, end_date_validation, epochs, output_folder):
    
    y_train =  Market.get_df_by_data_range(df_1d, start_date_train, end_date_train)['label'].to_list()
    y_val =  Market.get_df_by_data_range(df_1d, start_date_validation, end_date_validation)['label'].to_list()
    y_val=  np.asarray(y_val)
    print("Label originali: ")
    print("[run_classifier_multi_resolution] Vettore delle label, numero di 0 presenti: " + str(np.count_nonzero(y_val == 0)))
    print("[run_classifier_multi_resolution] Vettore delle label, numero di 1 presenti: : " + str(np.count_nonzero(y_val == 1)))

    x_train = get_1d_x_vector_multi_resolution(df_1d=df_1d, df_1h=df_1h, df_4h=df_4h, df_8h=df_8h, vector_len=len(y_train), feature_size=80, start_date=start_date_train, end_date=end_date_train)
    x_val = get_1d_x_vector_multi_resolution(df_1d=df_1d, df_1h=df_1h, df_4h=df_4h, df_8h=df_8h, vector_len=len(y_val), feature_size=80, start_date=start_date_validation, end_date=end_date_validation)
    
    clf = RandomForestClassifier(n_estimators=1000, max_depth=300, random_state=0)
    clf.fit(x_train, y_train)  
    y_pred = clf.predict(x_val)
    accuracy_non_norm = accuracy_score(y_val, y_pred, normalize=False)
    accuracy_norm = accuracy_score(y_val, y_pred, normalize=True)

    print("Label predette: ")
    print("[run_classifier_multi_resolution] Accuracy NON normalizzata: ", accuracy_non_norm)
    print("[run_classifier_multi_resolution] Accuracy normalizzata: ", accuracy_norm)
    print("[run_classifier_multi_resolution] Vettore delle label, numero di 0 presenti: " + str(np.count_nonzero(y_pred == 0)))
    print("[run_classifier_multi_resolution] Vettore delle label, numero di 1 presenti: : " + str(np.count_nonzero(y_pred == 1)))


    #print(y_pred)
    #print(np.count_nonzero(y_pred == 1))
    
    #clf = svm.SVC(gamma='scale')
    #clf.fit(x_train, y_train)  
    #y_pred = clf.predict(x_val)
    #accuracy = accuracy_score(y_val, y_pred    )
    #print("Multi Resolution SVM Accuracy: ", accuracy)




##### Parametri dello script
dataset_path = '../datasets/temperature/temperature.csv'
choosen_city = 'Los Angeles'
#####


# leggo il dataset
df = pd.read_csv(dataset_path)

# Rimuovo le righe contenenti NaN
df = df[['datetime', choosen_city]].dropna()

# converto la temperatura in Celsius
df[choosen_city] =  df[choosen_city]  - 273.15

# converto in datetime la colonna, per essere usata in futuro
df['datetime'] = pd.to_datetime(df['datetime'])
# converto il nome della città in temperature, così lo script può essere generico
df = df.rename(columns={'datetime': 'date_time', choosen_city: 'temperature'})

df_4h = group(df=df, freq='4h')
df_8h = group(df=df, freq='8h')
df_1d = group(df=df, freq='1d')

df_1h = df.set_index('date_time')

# Aggiungo la label a fianco alla temperatura
df_1d = get_label_next_day_using_close(df=df_1d, columns=['datetime', 'temperature'])
df_1h = get_label_next_day_using_close(df=df_1h, columns=['datetime', 'temperature'])
df_4h = get_label_next_day_using_close(df=df_4h, columns=['datetime', 'temperature'])
df_8h = get_label_next_day_using_close(df=df_8h, columns=['datetime', 'temperature'])

for i in range(len(training_set)):
    
    print("CNN 1D Single Resolution Walk " + str(i))
    run_CNN_1D(df=df_1d,
            start_date_train=training_set[i][0], 
            end_date_train=training_set[i][1], 
            start_date_validation=validation_set[i][0], 
            end_date_validation=validation_set[i][1], 
            epochs=50, 
            output_folder='../results/temperature',
            walk_index=i,
            approach='single_resolution'
    )


    print("CNN 1D Multi Resolution Walk " + str(i))
    run_CNN_1D_multi_resolution(
            df_1d=df_1d,
            df_1h=df_1h,
            df_4h=df_4h, 
            df_8h=df_8h,
            start_date_train=training_set[i][0], 
            end_date_train=training_set[i][1], 
            start_date_validation=validation_set[i][0], 
            end_date_validation=validation_set[i][1],  
            epochs=50, 
            output_folder='../results/temperature',
            walk_index=i,
            approach='multi_resolution'
    )

i = 0

#print("Classifier Single Resolution Walk " + str(i))
run_classifier(df=df_1d,
        start_date_train=training_set[i][0], 
        end_date_train=training_set[i][1], 
        start_date_validation=validation_set[i][0], 
        end_date_validation=validation_set[i][1], 
        epochs=50, 
        output_folder='../results/temperature'
)

print("\n\n------------------------------------------\n\n")

#print("Classifier Multi Resolution Walk " + str(i))
run_classifier_multi_resolution(
        df_1d=df_1d,
        df_1h=df_1h,
        df_4h=df_4h, 
        df_8h=df_8h,
        start_date_train=training_set[i][0], 
        end_date_train=training_set[i][1], 
        start_date_validation=validation_set[i][0], 
        end_date_validation=validation_set[i][1],
        epochs=50, 
        output_folder='../results/temperature'
)
'''


# Replica esperimento con Seba
training_set =      [['2012-10-20', '2015-10-20'],  ['2013-03-20', '2016-03-20'],  ['2013-08-20', '2016-08-20'],  ['2014-01-20', '2017-01-20']]
validation_set =    [['2015-10-21', '2016-03-20'],  ['2016-03-21', '2016-08-20'],  ['2016-08-21', '2017-01-20'],  ['2017-01-21', '2017-06-20']]
test_set =          [['2016-03-21', '2016-08-20'],  ['2016-08-21', '2017-01-20'],  ['2017-01-21', '2017-06-20'],  ['2017-06-21', '2017-11-20']]

#experiment_name = 'exp_temp_CNN1_single_resolution'
experiment_name = 'exp_temp_CNN1_multi_resolution'

vgg = VggHandlerTemperature()

vgg.net_config(number_of_nets=20, epochs=50, save_pkl=True, save_model_history=True, model_history_period=50)

vgg.run_initialize(
        training_set=training_set,
        validation_set=validation_set,
        test_set=test_set,
        output_folder=experiment_name)

#vgg.run_CNN_1D()
#vgg.run_CNN_1D_multi_resolution()

results_handler = ResultsHandlerTemperature(experiment_name=experiment_name)

#vgg.get_predictions_1D_multi_resolution(set_type='validation')
results_handler.generate_ensemble(set_type='validation')
results_handler.generate_plots(set_type='validation')

#vgg.get_predictions_1D_multi_resolution(set_type='test')
results_handler.generate_ensemble(set_type='test')
results_handler.generate_plots(set_type='test')

