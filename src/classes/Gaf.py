import os
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import sys
import matplotlib
import numpy as np
import pandas as pd
from PIL import Image
from datetime import timedelta
import matplotlib.pyplot as plt
from pyts.image import GASF, GADF
pd.options.mode.chained_assignment = None

'''
' @author Andrea Corriga
' Partendo da un dataset passato come parametro, salvo le trasformate 
' Gramian Angular Field ad esso associato. 
' @link https://pyts.readthedocs.io/en/latest/auto_examples/image/plot_gaf.html#sphx-glr-auto-examples-image-plot-gaf-py
'''

class Gaf:

    __df = pd.DataFrame()
    __size = 40
    __dataset_name = ""
    __subfolder = ""
    __output_path = "../images/"

    __gadf_path = ""
    __gasf_path = ""

    # Costruttore, setto i parametri di base per le trasformate
    def __validate(self, df=pd.DataFrame(), dataset_name=None, subfolder=None, type='gadf', size=40):

        if df.empty:
            sys.exit("Gaf.__init__: df cannot be null")
        else:
            self.__df = df

        if dataset_name is None:
            sys.exit("Gaf.__init__: dataset_name cannot be null")
        else:
            self.__dataset_name = dataset_name
        if subfolder is None:
            sys.exit("Gaf.__init__: subfolder cannot be null")
        else:
            self.__subfolder = subfolder

        if type != 'gadf' and type != 'gasf':
            sys.exit('Gaf.run: type must be gadf or gasf')

        # Size può non essere specificato, perché settato di default a 40 dalla classe
        if size is not None:
            if isinstance(size, int):
                self.__size = size
            else:
                sys.exit("Gaf.run: size must be int")

        self.__gadf_path = self.__output_path + self.__dataset_name + '/' + self.__subfolder + '/' + 'gadf/'
        self.__gasf_path = self.__output_path + self.__dataset_name + '/' + self.__subfolder + '/' + 'gasf/'

        # Se le cartella del dataset non esistono, la creo a runtime
        if not os.path.isdir(self.__gadf_path):
            os.makedirs(self.__gadf_path)

        if not os.path.isdir(self.__gasf_path):
            os.makedirs(self.__gasf_path)

    '''
    ' Questa funzione calcola le coordinate la GADF e la GASF per un data_frame.
    ' Si da per scontato che il Data Frame passato alla funzione sia gia' un sotto-insieme
    ' del Dataset originale (dovrebbe essere calcolato su un segnale ridotto, ad esempio 40 ore precedenti)
    ' La dimensione dell'immagine viene specificata dall'atrributo self.__size della classe
    ' Come prima cosa i vari segnali vengono convertiti in un range [-1, 1], dopodiche'
    ' si convertono i valori in coordinate polari. Infine si calcolano le trasformate.
    '''
    def __calculate_gaf(self, df, type):
        # Check di sicurezza, rimuovo le righe NaN
        df = df.dropna()
        # Scalo tutti i valori del Dataset, inserendoli in un range [-1, 1]
        min_ = np.amin(df)
        max_ = np.amax(df)
        scaled_serie = (2 * df - max_ - min_) / (max_ - min_)
        # Floating point inaccuracy!
        scaled_serie = np.where(scaled_serie >= 1., 1., scaled_serie)
        scaled_serie = np.where(scaled_serie <= -1., -1., scaled_serie)

        # Calcolo le coordinate polari
        phi = np.arccos(scaled_serie)
        # La computazione di r non e' necessaria ai fini della GADF GASF,
        # serve se si vuole plottare il grafico con le C.P.
        # r = np.linspace(0, 1, len(scaled_serie))

        # Calcolo la GADF e GASF
        if type == 'gadf':
            gadf = GADF(self.__size)
            return gadf.fit_transform(phi.transpose())

        if type == 'gasf':
            gasf = GASF(self.__size)
            return gasf.fit_transform(phi.transpose())

    '''
    ' Questa funzione calcola le coordinate la GADF e la GASF per un data_frame.
    ' Si da per scontato che il Data Frame passato alla funzione sia gia' un sotto-insieme
    ' del Dataset originale (dovrebbe essere calcolato su un segnale ridotto, ad esempio 40 ore precedenti)
    ' La dimensione dell'immagine viene specificata dall'atrributo self.__size della classe
    ' Come prima cosa i vari segnali vengono convertiti in un range [-1, 1], dopodiche'
    ' si convertono i valori in coordinate polari. Infine si calcolano le trasformate.
    '''
    def __calculate_label_gaf(self, df, type):
        # Scalo tutti i valori del Dataset, inserendoli in un range [-1, 1]
        min_ = np.amin(df)
        max_ = np.amax(df)


        scaled_serie = (2 * df - max_ - min_) / (max_ - min_)

        # Floating point inaccuracy!
        scaled_serie = np.where(scaled_serie >= 1., 1., scaled_serie)
        scaled_serie = np.where(scaled_serie <= -1., -1., scaled_serie)

        # Calcolo le coordinate polari
        phi = np.arccos(scaled_serie)
        # La computazione di r non e' necessaria ai fini della GADF GASF,
        # serve se si vuole plottare il grafico con le C.P.
        # r = np.linspace(0, 1, len(scaled_serie))

        # Calcolo la GADF e GASF
        if type == 'gadf':
            gadf = GADF(self.__size)
            return gadf.fit_transform(phi)

        if type == 'gasf':
            gasf = GASF(self.__size)
            return gasf.fit_transform(phi)

    '''
    ' Passo alla funzione le gaf (gasf o gadf), la data di riferimento di questa trasformata
    ' la lista delle colonne su cui ho costruito le varie gaf ed infine il tipo (gadf o gasf)
    ' Le immagini vengono salvate usando questa struttura: 
    ' gaf_path definita dal costruttore
    ' nome della colonna su cui è generata la gaf
    ' data a cui si riferisce l'immagine .png
    '''
    def __save_pics(self, x_gaf, date, columns_list, type):
        for index, value in enumerate(columns_list):
            full_path = ''

            if type == 'gadf':
                full_path = self.__gadf_path + value + '/'

            if type == 'gasf':
                full_path = self.__gasf_path + value + '/'

            # Se le cartella del dataset non esistono, la creo a runtime
            if not os.path.isdir(full_path):
                os.makedirs(full_path)

            # Salvo l'immagine come png
            matplotlib.image.imsave(full_path + str(date.date()) + '.png', x_gaf[index], cmap='rainbow')


    '''
    ' Prendo come primo parametro il tipo di trasformata che voglio calcolare
    ' Il secondo parametro specifica la dimensione dell'immagine finale e anche 
    ' il numero di righe precenti da prendere per ogni giro di for
    ' Partendo dall'ultima riga sino alla prima, calcola le gadf o le gasf
    ' Calcola per l'intero dataset
    ' Con il parametro columns, seleziono solo le colonne che mi interessano nel dataframe
    '''
    def run(self, df, dataset_name, subfolder, type, size, columns=None):

        self.__validate(df=df, dataset_name=dataset_name, subfolder=subfolder, type=type, size=size)

        # Tecnicamente non dovrebbero esserci valori na, per sicurezza rieseguo
        df = self.__df.dropna().reset_index()

        # Se ho inserito il parametro columns, allora seleziono solo le colonne che mi interessano nel dataframe
        if columns is not None and isinstance(columns, list):
            if 'date_time' not in columns:
                columns.append("date_time") 

            df = df[columns]

        # Prendo la data dell'ultima riga e ci sommo un giorno.
        # In questo modo all'interno del ciclo posso effettuare un controllo sul giorno corrente
        # ed assicurarmi di calcolare le gadf e le gasf una volta per giorno. E' importante perché
        # se ho raggruppato il dataset per 1 ora, avrei 24 righe per giorno e di conseguenza lo script proverebbe
        # a calcolare 24 diverse immagini per giorno. Invece, aggiungendo il controllo sulla data, skippo tutte le righe
        # che hanno la stessa data del giorno che ho già calcolando, shiftando semplicemente di 1 giorno ogni volta
        # che calcolo le gadf o gasf
        last_iterated_day = (df['date_time'][df.index[-1]]).date() + timedelta(days=1)

        # Itero per ogni riga. Per ogni giorno calcolo a ritrovo le GADF e GASF
        # in base ai parametri specificati in precedenza
        for index, row in df[::-1].iterrows():

            # Ottengo il giorno corrente della riga
            current_day = row['date_time'].date()

            # Come specificato nel commento alla variabile last_iterated_day
            # controllo se ho già calcolato questo giorno controllando che il giorno corrente
            # sia maggiore o uguale all'ultimo iterato.
            if current_day >= last_iterated_day:
                continue

            # Uso la maschera per prendere solo le righe a partire dalla data corrente
            mask = df['date_time'] <= row['date_time']
            subset = df.loc[mask]
            # Rimuovo la colonna date_time che non mi serve per il calcolo della trasformata
            subset = subset.drop('date_time', axis=1)
            # Ottengo un sottoinsieme del DataFrame, prendendo le n righe specificate con size
            df_range = subset.tail(self.__size)

            # Controllo se la dimensione e' la stessa. Se non lo è vuol dire che il ciclo
            # e' arrivato quasi alla fine, quindi blocco l'esecuzione
            if df_range.shape[0] == self.__size:

                # Calcolo le trasformate
                if type == 'gadf':
                    x_gadf = self.__calculate_gaf(df=df_range, type=type)
                    self.__save_pics(x_gadf, row['date_time'], list(df_range), type=type)
                else:
                    x_gasf = self.__calculate_gaf(df=df_range, type=type)
                    self.__save_pics(x_gasf, row['date_time'], list(df_range), type=type)

                last_iterated_day = last_iterated_day - timedelta(days=1)
            # Se tail restituisce un subset di dimensione minore a quello specificato in size, devo uscire
            else:
                break

    '''
    ' Prendo come primo parametro il tipo di trasformata che voglio calcolare
    ' Il secondo parametro specifica la dimensione dell'immagine finale e anche 
    ' il numero di righe precenti da prendere per ogni giro di for
    ' Partendo dall'ultima riga sino alla prima, calcola le gadf o le gasf
    ' Calcola per l'intero dataset
    '''
    def label_run(self, df, dataset_name, subfolder, type, size):

        self.__validate(df=df, dataset_name=dataset_name, subfolder=subfolder, type=type, size=size)

        # Tecnicamente non dovrebbero esserci valori na, per sicurezza rieseguo
        df = self.__df.dropna().reset_index()

        # Prendo la data dell'ultima riga e ci sommo un giorno.
        # In questo modo all'interno del ciclo posso effettuare un controllo sul giorno corrente
        # ed assicurarmi di calcolare le gadf e le gasf una volta per giorno. E' importante perché
        # se ho raggruppato il dataset per 1 ora, avrei 24 righe per giorno e di conseguenza lo script proverebbe
        # a calcolare 24 diverse immagini per giorno. Invece, aggiungendo il controllo sulla data, skippo tutte le righe
        # che hanno la stessa data del giorno che ho già calcolando, shiftando semplicemente di 1 giorno ogni volta
        # che calcolo le gadf o gasf
        last_iterated_day = (df['date_time'][df.index[-1]]).date() + timedelta(days=1)

        # Itero per ogni riga. Per ogni giorno calcolo a ritrovo le GADF e GASF
        # in base ai parametri specificati in precedenza
        for index, row in df[::-1].iterrows():

            # Ottengo il giorno corrente della riga
            current_day = row['date_time'].date()

            # Come specificato nel commento alla variabile last_iterated_day
            # controllo se ho già calcolato questo giorno controllando che il giorno corrente
            # sia maggiore o uguale all'ultimo iterato.
            if current_day >= last_iterated_day:
                continue

            # Per calcolare la GADF o GASF relativa alla label eseguo un semplice trick
            # Siccome un array di tutto 0 o tutto 1 darebbe la stessa immagine, genero un array
            # crescente o decrescente a seconda della label. Quindi da 0 a size o viceversa.
            # In questo modo posso avere immagini speculari a seconda della classe di appartenenza.
            # L'array dev'essere 2D per motivi di compatibilità con la classe Gaf
            label_array = []
            label_array.append(np.arange(0, size, 1))

            if row['label'] == 0:
                label_array[0] = np.flip(label_array[0])

            # Calcolo le trasformate
            if type == 'gadf':
                x_gadf = self.__calculate_label_gaf(df=label_array, type=type)
                self.__save_pics(x_gadf, row['date_time'], ['label'], type=type)
            else:
                x_gasf = self.__calculate_label_gaf(df=label_array, type=type)
                self.__save_pics(x_gasf, row['date_time'], ['label'], type=type)

            last_iterated_day = last_iterated_day - timedelta(days=1)
            # Se tail restituisce un subset di dimensione minore a quello specificato in size, devo uscire









    '''
    '
    '''
    def save_pics_pattern(self, x_gaf, perc, date, columns_list, type):
        for index, value in enumerate(columns_list):
            full_path = ''
            gadf_path = self.__output_path + self.__dataset_name + '_perc_' + str(perc) + '/' + self.__subfolder + '/' + 'gadf/'
            gasf_path = self.__output_path + self.__dataset_name + '_perc_' + str(perc) + '/' + self.__subfolder + '/' + 'gasf/'

            if type == 'gadf':
                full_path = gadf_path + value + '/'

            if type == 'gasf':
                full_path = gasf_path + value + '/'

            # Se le cartella del dataset non esistono, la creo a runtime
            if not os.path.isdir(full_path):
                os.makedirs(full_path)

            # Salvo l'immagine come png
            matplotlib.image.imsave(full_path + str(date.date()) + '.png', x_gaf[index], cmap='rainbow')

    '''
    ' 09/10/2020
    ' Metodo creato per generare img con 
    ' pattern di intesità A, lavorando su risoluzione 1H
    '
    '''
    def run_delta_change(self, df, label_df, dataset_name, subfolder, type, size, perc=0, columns=None):
        self.__validate(df=df, dataset_name=dataset_name, subfolder=subfolder, type=type, size=size)

        pattern = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
        if perc > 0:
            pattern = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, # non cambio i primi 15 elementi 
                        1 + (perc / 100), 
                        -1 - (2 * + (perc / 100)), 
                        1 + (3 * + (perc / 100)), 
                        -1 - (2 * + (perc / 100)), 
                        1 + (perc / 100) # cambio le ultime 5 ore
            ]

        # Tecnicamente non dovrebbero esserci valori na, per sicurezza rieseguo
        df = self.__df.dropna().reset_index()

        # Se ho inserito il parametro columns, allora seleziono solo le colonne che mi interessano nel dataframe
        if columns is not None and isinstance(columns, list):
            if 'date_time' not in columns:
                columns.append("date_time") 

            df = df[columns]


        last_iterated_day = (df['date_time'][df.index[-1]]).date() + timedelta(days=1)

        # Itero per ogni riga. Per ogni giorno calcolo a ritrovo le GADF e GASF
        # in base ai parametri specificati in precedenza
        for index, row in df[::-1].iterrows():

            # Ottengo il giorno corrente della riga
            current_day = row['date_time'].date()

            # Come specificato nel commento alla variabile last_iterated_day
            # controllo se ho già calcolato questo giorno controllando che il giorno corrente
            # sia maggiore o uguale all'ultimo iterato.
            if current_day >= last_iterated_day:
                #print("current_day:", current_day, "last_iterated_day:", last_iterated_day)
                continue
            
            label_next_day = label_df.loc[label_df['date_time'] ==  row['date_time'].date()]

            if label_next_day.empty == True:
                #print("[IF label_next_day.empy] Giorno con label == 1", row['date_time'].date())
                #print("Skippo il giorno:", last_iterated_day, label_next_day)
                #last_iterated_day = last_iterated_day - timedelta(days=1)
                #continue
                label_next_day = 1
            else:
                label_next_day = label_next_day['label_next_day'].tolist()[0]
                #print("[IF label_next_day.empy] Giorno con label == ", label_next_day, row['date_time'].date())


            
            # Uso la maschera per prendere solo le righe a partire dalla data corrente
            mask = df['date_time'] <= row['date_time']
            subset = df.loc[mask]
            # Rimuovo la colonna date_time che non mi serve per il calcolo della trasformata
            subset = subset.drop('date_time', axis=1)
            # Ottengo un sottoinsieme del DataFrame, prendendo le n righe specificate con size
            df_range = subset.tail(self.__size)
            
            
            if label_next_day == 0: 
                #print("[IF label_next_day == 0] Giorno con label == 0:", current_day)
                df_range['pattern'] = pattern
                df_range['delta_current_day_percentage'] = df_range['delta_current_day_percentage'] * df_range['pattern']
                df_range = df_range.drop('pattern', 1)
                #df_range = df_range.drop('delta_current_day_percentage', 1)
                #df_range = df_range.rename(columns={"new_delta": "delta_current_day_percentage"})
                #print(df_range)
                #print(current_day, label_next_day)
                #input()
            
            # Controllo se la dimensione e' la stessa. Se non lo è vuol dire che il ciclo
            # e' arrivato quasi alla fine, quindi blocco l'esecuzione
            if df_range.shape[0] == self.__size:
                
                # Calcolo le trasformate
                if type == 'gadf':
                    x_gadf = self.__calculate_gaf(df=df_range, type=type)
                    self.save_pics_pattern(x_gaf=x_gadf, perc=perc, date=row['date_time'], columns_list=list(df_range), type=type)
                else:
                    x_gasf = self.__calculate_gaf(df=df_range, type=type)
                    self.save_pics_pattern(x_gaf=x_gasf, perc=perc, date=row['date_time'], columns_list=list(df_range), type=type)
                
                last_iterated_day = last_iterated_day - timedelta(days=1)
        
            # Se tail restituisce un subset di dimensione minore a quello specificato in size, devo uscire
            else:
                break
    
    '''
    ' Funziona con img di 96x96
    '''
    def run_delta_change_5_min(self, df, label_df, dataset_name, subfolder, type, size, perc=0, columns=None):
        self.__validate(df=df, dataset_name=dataset_name, subfolder=subfolder, type=type, size=size)

        pattern_a = np.ones(222) #91 #222 
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


        # Tecnicamente non dovrebbero esserci valori na, per sicurezza rieseguo
        df = self.__df.dropna().reset_index()

        # Se ho inserito il parametro columns, allora seleziono solo le colonne che mi interessano nel dataframe
        if columns is not None and isinstance(columns, list):
            if 'date_time' not in columns:
                columns.append("date_time") 

            df = df[columns]


        last_iterated_day = (df['date_time'][df.index[-1]]).date() + timedelta(days=1)

        # Itero per ogni riga. Per ogni giorno calcolo a ritrovo le GADF e GASF
        # in base ai parametri specificati in precedenza
        for index, row in df[::-1].iterrows():

            # Ottengo il giorno corrente della riga
            current_day = row['date_time'].date()

            # Come specificato nel commento alla variabile last_iterated_day
            # controllo se ho già calcolato questo giorno controllando che il giorno corrente
            # sia maggiore o uguale all'ultimo iterato.
            if current_day >= last_iterated_day:
                #print("current_day:", current_day, "last_iterated_day:", last_iterated_day)
                continue
            
            label_next_day = label_df.loc[label_df['date_time'] ==  row['date_time'].date()]

            if label_next_day.empty == True:
                #print("[IF label_next_day.empy] Giorno con label == 1", row['date_time'].date())
                #print("Skippo il giorno:", last_iterated_day, label_next_day)
                #last_iterated_day = last_iterated_day - timedelta(days=1)
                #continue
                label_next_day = 1
            else:
                label_next_day = label_next_day['label_next_day'].tolist()[0]
                #print("[IF label_next_day.empy] Giorno con label == ", label_next_day, row['date_time'].date())


            
            # Uso la maschera per prendere solo le righe a partire dalla data corrente
            mask = df['date_time'] <= row['date_time']
            subset = df.loc[mask]

            # Rimuovo la colonna date_time che non mi serve per il calcolo della trasformata
            subset = subset.drop('date_time', axis=1)
            # Ottengo un sottoinsieme del DataFrame, prendendo le n righe specificate con size
            df_range = subset.tail(self.__size)
            
            
            if label_next_day == 0: 
                #print("[IF label_next_day == 0] Giorno con label == 0:", current_day)
                df_range['pattern'] = pattern
                df_range['delta_current_day_percentage'] = df_range['delta_current_day_percentage'] * df_range['pattern']
                df_range = df_range.drop('pattern', 1)
                #df_range = df_range.drop('delta_current_day_percentage', 1)
                #df_range = df_range.rename(columns={"new_delta": "delta_current_day_percentage"})
                #print(df_range)
                #print(current_day, label_next_day)
                #input()
            
            # Controllo se la dimensione e' la stessa. Se non lo è vuol dire che il ciclo
            # e' arrivato quasi alla fine, quindi blocco l'esecuzione
            if df_range.shape[0] == self.__size:
                
                # Calcolo le trasformate
                if type == 'gadf':
                    x_gadf = self.__calculate_gaf(df=df_range, type=type)
                    self.save_pics_pattern(x_gaf=x_gadf, perc=perc, date=row['date_time'], columns_list=list(df_range), type=type)
                else:
                    x_gasf = self.__calculate_gaf(df=df_range, type=type)
                    self.save_pics_pattern(x_gaf=x_gasf, perc=perc, date=row['date_time'], columns_list=list(df_range), type=type)
                
                last_iterated_day = last_iterated_day - timedelta(days=1)
        
            # Se tail restituisce un subset di dimensione minore a quello specificato in size, devo uscire
            else:
                break
        
          