import os
import sys
import pandas as pd

# Importo il file di configurazione del database
from config import db

'''
' @Author Andrea Corriga
' Questa classe rappresenta, come oggetto, un dataset
' Può essere letto da DB remoto se necessario o da file locale
'''

class Downloader:

    # Path locale dei datasets
    dataset_path = '../datasets/'

    # Colonne presenti nel dataset originale
    original_columns = ['date', 'time', 'open', 'close', 'close_adj', 'high', 'low', 'up', 'down', 'volume']

    # Dataframe di lavoro
    df = pd.DataFrame()

    # Costruttore, setto la connessione al Database
    def __init__(self):
        self.db_connection = db.datasets_conn

    '''
    ' @private 
    ' Prima di leggere o inizializzare
    ' controllo che il dataset specificato sia accettato dall'array 
    ' e che non sia un parametro vuoto
    '''
    def __validate(self, dataset):
        # Se non specifico il dataset esco dalla funzione
        if dataset == None:
            sys.exit("Downloader.validate: Dataset can't be none")

    '''
    ' @public
    ' Questa funzione viene lanciata se non è presente
    ' il datasets in locale. Viene letto da mysql e salvato
    ' nella cartella dataset_path
    '''
    def run(self, dataset=None):
        self.__validate(dataset)

        # Leggo da mysql il dataset che mi serve salvarmi in locale
        print("Downloader.__initialize: Reading the dataset '" + dataset + "' from Mysql... It can take a while...")
        df = pd.read_sql('SELECT * FROM ' + dataset, con=self.db_connection)
        df = df.set_index('id')

        # Casto il campo Time. Leggendo da mysql viene convertito in timedelta e non in time.
        df['time'] = pd.to_datetime(df['time']).dt.time

        # Se la cartella del dataset non esiste, la creo a runtime
        if not os.path.isdir(self.dataset_path):
            os.makedirs(self.dataset_path)

        # Salvo in csv quando appena letto
        df.to_csv(self.dataset_path + dataset + ".csv", header=True, index=True, date_format=str)