import os
import sys
import pandas as pd
import numpy as np
# Importo il file di configurazione del database

'''
' @Author Andrea Corriga
' Questa classe rappresenta, come oggetto, un dataset
' Può essere letto da DB remoto se necessario o da file locale
'''

class Market:

    # Path locale dei datasets
    dataset_path = '../datasets/'

    
    # Colonne presenti nel dataset originale
    original_columns = ['date', 'time', 'open', 'close', 'close_adj', 'high', 'low', 'up', 'down', 'volume']

    # Dataframe di lavoro
    df = pd.DataFrame()

    # Costruttore, setto la connessione al Database
    def __init__(self, dataset):
        self.__read(dataset)

    '''
    ' Leggo un dataset specificato come parametro
    ' Aggiungo una colonna date_time unendo i campi date e time
    '''
    def __read(self, dataset):
        if dataset == None:
            sys.exit("Market.validate: Dataset can't be none")

        df = pd.read_csv(self.dataset_path + dataset + '.csv').set_index('id')
        df['date_time'] = pd.to_datetime(df['date'] + ' ' + df['time'], format='%Y-%m-%d %H:%M:%S')
        self.df = df

        if 'delta' not in self.df.columns:
            self.__get_delta()
            #self.__get_delta_percentage()
            #self.__get_delta_percentage_previous_day()

        # Delta percentage da numeri troppo piccoli, disattivato per ora
        #self.__get_delta_percentage()

    '''
    ' utilizzata come funzione d'appoggio nelle groupby. 
    ' Prendo il secondo elemento di una lista, in questo caso dell'open
    ''' 
    def get_open_from_close(self, x):
        # [2] => sembra essere corretto prendere il 3° elemento
        return x.iat[1] if len(x) > 1 else np.nan

    '''
    ' Restituisco il dataset raggruppato per una risoluzione specifica
    ' eliminando le colonne date e time
    ' per open e close prendo rispettivamente il secondo elemento ed
    ' il penultimo, per allineare i risultati con multicharts
    ' @param freq -> 1m, 1h, 1d
    '''
    def group(self, freq, nan=False):

        if 'close_adj' in  self.df.columns:
            # Get first Open, Last Close, Highest value of "High" and Lower value of "Low", and sum(Volume)
            grouped = self.df.drop(['date', 'time', 'delta_current_day'], axis=1).groupby(pd.Grouper(key='date_time', freq=freq), sort=True).agg({
                'close': 'last', # penultimo self.__group_second_last
                'close_adj': [self.get_open_from_close, 'last'],
                'high': 'max',
                'low': 'min',
                'up': 'sum',
                'down': 'sum',
                'volume': 'sum'
            })
        else:
            # Get first Open, Last Close, Highest value of "High" and Lower value of "Low", and sum(Volume)
            grouped = self.df.drop(['date', 'time', 'delta_current_day', 'delta_next_day'], axis=1).groupby(pd.Grouper(key='date_time', freq=freq), sort=True).agg({
                #'open': self.__group_open(grouped.date_time), #secondo elemento
                'close': [self.get_open_from_close, 'last'], # penultimo self.__group_second_last
                'high': 'max',
                'low': 'min',
                'up': 'sum',
                'down': 'sum',
                'volume': 'sum'
            })

        
        grouped.columns = ["_".join(x) for x in grouped.columns.ravel()]
        grouped = grouped.rename(columns={"close_get_open_from_close": "open", "close_last": "close", "high_max": "high", "low_min": "low", "up_sum": "up", "down_sum": "down", "volume_sum": "volume", })
        
        #grouped['delta'] = self.__get_delta_overload(df=grouped)
        #grouped['delta_percentage'] = self.__get_delta_percentage_overload(df=grouped)
        #grouped['delta_percentage_previous_day'] = self.__get_delta_percentage_previous_day_overload(df=grouped)
        grouped = self.__get_delta_overload(df=grouped)
        
        #self.__get_delta_percentage_overload(df=grouped)
        #self.__get_delta_percentage_previous_day_overload(df=grouped)

        if nan == True:
            return grouped
        else:
            return grouped


    '''
    ' Restituisce una copia del dataset, con meno colonne
    ' le colonne da rimuovere sono passate come parametro
    ' Se vengono passate colonne (dentro la lista) che non sono ammesse
    ' lo script darà errore. 
    ' E' possibile specificare anche una risoluzione temporale, per avere
    ' un subset del dataset specifico
    '''
    def remove_columns(self, columns=[], freq=None):
        if type(columns) is not list:
            sys.exit("Market.remove_columns: The columns parameter is not a list")

        # Calculate the difference in order to check if columns contains valid values
        difference = list(set(columns) - set(self.original_columns))
        
        # If difference > 0 it means that some columns passed as parameter is not accepted
        if len(difference) > 0:
            sys.exit("Market.remove_columns: The columns passed as parameters is not correct")

        if freq is not None:
            df = self.group(freq)
        else:
            df = self.df

        return df.drop(columns, axis=1)


    '''
    ' Restituisce la label del dataset (ovvero i valori -1 e 1)
    ' per una specifica risoluzione temporale. La label viene calcolata 
    ' controllando la differenza tra close di quella riga rispetto a quella precedente
    ' Se la differenza è positiva la label sarà 1, -1 altrimenti. 
    ' Si può specificare anche quali colonne tenere nel dataset (oltre la data)
    ' e la risoluzione temporale. Se si raggruppa per ore quindi si avrà la label ora per ora
    '''
    def get_label_next_day_using_close(self, freq=None, columns=[]):

        if freq is not None:
            df = self.group(freq)
        else:
            df = self.df
        
        df['label'] = -1

        # Calcolo per ogni riga la Label, ovvero se il giorno dopo il mercato
        # salira' o scendera'
        df['label'] = (df['close'].diff().shift(-1) > 0).astype(int)
        # Converto lo 0 in -1, visto che lo 0 lo interpreteremo come hold
        df.loc[df['label'] == 0, ['label']] = -1
        # Invece di mettere la label sul giorno precedente, la mette sul giorno corrente. Può essere utile per altri scopi ma per ora è deprecata
        #df['label'] = (df['close'].diff().bfill() > 0).astype(int)
        
        # elimino le colonne inutili
        if type(columns) is not list:
            sys.exit("Market.remove_columns: The columns parameter is not a list")

        #columns.append('label')
        
        if len(columns) > 0:
            columns.append('label')
            df = df.filter(columns, axis=1)
        else:
            df = df.filter(['label'], axis=1) 

        return df


    '''
    '
    '''
    def get_label(self, freq=None, thr=0.3, next_days=5, columns=[]):
        if freq is not None:
            df = self.group(freq)
        else:
            df = self.df

        #df['label_next_week'] = df['label'] = (df['close'].diff().shift(-5) > 0).astype(int)
        #df['label_next_week'] = df['close'].diff(5).shift(-5) # CLOSE - CLOSE DI GIORNI DIVERSI

        df['variazione_percentuale'] =  ((df['close'].shift(-next_days) - df['open']) / df['open']) * 100 # CLOSE TOT GIORNI IN AVANTI - open del giorno corrente in percentuale
        df['variazione_esatta_next_days'] =  (df['close'].shift(-next_days) - df['open']) # [debug] CLOSE TOT GIORNI IN AVANTI - open del giorno corrente
        df['label_next_day'] = 1 # hold
        df.loc[df['variazione_percentuale'] > thr, ['label_next_day']] = 2 # long
        df.loc[df['variazione_percentuale'] < - thr, ['label_next_day']] = 0 # short
        
        df['label_current_day'] = 1
        
        print("---------------------------------------")
        hold_totali = len(df[(df.label_next_day == 1)])
        long_totali = len(df[(df.label_next_day == 2)])
        short_totali = len(df[(df.label_next_day == 0)])
        
        op_totali = hold_totali + long_totali + short_totali
        print("Soglia hold: ", thr)
        print("long totali: ", long_totali, " - long in %: ", round( ((long_totali / op_totali) * 100), 2))
        print("hold totali: ", hold_totali, " - hold in %: ", round( ((hold_totali / op_totali) * 100), 2))
        print("short totali: ", short_totali, " - short in %: ", round( ((short_totali / op_totali) * 100), 2))
        print("Operazioni totali: ", op_totali)
        print(df.shape)
        print("---------------------------------------")
        
        
        # Controllo che il parametro columns sia una lista prima di selezionare le colonne
        if type(columns) is not list:
            sys.exit("Market.remove_columns: The columns parameter is not a list")

        #columns.append('label')
        # elimino le colonne inutili
        if len(columns) > 0:
            columns.append('label_next_day')
            columns.append('label_current_day')
            
            df = df.filter(columns, axis=1)
        else:
            df = df.filter(['label_next_day', 'label_current_day'], axis=1) 

        return df.dropna()
    
    '''
    ' Restituisce la label del dataset (ovvero i valori -1 e 1)
    ' per una specifica risoluzione temporale. La label viene calcolata 
    ' come close - open
    ' Se la differenza è positiva la label sarà 1, -1 altrimenti. 
    ' Si può specificare anche quali colonne tenere nel dataset (oltre la data)
    ' e la risoluzione temporale. Se si raggruppa per ore quindi si avrà la label ora per ora
    '' DEPRECATED
    def get_label_current_day(self, freq=None, columns=[]):

        if freq is not None:
            df = self.group(freq)
        else:
            df = self.df
        
        df['label'] = 1

        # Calcolo per ogni riga la Label, ovvero se il giorno dopo il mercato
        # salira' o scendera'
        df['label'] = (df['close'] - df['open'] > 0).astype(int)
        
        # Converto lo 0 in -1, visto che lo 0 lo interpreteremo come hold
        df.loc[df['label'] == 0, ['label']] = -1
        
        # Controllo che il parametro columns sia una lista prima di selezionare le colonne
        if type(columns) is not list:
            sys.exit("Market.remove_columns: The columns parameter is not a list")

        #columns.append('label')
        # elimino le colonne inutili
        if len(columns) > 0:
            columns.append('label')
            df = df.filter(columns, axis=1)
        else:
            df = df.filter(['label'], axis=1) 

        return df
    '''


    '''
    ' Restituisce il dataset per intero, così com'è presente nell'istanza
    ' della classe
    '''
    def get(self):
        return self.df

    '''
    ' Aggiungo al dataset una colonna delta, dato dalla 
    ' differenza tra close - open. Se esiste la colonna close_adj, utilizzo
    ' quella colonna
    '''
    def __get_delta(self):
        
        if 'close_adj' in self.df.columns:
            self.df['delta_current_day'] = self.df['close_adj'] - self.df['open']
            #self.df = self.df[pd.notnull(self.df['delta_current_day'])]

            self.df = self.df[pd.notnull(self.df['delta_current_day'])].copy()
            self.df['delta_next_day'] = self.df['delta_current_day'].shift(-1)
        else:
            self.df['delta_current_day'] = self.df['close'] - self.df['open']
            #self.df = self.df[pd.notnull(self.df['delta_current_day'])]
            
            self.df = self.df[pd.notnull(self.df['delta_current_day'])].copy()
            self.df['delta_next_day'] = self.df['delta_current_day'].shift(-1)

    # @overload del metodo get_delta(), invece di lavorare sul dataset locale dell'istanza dell'oggetto
    # lavoro sul dataset passato come parametro. Non essendoci di sua natura l'overfload, aggiungo un _overload
    def __get_delta_overload(self, df): 
        if 'close_adj' in df.columns:
            df['delta_current_day'] = df['close_adj'] - df['open']
            #df = df[pd.notnull(df['delta_current_day'])]
            
            df = df[pd.notnull(df['delta_current_day'])].copy()
            df['delta_next_day'] = df['delta_current_day'].shift(-1)
        else:
            df['delta_current_day'] = df['close'] - df['open']
            #df = df[pd.notnull(df['delta_current_day'])]

            df = df[pd.notnull(df['delta_current_day'])].copy()
            df['delta_next_day'] = df['delta_current_day'].shift(-1)

        return df

    '''
    ' Aggiungo al dataset una colonna delta, dato dalla 
    ' differenza tra close - open / open. 
    ' Se esiste la colonna close_adj, utilizzo quella colonna
    '''
    def __get_delta_percentage(self):
        if 'close_adj' in self.df.columns:
            self.df['delta_percentage'] = ((self.df['close_adj'] - self.df['open']) / self.df['open'] ) * 100
        else:
            self.df['delta_percentage'] = ((self.df['close'] - self.df['open']) / self.df['open']) * 100

    # @override del metodo get_delta_percentage(), invece di lavorare sul dataset locale dell'istanza dell'oggetto
    # lavoro sul dataset passato come parametro
    def __get_delta_percentage_overload(self, df):
        if 'close_adj' in df.columns:
            return ((df['close_adj'] - df['open']) / df['open']) * 100
        else:
            return ((df['close'] - df['open']) / df['open'] ) *100

    '''
    ' Aggiungo al dataset una colonna delta_percentage
    ' data dalla differenza in percentuale tra due close
    ' ovvero riga attuale e riga preceddente
    '''
    def __get_delta_percentage_previous_day(self):
        if 'close_adj' in self.df.columns:
            self.df['delta_percentage_previous_day'] = self.df['close_adj'].pct_change() * 100
        else:
            self.df['delta_percentage_previous_day'] = self.df['close'].pct_change() * 100

    # @override del metodo delta_percentage_previous_day(), invece di lavorare sul dataset locale dell'istanza dell'oggetto
    # lavoro sul dataset passato come parametro
    def __get_delta_percentage_previous_day_overload(self, df):
        if 'close_adj' in df.columns:
            return df['close_adj'].pct_change() * 100
        else:
            return df['close'].pct_change() * 100
    
    '''
    ' Restituisco una copia del dataset
    ' filtrato per data. 
    ' Prendo in ingresso data di inizio e data di fine per effettuale
    ' la search
    '''
    @staticmethod
    def get_df_by_data_range(df, start_date, end_date):
        df = df.reset_index()
        if 'date_time' in df.columns:
            # Search mask
            mask = (df['date_time'] >= start_date) & (df['date_time'] <= end_date)
            # Get the subset of sp500
            return df.loc[mask]
        
        if 'date' in df.columns:
            # Search mask
            mask = (df['date'] >= start_date) & (df['date'] <= end_date)
            # Get the subset of sp500
            return df.loc[mask]

        if 'date_time' not in df.columns and 'date' not in df.columns:
           return df

    
    @staticmethod
    def get_df_until_data(df, end_date):
        df = df.reset_index()
        if 'date_time' in df.columns:
            # Search mask
            mask = (df['date_time'] <= end_date)
            # Get the subset of sp500
            return df.loc[mask]
        
        if 'date' in df.columns:
            # Search mask
            mask = (df['date'] <= end_date)
            # Get the subset of sp500
            return df.loc[mask]

        if 'date_time' not in df.columns and 'date' not in df.columns:
           return df


    '''
    '
    '''
    @staticmethod
    def get_delta_penalty_stop_loss(df, stop_loss=1000, penalty=0, multiplier=50, delta_to_use='delta_current_day'):
        stop_loss_point = stop_loss / multiplier

        if penalty > 0: 
            penalty_points = penalty / multiplier
        else:
            penalty_points = 0

        conditions = [ (df['low'] - df['open'] <= - stop_loss_point)  & (df['decision'] == 2), (df['low'] - df['open'] >= stop_loss_point) & (df['decision'] == 0), (df['low'] - df['open'] < stop_loss_point) & (df['decision'] == 0)]
        choices = [(-stop_loss_point - penalty_points) , (stop_loss_point + penalty_points),  (df[delta_to_use] + penalty_points)]

        df['delta_penalty_stop_loss'] = np.select(conditions, choices, default=((df[delta_to_use] - penalty_points)))

        return df

    '''
    '
    '''
    @staticmethod
    def get_delta_penalty(df, penalty=0, multiplier=50, delta_to_use='delta_current_day'):
        if penalty > 0: 
            penalty_points = penalty / multiplier
        else:
            penalty_points = 0

        conditions = [ df['decision'] == 2, df['decision'] == 0]
        choices = [(df[delta_to_use] - penalty_points),  (df[delta_to_use] + penalty_points)]

        df['delta_penalty'] = np.select(conditions, choices)

        return df