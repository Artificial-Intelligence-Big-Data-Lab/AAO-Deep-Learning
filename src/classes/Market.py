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

    '''
    ' classic ternary 
    ''
    long_value = 2
    hold_value = 1
    short_value = 0
    '''

    ''' Binary'''
    long_value = 1
    hold_value = -2
    short_value = 0


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
            self.__get_delta_percentage()
            self.__get_delta_percentage_previous_day()

        if 'volume' not in self.df.columns:
            self.df['volume'] = 0
        if 'up' not in self.df.columns:
            self.df['up'] = 0
        if 'down' not in self.df.columns:
            self.df['down'] = 0

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
    ' Calcolo la colonna low togliendo le prime due riga
    ' la prima perché multicharts entra alla seconda riga 
    ' la seconda perché non può entrare in stop loss nello stesso momento in cui entra (?)
    ' visto che non viene calcolata da multicharts
    '''
    def get_low_column(self, x):
        x = x[1:] # PROVARE ANCHE 2 PER IL MOTIVO DI CUI SOPRA
        return x.min() if len(x) > 1 else np.nan

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
                #'close': 'last', # penultimo self.__group_second_last
                'close_adj': [self.get_open_from_close, 'last'],
                'high': 'max',
                'low': [self.get_low_column],
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
                'low': [self.get_low_column],
                'up': 'sum',
                'down': 'sum',
                'volume': 'sum',
                #'close_hack': [self.get_open_from_close, 'last'],
                #'close_hack_2': [self.get_open_from_close, 'last']
            })

        
        grouped.columns = ["_".join(x) for x in grouped.columns.ravel()]

        if 'close_adj' in  self.df.columns:
            grouped = grouped.rename(columns={"close_adj_get_open_from_close": "open", "close_adj_last": "close", "high_max": "high", "low_get_low_column": "low", "up_sum": "up", "down_sum": "down", "volume_sum": "volume" })
        else:
            grouped = grouped.rename(columns={"close_get_open_from_close": "open", "close_last": "close", "high_max": "high", "low_get_low_column": "low", "up_sum": "up", "down_sum": "down", "volume_sum": "volume"})
                #'close_hack_get_open_from_close': 'open_hack', 'close_hack_last': 'close_hack', \
                #'close_hack_2_get_open_from_close': 'open_hack_2', 'close_hack_2_last': 'close_hack_2' 
            #grouped = grouped.rename(columns={"close_get_open_from_close": "open", "close_last": "close", "high_max": "high", "low_min": "low" })
        

        #grouped['delta'] = self.__get_delta_overload(df=grouped)
        #grouped['delta_percentage'] = self.__get_delta_percentage_overload(df=grouped)
        grouped = self.__get_delta_overload(df=grouped)
        grouped = self.__get_delta_percentage_overload(df=grouped)
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
    ' Restituisce la label del dataset (ovvero i valori `-1` e `1`) per una specifica risoluzione temporale. 
    ' La label viene calcolata come `close` - `open` sul **giorno successivo**. Se la differenza è positiva la label sarà `1`, `-1` altrimenti. 
    ' Si può specificare anche quali colonne tenere nel dataset (oltre la data) e la risoluzione temporale. 
    ' Se si raggruppa per ore quindi si avrà la label ora per ora
    ' Scegliendo come frequenza `freq=1d` si otterranno le label per l'andamento del mercato giornaliero. 
    ' Il campo `columns` serve a specificare quali altre colonne si vogliono inserire nel DF restituito dal metodo. 
    ' Di default il metodo restituisce solamente il campo `date` e il campo `label`.
    '''
    def get_label(self, freq=None, thr=0.3, columns=[]):
        if freq is not None:
            df = self.group(freq)
        else:
            df = self.df

        ''' LABEL TERNARIA ''' 
        df['co'] = ((df['close'] - df['open'] ) / df['open'] ) * 100
        df['label_current_day'] = 1 # hold
        df.loc[df['co'] > thr, ['label_current_day']] = self.long_value # long
        df.loc[df['co'] < - thr, ['label_current_day']] = self.short_value # short

        df['label_next_day'] = df['label_current_day']
        df['label_next_day'] = df['label_next_day'].shift(-1)
        df = df[:-1]
        df['label_next_day'] = df['label_next_day'].astype(int)

        '''
        print("---------------------------------------")
        hold_totali = len(df[(df.label_next_day == self.hold_value)])
        long_totali = len(df[(df.label_next_day == self.long_value)])
        short_totali = len(df[(df.label_next_day == self.short_value)])
        
        op_totali = hold_totali + long_totali + short_totali

        print("long totali: ", long_totali, " - long in %: ", (long_totali / op_totali) * 100)
        print("hold totali: ", hold_totali, " - hold in %: ", (hold_totali / op_totali) * 100)
        print("short totali: ", short_totali, " - short in %: ", (short_totali / op_totali) * 100)
        print("Operazioni totali: ", op_totali)
        print(df.shape)
        print("---------------------------------------")
        '''

        ''' BINARY
        df['label'] = 1

        # Calcolo per ogni riga la Label, ovvero se il giorno dopo il mercato
        # salira' o scendera'
        df['label'] = (df['close'] - df['open'] > 0).astype(int)
        
        # Converto lo 0 in -1, visto che lo 0 lo interpreteremo come hold
        df.loc[df['label'] == 0, ['label']] = -1
        
        # shifto all'indietro la label, perché voglio che per il giorno x mi dia la label di x+1
        df['label'] = df['label'].shift(-1)
        

        # rimuovo l'ultima riga visto che non conterrà una label
        df = df[:-1]
        '''
        
        # Controllo che il parametro columns sia una lista prima di selezionare le colonne
        if type(columns) is not list:
            sys.exit("Market.get_label: The columns parameter is not a list")

        #columns.append('label')
        # elimino le colonne inutili
        if len(columns) > 0:
            columns.append('label_next_day')
            columns.append('label_current_day')
            df = df.filter(columns, axis=1)
        else:
            df = df.filter(['label_next_day', 'label_current_day'], axis=1) 

        return df

    '''
    ' Restituisce un dataset con label binaria
    ' 1:  [(c - o) / o] > thr & [(c - o) / o] < -thr
    ' 0: -thr < [(c - o) / o] < thr 
    ''
    def get_binary_labels(self, freq=None, thr=3, columns=[]):
        if freq is not None:
            df = self.group(freq)
        else:
            df = self.df

        # Label binary 
        df['co'] = ((df['close'] - df['open'] ) / df['open'] ) * 100
        df['label_current_day'] = 0 # hold
        df.loc[df['co'] > thr, ['label_current_day']] = 1 
        df.loc[df['co'] < - thr, ['label_current_day']] = 1 

        df['label_next_day'] = df['label_current_day']
        df['label_next_day'] = df['label_next_day'].shift(-1)
        df = df[:-1]
        df['label_next_day'] = df['label_next_day'].astype(int)

        # Controllo che il parametro columns sia una lista prima di selezionare le colonne
        if type(columns) is not list:
            sys.exit("Market.get_binary_labels: The columns parameter is not a list")

        #columns.append('label')
        # elimino le colonne inutili
        if len(columns) > 0:
            columns.append('label_next_day')
            columns.append('label_current_day')
            df = df.filter(columns, axis=1)
        else:
            df = df.filter(['label_next_day', 'label_current_day'], axis=1) 

        return df

        return df
    '''

    '''
    ' Restituisce un dataset con label binaria
    ' 1:  [(c - o) / o] > thr -> +infinito 
    ' 0: [(c - o) / o] < thr 
    '''
    def get_binary_labels(self, freq=None, thr=-0.5, columns=[]):
        if freq is not None:
            df = self.group(freq)
        else:
            df = self.df

        # Label binary 
        df['co'] = ((df['close'] - df['open'] ) / df['open'] ) * 100
        df['label_current_day'] = -1 # hold
        df.loc[df['co'] > thr, ['label_current_day']] = 1 
        df.loc[df['co'] < thr, ['label_current_day']] = 0 

        df['label_next_day'] = df['label_current_day']
        df['label_next_day'] = df['label_next_day'].shift(-1)
        df = df[:-1]
        df['label_next_day'] = df['label_next_day'].astype(int)

        # Controllo che il parametro columns sia una lista prima di selezionare le colonne
        if type(columns) is not list:
            sys.exit("Market.get_binary_labels: The columns parameter is not a list")

        #columns.append('label')
        # elimino le colonne inutili
        if len(columns) > 0:
            columns.append('label_next_day')
            columns.append('label_current_day')
            df = df.filter(columns, axis=1)
        else:
            df = df.filter(['label_next_day', 'label_current_day'], axis=1) 

        return df
    
    # label binaria su volatilità, quindi 1 > e < soglia. 
    # 0 se compreso nella soglia
    def get_binary_labels_volatility(self, freq=None, thr=0.4, columns=[]):
        if freq is not None:
            df = self.group(freq)
        else:
            df = self.df

        # Label binary 
        df['co'] = ((df['close'] - df['open'] ) / df['open'] ) * 100
        df['label_current_day'] = 0
        df.loc[df['co'] > thr, ['label_current_day']] = 1 
        df.loc[df['co'] < -thr, ['label_current_day']] = 1 

        df['label_next_day'] = df['label_current_day']
        df['label_next_day'] = df['label_next_day'].shift(-1)
        df = df[:-1]
        df['label_next_day'] = df['label_next_day'].astype(int)

        # Controllo che il parametro columns sia una lista prima di selezionare le colonne
        if type(columns) is not list:
            sys.exit("Market.get_binary_labels: The columns parameter is not a list")

        #columns.append('label')
        # elimino le colonne inutili
        if len(columns) > 0:
            columns.append('label_next_day')
            columns.append('label_current_day')
            df = df.filter(columns, axis=1)
        else:
            df = df.filter(['label_next_day', 'label_current_day'], axis=1) 

        return df
    
    # label binaria su volatilità, quindi 1 > e < soglia. 
    # 0 se compreso nella soglia
    def get_binary_labels_volatility_7_days(self, freq=None, thr=0.5, days=7, columns=[]):
        if freq is not None:
            df = self.group(freq)
        else:
            df = self.df

        # Label binary 
        df['diff'] = df['close'].diff(days) # calcolo la differenza a 7 giorni
        df['diff'] = df['diff'].shift(-days) # allineo con il giorno corrente
        df['diff_perc'] = (df['diff'] / df['close']) * 100 # calcolo in percentuale
        df['label_next_day'] = 0
        df.loc[df['diff_perc'] > thr, ['label_next_day']] = 1 # calcolo label 1 > thr
        df.loc[df['diff_perc'] < -thr, ['label_next_day']] = 1 # calcolo label 1 < thr

        '''
        df['co'] = ((df['close'] - df['open'] ) / df['open'] ) * 100
        df['label_current_day'] = 0
        df.loc[df['co'] > thr or df['co'] < -thr, ['label_current_day']] = 1 

        df['label_next_day'] = df['label_current_day']
        df['label_next_day'] = df['label_next_day'].shift(-1)
        df = df[:-1]
        '''
        df['label_next_day'] = df['label_next_day'].astype(int)
        df['label_current_day'] = df['label_next_day'].astype(int)
        df = df.dropna()

        # Controllo che il parametro columns sia una lista prima di selezionare le colonne
        if type(columns) is not list:
            sys.exit("Market.get_binary_labels: The columns parameter is not a list")

        #columns.append('label')
        # elimino le colonne inutili
        if len(columns) > 0:
            columns.append('label_next_day')
            columns.append('label_current_day')
            df = df.filter(columns, axis=1)
        else:
            df = df.filter(['label_next_day', 'label_current_day'], axis=1) 

        return df

    '''
    '
    '''
    def get_label_next_days(self, freq=None, thr=0.3, next_days=5, columns=[]):
        if freq is not None:
            df = self.group(freq)
        else:
            df = self.df

        #df['label_next_week'] = df['label'] = (df['close'].diff().shift(-5) > 0).astype(int)
        #df['label_next_week'] = df['close'].diff(5).shift(-5) # CLOSE - CLOSE DI GIORNI DIVERSI

        df['variazione_percentuale'] =  ((df['close'].shift(-next_days) - df['open']) / df['open']) * 100 # CLOSE TOT GIORNI IN AVANTI - open del giorno corrente in percentuale
        df['variazione_esatta_next_days'] =  (df['close'].shift(-next_days) - df['open']) # [debug] CLOSE TOT GIORNI IN AVANTI - open del giorno corrente
        df['label_next_days'] = 1 # hold
        df.loc[df['variazione_percentuale'] > thr, ['label_next_days']] = self.long_value # long
        df.loc[df['variazione_percentuale'] < - thr, ['label_next_days']] = self.short_value # short

        
        print("---------------------------------------")
        hold_totali = len(df[(df.label_next_days == self.hold_value)])
        long_totali = len(df[(df.label_next_days == self.long_value)])
        short_totali = len(df[(df.label_next_days == self.short_value)])
        
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
            columns.append('label_next_days')
            columns.append('variazione_esatta_next_days')
            
            df = df.filter(columns, axis=1)
        else:
            df = df.filter(['label_next_days', 'variazione_esatta_next_days'], axis=1) 

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
    # lavoro sul dataset passato come parametro. Non essendoci di sua natura l'overload, aggiungo un _overload
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
            self.df['delta_current_day_percentage'] = ((self.df['close_adj'] - self.df['open']) / self.df['open'] ) * 100

            self.df = self.df[pd.notnull(self.df['delta_current_day_percentage'])].copy()
            self.df['delta_next_day_percentage'] = self.df['delta_current_day_percentage'].shift(-1)
        else:
            self.df['delta_current_day_percentage'] = ((self.df['close'] - self.df['open']) / self.df['open']) * 100

            self.df = self.df[pd.notnull(self.df['delta_current_day_percentage'])].copy()
            self.df['delta_next_day_percentage'] = self.df['delta_current_day_percentage'].shift(-1)

    # @override del metodo get_delta_percentage(), invece di lavorare sul dataset locale dell'istanza dell'oggetto
    # lavoro sul dataset passato come parametro
    def __get_delta_percentage_overload(self, df):

        if 'close_adj' in df.columns:
            df['delta_current_day_percentage'] = ((df['close_adj'] - df['open']) / df['open'] ) * 100

            df = df[pd.notnull(df['delta_current_day_percentage'])].copy()
            df['delta_next_day_percentage'] = df['delta_current_day_percentage'].shift(-1)
        else:
            df['delta_current_day_percentage'] = ((df['close'] - df['open']) / df['open']) * 100

            df = df[pd.notnull(df['delta_current_day_percentage'])].copy()
            df['delta_next_day_percentage'] = df['delta_current_day_percentage'].shift(-1)

        return df


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

        if "index" in df.columns:
            df = df.drop(columns="index")

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
    ' Ricalcola il verrore delta prendendo in considerazione sia la stop loss che il delta
    ' Supponendo di avere sempre stop loss 1000 e multiplier 50
    ' Se la stop loss è 0 si comporterà esattamente come get_delta_penalty
    ' Le condizioni per la stop loss sono specificate dentro il metodo
    '''
    @staticmethod
    def get_delta_penalty_stop_loss(df, stop_loss=1000, penalty=0, multiplier=50, delta_to_use='delta_current_day'):
        
        # Se uso il delta current day non ho bisogno di shiftare
        # i dati, mentre se uso quello del giorno successivo
        # per calcolare la stop loss devo controllare che il giorno successivo
        # quindi shift(-1) sia minore/maggiore del valore della stop loss
        shift = 0
        if delta_to_use == 'delta_next_day':
            shift = -1

        # calcolo di quanti punti dev'essere la stop loss. 
        # Se la stop loss è di 1000 ed il mercato è sp500: 1000 / 50 = 20
        stop_loss_point = stop_loss / multiplier
        
        # Calcolo di quanti punti dev'essere la penalty. 
        # Se la penalty è 25$ ed il mercato è sp500: 25 / 50 = 0.5
        penalty_points = 0
        if penalty > 0: 
            penalty_points = penalty / multiplier
        
        # Esempio: sp500 (multiplier = 50) e stop loss 1000$ (20 punti)
        
        # Se l'operazione è long, ed il mercato va sotto di 20 punti (-20)
        long_with_stop_loss = (df['low'].shift(shift) - df['open'].shift(shift) <= -stop_loss_point)  & (df['decision'] == Market.long_value)

        # Se l'operazione è short ed il mercato sale di 20 punti
        short_with_stop_loss = (df['high'].shift(shift) - df['open'].shift(shift) >= stop_loss_point) & (df['decision'] == Market.short_value)

        # Se l'operazione è short ma non entra in stop loss (serve perchè su np.select si 
        # può esprimere un solo default, che sarà quando la long non entra in stop loss e verrà 
        # applicata semplicemente la penalty)
        long_no_stop_loss = (df['low'].shift(shift) - df['open'].shift(shift) > -stop_loss_point)  & (df['decision'] == Market.long_value)
        short_no_stop_loss = (df['high'].shift(shift) - df['open'].shift(shift) < stop_loss_point) & (df['decision'] == Market.short_value)

        # metto assieme le condizioni
        ###conditions = [ long_condition, short_condition_1, short_condition_2 ]
        conditions = [ long_with_stop_loss, long_no_stop_loss, short_with_stop_loss, short_no_stop_loss]
        ##conditions = [ long_condition, short_condition_1]

        # Quando faccio long il deltà sarà -(20 + penalty)
        # Quando faccio short il deltà sarà (20 + penalty)
        # Ultima condizione aggiungo la penalty al delta senza stop loss
        ###choices = [(-stop_loss_point - penalty_points), (stop_loss_point + penalty_points), (df[delta_to_use] + penalty_points)]
        
        #choices = [(-stop_loss_point - penalty_points), (df[delta_to_use] - penalty_points), (stop_loss_point + penalty_points), (df[delta_to_use] + penalty_points)]
        choices = [(-stop_loss_point - penalty_points), (df[delta_to_use] - penalty_points), (stop_loss_point + penalty_points), (df[delta_to_use] + penalty_points)]
        
        ##choices = [(stop_loss_point - penalty_points), (stop_loss_point + penalty_points)]

        # Il default è la long senza stop loss
        df['delta_penalty_stop_loss'] = np.select(conditions, choices)

        #print(df.loc[df['date_time'] == "2020-04-06"])
        #input()
        return df

    '''
    ' Aggiungo la penalità al delta della serie storica
    ' Se è una long la penalty verrà sottratta
    ' se è una short la penalty verrà aggiunta
    '''
    @staticmethod
    def get_delta_penalty(df, penalty=0, multiplier=50, delta_to_use='delta_current_day'):
        # Calcolo di quanti punti dev'essere la penalty. 
        # Se la penalty è 25$ ed il mercato è sp500: 25 / 50 = 0.5
        if penalty > 0: 
            penalty_points = penalty / multiplier
        else:
            penalty_points = 0

        conditions = [ df['decision'] == Market.long_value, df['decision'] == Market.short_value]
        choices = [(df[delta_to_use] - penalty_points),  (df[delta_to_use] + penalty_points)]

        df['delta_penalty'] = np.select(conditions, choices)

        return df