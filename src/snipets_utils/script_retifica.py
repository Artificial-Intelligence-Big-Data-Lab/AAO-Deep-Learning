import time
import numpy as np 
import pandas as pd 
from os import listdir
from datetime import date
import mysql.connector as sql
from os.path import isfile, join
from yahoo_historical import Fetcher
from sqlalchemy import create_engine

'''
'
'''
def create_table_query(table_name):
        table_name = table_name.lower()

        query = """CREATE TABLE `""" + table_name + """`  (
                `id` int(11) NOT NULL AUTO_INCREMENT,
                `date` date NOT NULL,
                `time` time(0) NOT NULL,
                `open` float NOT NULL,
                `close` float NOT NULL,
                `close_adj` float NOT NULL,
                `high` float NOT NULL,
                `low` float NOT NULL,
                `up` int(11) NOT NULL,
                `down` int(11) NOT NULL,
                `volume` float NOT NULL,
                PRIMARY KEY (`id`) USING BTREE
                ) ENGINE = InnoDB AUTO_INCREMENT = 1 CHARACTER SET = utf8 COLLATE = utf8_general_ci ROW_FORMAT = Compact;
                """
        return query

'''
'
'''
def get_market_from_yahoo(symbol): 
        today = date.today()

        day = int(today.strftime("%d"))
        month = int(today.strftime("%m"))
        year = int(today.strftime("%Y"))

        data = Fetcher(symbol, [1998,1,1], [year, month, day])

        return data.getHistorical()

'''
'
'''
def apply_correction(df, start_date, delta): 
    mask = (df['Date'] <= start_date)
    
    df_founded = df[mask]

    df.loc[mask, 'Adj Close'] = df_founded['Close'] / delta

'''
'
'''
def get_symbols_list(dataset_path):
        symbols = []
        files = [f for f in listdir(dataset_path) if isfile(join(dataset_path, f))]

        for file in files: 
                symbols.append(file.split('_')[0])

        return symbols

################
# PARAMETERS #
################

db_host = '192.167.149.145'
db_user = 'root'
db_pass = '7911qgtr5491'
db_port = '3306'
db_name = 'datasets'

dataset_path = 'C:/Users/andre/Desktop/TOP 30 SP500/'

symbols = get_symbols_list(dataset_path=dataset_path)
#clssymbols = ['AIZ']

################
# START SCRIPT #
################

datasets_conn = sql.connect(host=db_host, database=db_name, user=db_user, password=db_pass)

engine = create_engine('mysql+mysqlconnector://' + db_user + ':' + db_pass + '@' + db_host + ':' + db_port + '/' + db_name, echo=False)

for symbol in symbols:
        
        print('Inizio ad elaborare il seguente simbolo: ' + symbol)

        # Create the new database
        cur = datasets_conn.cursor()
        query = create_table_query(symbol)
        cur.execute(query)

        dataset_5_min_path = dataset_path + symbol + '_5MIN.txt'

        # dataset giornaliero su cui cerco le divisioni di utili
        dataset_daily = get_market_from_yahoo(symbol=symbol)

        # dataset che dovrò retificare
        dataset_5_min = pd.read_csv(dataset_5_min_path)

        # rimuovo la colonna OI
        if 'OI' in dataset_5_min.columns:
                dataset_5_min = dataset_5_min.drop(['OI'], axis=1)

        print("Converto le date in pd.datetime...")
        dataset_daily['Date'] = pd.to_datetime(dataset_daily['Date']).dt.date
        dataset_5_min['Date'] = pd.to_datetime(dataset_5_min['Date']).dt.date

        dataset_5_min['Adj Close'] = dataset_5_min['Close']

        # ultimo delta trovato. 
        last_delta = 1.0

        # il nuovo delta trovato. Sarà maggiore rispetto al global ogni volta che ci sarà stato uno split di utili
        new_delta = 1.0

        print("Inizio la retifica...")
        for index, row in dataset_daily[::-1].iterrows():
                new_delta = dataset_daily.at[index, 'Close'] / dataset_daily.at[index, 'Adj Close']

                if new_delta > last_delta:
                        apply_correction(df=dataset_5_min, start_date=dataset_daily.at[index, 'Date'], delta=new_delta)
                        last_delta = new_delta # aggiorno l'ultimo delta trovato
        
        #print("Salvo il file...")
        #dataset_5_min.to_csv('C:/Users/andre/Documents/GitHub/PhD-Market-Nets/datasets/single_company/5min_adjusted/' + symbol + '_5MIN_ADJUSTED.txt', header=True, index=False)
        
        # Rinomino le colonne
        print('Rinomino le colonne in minuscolo...')

        # Rinomino le colonne a seconda di quale dataset stia arrivando
        if 'Vol' in dataset_5_min.columns:
                dataset_5_min = dataset_5_min.rename(columns={"Date": "date", "Time": "time", "Open": "open", "Close": "close", "High": "high", "Low":"low", "Up": "up", "Down": "down", "Adj Close": "close_adj", "Vol": "volume"})

        if 'Up' in dataset_5_min.columns and 'Down' in dataset_5_min.columns:
                dataset_5_min = dataset_5_min.rename(columns={"Date": "date", "Time": "time", "Open": "open", "Close": "close", "High": "high", "Low":"low", "Up": "up", "Down": "down", "Adj Close": "close_adj"})


        if 'up' in dataset_5_min.columns and 'down' in dataset_5_min.columns:
                dataset_5_min['volume'] = dataset_5_min['up'] + dataset_5_min['down']
        else:
                dataset_5_min['volume'] = -1
                dataset_5_min['up'] = -1
                dataset_5_min['down'] = -1

        print('Converto il campo date nel formato Y-m-d...')
        dataset_5_min['date'] = pd.to_datetime(dataset_5_min['date']).dt.date

        #dataset_5_min.to_csv('../datasets/' + dataset_name + '.csv', header=True, index=False, date_format=str)

        print("Inserisco all'interno del DB le righe...")
        start = time.time()

        # Creo la connessione al DB
        engine = create_engine('mysql+mysqlconnector://' + db_user + ':' + db_pass + '@' + db_host + ':' + db_port + '/' + db_name, echo=False)

        symbol = symbol.lower()

        dataset_5_min.to_sql(name=symbol, con=engine, if_exists='append', index=False, method='multi', chunksize=100000)

        end = time.time()
        print("Inserimento completato in : " + str(end - start) + " secondi")

datasets_conn.close()
