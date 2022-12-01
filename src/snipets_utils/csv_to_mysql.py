import time
import pandas as pd
from sqlalchemy import create_engine

######################################


db_host = ' 192.167.149.145'
db_name = 'datasets'
db_user = 'root'
db_pass = '7911qgtr5491'
db_port = '3306'
######################################

#symbols = ['AAPL', 'ABT', 'AMZN', 'JNJ', 'JPM', 'KO', 'MMM', 'MSFT']
#symbols = ['KO', 'MMM', 'MSFT']
#symbols = ['A', 'AAL', 'AAP', 'ABBV', 'ABC', 'ACN']
#symbols = ['ADBE', 'ADI', 'ADM', 'ADP', 'ADS', 'ADSK', 'AEE', 'AEP', 'AES', 'AFL', 'AGN', 'AIG', 'AIV']

#symbols = ['gold_cet_pit']
symbols = ['vix_cet']
#symbols = ['skew_cet']

#symbols = ['sp500_cet']
#symbols = ['dax_cet']
for symbol in symbols:

    dataset_name = symbol.lower()

    print('Leggo il dataset: ' + dataset_name)
    df = pd.read_csv('C:/Users/Andrea/Documents/GitHub/PhD-Market-Nets/datasets/nuovi_raff/' + symbol + '.csv')
    #df = pd.read_csv('C:/Users/Utente/Desktop/dataset_raff/' + 'dax_cet_original' + '.csv')
    
    # Rinomino le colonne
    print('Rinomino le colonne in minuscolo...')
    #df = df.rename(columns={"Date": "date", "Time": "time", "Open": "open", "Close": "close", "High": "high", "Low":"low", "Up": "up", "Down": "down", "Volume": "volume"})
    
    #skew
    #df = df.rename(columns={"Date": "date", "Time": "time", "Open": "open", "Close": "close", "High": "high", "Low":"low", "Vol": "vol", "OI": "oi"})
    
    #vix
    df = df.rename(columns={"Date": "date", "Time": "time", "Open": "open", "Close": "close", "High": "high", "Low":"low", "Up": "up", "Down": "down",})

    
    #df = df.rename(columns={"Date": "date", "Time": "time", "Open": "open", "Close": "close", "High": "high", "Low":"low", "Up": "up", "Down": "down"})
    #df['volume'] = df['up'] + df['down']

    print('Converto il campo date nel formato Y-m-d...')
    df['date'] = pd.to_datetime(df['date']).dt.date

    #DAX FIXS
    #df.loc[df['volume'] > 1000000, ['up', 'down','volume']] = 0

    df['id'] = range(1, df.shape[0] + 1)
    df.to_csv('C:/Users/Andrea/Desktop/csv ok/' + dataset_name + '.csv', header=True, index=False, date_format=str)
    #input("Salvato")
    print("Salvato")
    '''
    print("Inserisco all'interno del DB le righe...")
    start = time.time()

    # Creo la connessione al DB
    engine = create_engine('mysql+mysqlconnector://' + db_user + ':' + db_pass + '@' + db_host + ':' + db_port + '/' + db_name, echo=False)

    df.to_sql(name=dataset_name, con=engine, if_exists='append', index=False, chunksize=10000000, method='multi')

    end = time.time()
    print("Inserimento completato in : " + str(end - start) + " secondi")
    '''