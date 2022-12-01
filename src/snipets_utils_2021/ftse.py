import pandas as pd 

#df = pd.read_csv('C:/Users/Utente/Desktop/tweet_uk.csv')
df = pd.read_csv('C:/Users/Utente/Desktop/yahoo_ftse2000-2020.csv', delimiter=';')

df['Date'] = pd.to_datetime(df['Date'])

df = df.rename(columns={
    "Date": "date",
    "Open": "open",
    'High': "high",
    "Low": "low",
    "Close": "close",
    "Adj_Close": "adj_close", 
    "Volume": "volume"
})

df = df.sort_values(by="date")
df['volume'] = df['volume'].map(lambda x: x.replace(',', ''))
df['open'] = df['open'].map(lambda x: x.replace(',', ''))
df['high'] = df['high'].map(lambda x: x.replace(',', ''))
df['low'] = df['low'].map(lambda x: x.replace(',', ''))
df['close'] = df['close'].map(lambda x: x.replace(',', ''))
df['adj_close'] = df['adj_close'].map(lambda x: x.replace(',', ''))

df.to_csv('C:/Users/Utente/Desktop/new_yahoo_ftse2000-2020.csv', header=True, index=False)