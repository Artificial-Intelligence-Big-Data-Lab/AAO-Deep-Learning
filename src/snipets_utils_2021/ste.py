import pandas as pd 
import json 

df = pd.read_csv('C:/Users/Utente/Documents/Downloads/Telegram Desktop/res2011-2015.csv')
#df['date_2'] = df['date'].tolist()
#df = df.groupby('date').count()
#print(df['date_2'].mean())

print(json.loads(df['sent'].iloc[0]))