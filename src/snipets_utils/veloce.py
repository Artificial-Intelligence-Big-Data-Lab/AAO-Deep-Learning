import pandas as pd 
from classes.Utils import df_date_merger

filename = 'training'

df = pd.read_csv('C:/Users/Utente/Documents/Downloads/Telegram Desktop/CSV Anselmo Applied Intelligence/sp500/' + filename + '_fix.csv')

print(df)
input()
df = df[['date_time', 'prediction_1', 'prediction_2', 'prediction_3', 'prediction_4', 'prediction_5', 'prediction_6', 'prediction_7', 'prediction_8', 'prediction_9', 'prediction_10']]

df = df_date_merger(df=df, thr_hold=0.3, columns=['open', 'close', 'delta_next_day', 'high', 'low', 'up', 'down', 'volume'], dataset='sp500_cet')
#df = df.drop(columns=['label_current_day'])
df = df[['date_time', 'prediction_1', 'prediction_2', 'prediction_3', 'prediction_4', 'prediction_5', 'prediction_6', 'prediction_7', 'prediction_8', 'prediction_9', 'prediction_10', 'high', 'low', 'up', 'down', 'volume', 'open', 'close', 'delta_next_day', 'label_next_day', 'label_current_day']]

print(df)
input()
#df.to_csv('C:/Users/Utente/Documents/Downloads/Telegram Desktop/CSV Anselmo Applied Intelligence/sp500/' + filename + '.csv', header=True, index=False)