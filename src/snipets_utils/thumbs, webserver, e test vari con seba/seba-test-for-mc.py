import numpy as np 
import pandas as pd 

df = pd.read_csv('C:/Users/Utente/Documents/GitHub/PhD-Market-Nets/experiments/026 - BIG - test lungo/predictions/predictions_during_training/validation/walk_0/net_6.csv')

df = df[['date_time', 'epoch_615']]

df['date_time'] = df['date_time'].shift(-1)
df = df.dropna()

df.to_csv('C:/Users/Utente/Desktop/test--seba/predizioni-strambe.csv', header=True, index=False)
