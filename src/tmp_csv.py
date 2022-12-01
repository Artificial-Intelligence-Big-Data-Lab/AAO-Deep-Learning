import time
import pandas as pd
from sqlalchemy import create_engine


df = pd.read_csv('C:/Users/Utente/Downloads/$VIX_X_5MIN_CET (1).txt')
df['volume'] = df['up'] + df['down']

df.to_csv('C:/Users/Utente/Downloads/vix_cet.csv', index=False, header=True)