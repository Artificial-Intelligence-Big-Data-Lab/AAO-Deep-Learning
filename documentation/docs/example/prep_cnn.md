# Preprocessing di un Dataset per la CNN

In questo esempio si vedranno tutti i passaggi da effettuare per ottenere un dataset, effettuarne il preprocessing e preparare le immagini da dare alla `CNN`.

Partendo dal dataset `sp500` si vuole generare 4 GADF differenti di dimensione `20x20`: raggrupate per 1 ora, 1 giorno, 1 week e 1 mese. 

Successivamente si vogliono raggruppare queste 4 immagini in un unica immagine `40x40` da dare in pasto alla CNN. 

```python

from classes import Market
from classes import Gaf
from classes import Mood
from classes import ImagesMerger

import time
import pandas as pd
import mysql.connector
from sqlalchemy import create_engine


# Legge il DF sp500. Se non c'è in locale verrà scaricato dal DB remoto
sp500 = Market.Market(dataset='sp500')

# Raggruppo il dataset per 1h, 1d, 1w, 1m
one_h = sp500.group(freq='1h', nan=False)
one_d = sp500.group(freq='1d', nan=False)
one_w = sp500.group(freq='1w', nan=False)
one_m = sp500.group(freq='1m', nan=False)

# Genero solo le gadf per i 4 dataset
gaf = Gaf.Gaf()
gaf.run(df=one_h, dataset_name="sp500", subfolder="1hour", type='gadf', size=20)
gaf.run(df=one_d, dataset_name="sp500", subfolder="1day", type='gadf', size=20)
gaf.run(df=one_w, dataset_name="sp500", subfolder="1week", type='gadf', size=20)
gaf.run(df=one_m, dataset_name="sp500", subfolder="1month", type='gadf', size=20)

# Unisco  le immagini in una sola immagine 40x40
imgmerger = ImagesMerger.ImagesMerger()

imgmerger.run(input_folders='sp500',
              resolutions=['1hour', '1day', '1week', '1month'],
              signals=['delta', 'volume'],
              positions=[(0, 0), (20, 0), (0, 20), (20, 20)],
              type='gadf',
              img_size=[40, 40],
              output_path="final_merge")

```