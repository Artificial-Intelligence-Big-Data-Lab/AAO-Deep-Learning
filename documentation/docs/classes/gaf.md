# Gaf()

Questa classe serve per poter trasformare un dataset in immagini `.png` relative alle trasformate `GADF` e `GASF`. 
Per maggiori informazioni: [https://pyts.readthedocs.io/en/latest/auto_examples/image/plot_gaf.html#sphx-glr-auto-examples-image-plot-gaf-py](https://pyts.readthedocs.io/en/latest/auto_examples/image/plot_gaf.html#sphx-glr-auto-examples-image-plot-gaf-py)

# run()
Lancia la generazione delle immagini.

**Parametri**:

- **df**: _DataFrame_
- **dataset_name**: _string_ 
- **subfolder**: _string_ 
- **type**: _string_ ['gadf', 'gasf']
- **size**: _int_
- **columns**: _list_ ['delta', 'high', 'volume', ...] 


**Esempio**: 

L'oggetto viene inizializzato passando il dataset che si vuole trasformare tramite le GAF con il parametro `df`.
I parametri `dataset_name` e `subfolder` servono per generare in automatico il path dove verranno salvate le immagini. L'idea di base è di avere la seguente struttura per le immagini: `./images/dataset_name/subfolder/segnale/data.png`.
Quindi in `subfolder` si consiglia di passare la risoluzione utilizzata per il dataset. 

Viene generato un dataset sp500 a risoluzione giornaliera. Successivamente viene eseguito il metodo `run()` per generare le immagini. Con il parametro `size` si specifica la volontà di volere immagini 40x40, prendendo quindi (in questo caso) i 40 giorni precedenti. Il parametro `columns` è opzionale, specifica quali colonne del DF trasformare in immagine. Se non specificato vengono utilizzate tutte quante ad eccezione di `date` e `date_time`.


```python

from classes.Market import Market
from classes.Gaf import Gaf

sp500 = Market(dataset='sp500')

grouped_1_day = sp500.group(freq='1d', nan=False)

gaf = Gaf()

gaf.run(df=grouped_1_day, dataset_name="sp500", subfolder="8hours", type='gadf', size=40, columns=['delta', 'volume', 'close'])
gaf.run(df=grouped_1_day, dataset_name="sp500", subfolder="8hours", type='gasf', size=40, columns=['delta', 'volume', 'close'])

```