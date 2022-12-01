# Market()

Questa classe serve per poter utilizzare con facilità i datasets dei mercati finanziari. 


Per poter funzionare correttamente, la classe genera una copia dell'intero dataset nella cartella `./datasets/`, 

Costruttore della classe. 

**Parametri**:

- **dataset**: _string_ [sp500, bund]

L'oggetto viene inizializzato facendo due controlli: il primo è che il dataset specificato sia valido il secondo che il dataset in questione sia presente all'interno del progetto. 
Nel caso non ci fosse, verrà effettuata una connessione al Database e verrà scaricato l'intero dataset in locale. L'operazione potrebbe prendere qualche minuto. 

**Esempio**: 

```python

from classes import Market

sp500 = Market.Market(dataset='sp500')

```
# get()

Restituisce l'intero datasets originale, tale e quale da come viene letto dal Datasets locale o tramite MySql. 

```python

from classes.Market import Market

sp500 = Market(dataset='sp500')

df = sp500.get()
```


# group()

Restituisce il dataset raggrupato per una risoluzione temporale, specificata tramite il parametro `freq` del metodo. 
Il parametro `nan` specifica se si vogliono tenere le righe contenenti valori nan. Ad esempio, se si raggruppa per 1 giorno (`1d`), alcune righe potrebbero contenere valori nan per quei giorni in cui il mercato è chiuso. 
Restituisce una copia del Datasets. Il metodo restituisce 3 tipologie si delta: `delta`, `delta_percentage`, `delta_percentage_previous_day`.

- `delta` è calcolato come `close` -  `open`
- `delta_percentage` è calcolato come `close` -  `open` / `open`
- `delta_percentage_previous_day` è calcolato come la differenza in percentuale tra il `close` del giorno corrente ed il `close` del giorno precedente

**Parametri**:

- **freq**: _string_ ['1m', '1w', '1d', '2d' ... ]
- **nan**: _boolean_

**Esempio**:

```python

from classes.Market import Market

sp500 = Market(dataset='sp500')

one_day = sp500.group(freq='1d', nan=False)

one_day_with_nan = sp500.group(freq='1d', nan=True)
```

# remove_columns()
Restituisce una copia del dataset, rimuovendo le colonne specificate come parametro. 
Se le colonne passate come parametro non sono una lista, o non sono presenti nel dataset originale verrà restituito un messaggio di errore

**Parametri**:

- **columns**: _list_ ['open', 'close', ...]

**Esempio**: 

```python

from classes.Market import Market

sp500 = Market(dataset='sp500')

# Rimuovo le colonne open, date, time, down
new_sp500 = sp500.remove_columns(columns=['open', 'date', 'time', 'down'])
```

# get_label_current_day()

Restituisce la label del dataset (ovvero i valori `-1` e `1`) per una specifica risoluzione temporale. 
La label viene calcolata come `close` - `open` sul **giorno corrente**. Se la differenza è positiva la label sarà `1`, `-1` altrimenti. Si può specificare anche quali colonne tenere nel dataset (oltre la data) e la risoluzione temporale. Se si raggruppa per ore quindi si avrà la label ora per ora

Scegliendo come frequenza `freq=1d` si otterranno le label per l'andamento del mercato giornaliero. 

Il campo `columns` serve a specificare quali altre colonne si vogliono inserire nel DF restituito dal metodo. Di default il metodo restituisce solamente il campo `date` e il campo `label`.

**Parametri**:

- **freq**: _string_ ['1m', '1w', '1d', '2d' ... ]
- **columns**: _list_ ['open', 'close', ...]

**Esempio**: 

```python

from classes.Market import Market

sp500 = Market(dataset='sp500')

# Ottengo il dataset il valore di label ed in aggiunta anche le colonne down, up, down
label_current_day = sp500.get_label_current_day(freq='1d', columns=['down', 'up', 'down'])
```

# get_label_next_day()

Restituisce la label del dataset (ovvero i valori `-1` e `1`) per una specifica risoluzione temporale. 
La label viene calcolata come `close` - `open` sul **giorno successivo**. Se la differenza è positiva la label sarà `1`, `-1` altrimenti. Si può specificare anche quali colonne tenere nel dataset (oltre la data) e la risoluzione temporale. Se si raggruppa per ore quindi si avrà la label ora per ora

Scegliendo come frequenza `freq=1d` si otterranno le label per l'andamento del mercato giornaliero. 

Il campo `columns` serve a specificare quali altre colonne si vogliono inserire nel DF restituito dal metodo. Di default il metodo restituisce solamente il campo `date` e il campo `label`.

**Parametri**:

- **freq**: _string_ ['1m', '1w', '1d', '2d' ... ]
- **columns**: _list_ ['open', 'close', ...]

**Esempio**: 

```python

from classes.Market import Market

sp500 = Market(dataset='sp500')

# Ottengo il dataset il valore di label ed in aggiunta anche le colonne down, up, down
label_current_day = sp500.get_label_next_day(freq='1d', columns=['down', 'up', 'down'])
```

# get_label_next_day_using_close()

Restituisce la label del dataset (ovvero i valori `-1` e `1`) per una specifica risoluzione temporale. 
La label viene calcolata come `close del giorno corrente` - `close del giorno sucessivo`. Se la differenza è positiva la label sarà `1`, `-1` altrimenti. Si può specificare anche quali colonne tenere nel dataset (oltre la data) e la risoluzione temporale. Se si raggruppa per ore quindi si avrà la label ora per ora

Scegliendo come frequenza `freq=1d` si otterranno le label per l'andamento del mercato giornaliero. 

Il campo `columns` serve a specificare quali altre colonne si vogliono inserire nel DF restituito dal metodo. Di default il metodo restituisce solamente il campo `date` e il campo `label`.

**Parametri**:

- **freq**: _string_ ['1m', '1w', '1d', '2d' ... ]
- **columns**: _list_ ['open', 'close', ...]

**Esempio**: 

```python

from classes.Market import Market

sp500 = Market(dataset='sp500')

# Ottengo il dataset il valore di label ed in aggiunta anche le colonne down, up, down
label_current_day = sp500.get_label_next_day_using_close(freq='1d', columns=['down', 'up', 'down'])

```

# @static get_df_by_data_range()

Restituisce una copia del dataset filtrato per data. 
Prende in ingresso data di inizio e data di fine per applicare la maschera di ricerca

**Parametri**:

- **start_date**: _string_
- **end_date**: _string_

**Esempio**: 

```python

from classes.Market import Market

sp500 = Market(dataset='sp500')
sp500_df = sp500.get()
# Ottengo un sottoinsieme del DF filtrato per data
subset_df = Market.get_df_by_data_range(df=sp500_df, start_date='2012-01-01', end_date='2018-01-01')
```