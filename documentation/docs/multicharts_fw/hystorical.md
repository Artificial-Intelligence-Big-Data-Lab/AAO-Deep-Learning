# Historical()
  Questa classe serve a memorizzare in stutture dati, come Series o DataFrame della libreria pandas, il dataset proveniente da un
  file.txt che contiene lo storico del mercato. Questo file è un csv composto da la data, il valore di apertura, il valore più alto del mercato,
  quello più basso, il valore di chiusura ed il volume, dato un periodo di tempo il numero di righe può cambiare a seconda del time scale.
  Oltre alla memorizzazione, i dati vengono gestiti per avere nuove Series con informzaioni nuove e  utili durante la fase di Trading.

---

## \_\_init\_\_()
Nel costruttore vengono salvate tutte le informazioni che arrivano dalla funzione `load_file_pandas()` in variabili che conterrano DataFrame o Series.

Parametri :

  - file_name: è il nome del file csv contenente lo storico

  - point_value: rappresenta il valore del singolo contratto

Le seguenti variabili, settate dentro il costruttore, sono tutte pandas.Series :

  - date : una data e ora(nel caso sia intraday) per ogni barra del file csv, cioè a seconda del time scale

  - open : il valore di apertura per ogni barra

  - high : la valutazione più alta assunta dal titolo per ogni barra

  - low : la valutazione minima assunta dal titolo per ogni barra

  - close : il valore di chiusura per ogni barra

  - volume : rappresenta la quantità che viene comprata e venduta per ogni barra

  - avg_price: prezzo medio tra open, high, low, close e volume per ogni barra

  - range : per ogni barra è il valore in cui è oscillata la valutazione

  - body : rappresenta l'intervallo tra l'apertura e la chiusura del mercato

  - med_price : prezzo medio che ha assunto titolo

  - med_body_price : prezzo medio che ha assunto il titolo tra l'apertura e la chiusura

  - body_range_percentage : rappresenta in percentuale la variazione tra open e close rispetto al range

  - session : definsce la sessione del mercato

  - high_s : assegnata dalla funzione `daily_high_s(1)`

  - low_s : assegnata dalla funzione `daily_low_s(1)`

  - delta_close_percentage : è il valore in percentuale della differenza tra la chiusura e la chiusura successiva diviso la successiva

  - close_open_percentage : è il valore in percentuale della differenza tra la chiusura e l'apertura diviso l'apertura

Variabili non create dalla funzione che carica il file csv :

  - file_name : nome del file

  - type: definita dalla funzione `define_in_tra_multi_day()`

  - point_value : valore del contratto

  - year : per ogni barra tiene l'anno

  - month : per ogni barra tiene il mese

  - day : per ogni barra tiene il giorno

  - hour : per ogni barra tiene l' ora

  - minute : per ogni barra tiene il minuto

---

## define_in_tra_multi_day()

Se nel nome del file è presente la scritta `"Daily"` sarà ritornato `"multiday"` in caso contrario `"intraday"`,
questo valore verrà associato alla variabile type e servirà per determinare come estrarre i dati dal file.txt

---

## load_file_pandas()

Dal file.txt contenente le serie storiche vengono create le series che poi verrano assegnate alle varibili dentro il costruttore. Oltre a quelle presenti nel file csv, ne verranno create altre. A seconda del tipo di serie storiche vengono prese azioni diverse:

  - in min il volume è diviso in up e down quindi prima devono essere sommati per ottenere il valore unico che rappresenti il volume.

  - per min viene creata la series Session, per detreminare la sessione di mercato.

---

## Funzioni di calcolo sui dati

Le successive funzioni servono a svolgere alcuni calcoli di raggruppamento dei dati utili per l'analisi su scala più larga, per esempio raggruppamenti per giorni, settimane, mesi, anni ecc. Ognuna di esse restituisce una Pandas series e prendono in ingresso un numero di offset. Nel caso sia intraday sono importanti le funzioni che tengono conto della sessione, ovvero tutte quelle che terminano con \_s. Gli spazi vuoti della serie diventano nan e la lunghezza della struttura dati rimane la stessa.

  - Funzioni per raggruppare in giorni che tiene conto della sessione:

    - **daily_open_s()**
      Resituisce una Series con le aperture per ogni giorno

    - **daily_high_s()**:
      Resituisce una Series con i valori massimi raggiunti per ogni giorno

    - **daily_low_s()**:
      Resituisce una Series con i valori minimi raggiunti per ogni giorno

    - **daily_close_s()**:
      Resituisce una Series con i valori di chiusura per ogni giorno

    - **daily_range_s()**:
      Restituisce l'intervallo di valore in cui è oscillata la valutazione del titolo per ogni giorno

  - Funzinoni per raggruppare in giorni:

    - **daily_body()**:
      Restituisce il valore dell'intervallo tra l'apertura e la chiusura della giornata per ogni giorno

    - **daily_body_percent()**:
      Uguale a `daily_body()`, solo che viene calcolato in percentuale

    - **daily_percent()**:
      Restituisce non lo so e non lo capisco

    - **daily_range()**:
      Restituisce l'intervallo di valore in cui è oscillata la valutazione del titolo per ogni giorno

    - **daily_body_range_percent()**:
      Restituisce la divisione tra `daily_body()` per `daily_range()` per ogni giorno

    - **daily_open()**:
      Resituisce una Series con le aperture per ogni giorno

    - **daily_high()**:
      Resituisce una Series con i valori massimi raggiunti per ogni giorno

    - **daily_low()**:
      Resituisce una Series con i valori minimi raggiunti per ogni giorno

    - **daily_close()**:
      Resituisce una Series con i valori di chiusura per ogni giorno


  - Funzioni per raggruppare in settimane :

    - **weekly_open()**:
      Resituisce una Series con le aperture per ogni settimana

    - **weekly_high()**:
      Resituisce una Series con i valori massimi raggiunti per ogni settimana

    - **weekly_low()**:
      Resituisce una Series con i valori minimi raggiunti per ogni settimana

    - **weekly_close()**:
      Resituisce una Series con i valori di chiusura per ogni settimana

  - Funzioni per raggruppare in mesi:

    - **monthly_open()**:
      Resituisce una Series con le aperture per ogni mese

    - **monthly_high()**:
      Resituisce una Series con i valori massimi raggiunti per ogni mese

    - **monthly_low()**:
      Resituisce una Series con i valori minimi raggiunti per ogni mese

    - **monthly_close()**:
      Resituisce una Series con i valori di chiusura per ogni mese

  - Queste funzioni in base al parametro `period` restituiscono una serie con determinate caratteristiche :

      - **hhd()**: Data la Series high, raccoglie tutte le righe per giorno e cerca il massimo. Trasla il vettore di uno ed elimina i valori nulli,
              poi per ogni periodo si cerca il massimo.

      - **lhd()**: Data la Series high, raccoglie tutte le righe per giorno e cerca il massimo. Trasla il vettore di uno ed elimina i valori nulli,
              poi per ogni periodo si cerca il minimo.

      - **lld()**: Data la Series low, raccoglie tutte le righe per giorno e cerca il minimo. Trasla il vettore di uno ed elimina i valori nulli,
              poi per ogni periodo si cerca il minimo.

      - **hld()**: Data la Series low, raccoglie tutte le righe per giorno e cerca il minimo. Trasla il vettore di uno ed elimina i valori nulli,
              poi per ogni periodo si cerca il massimo.

      - **hhd_s()** : uguale ad `hhd()`, però tiene conto della sessione

      - **lhd_s()** : uguale ad `lhd()`, però tiene conto della sessione

      - **lld_s()** : uguale ad `lld()`, però tiene conto della sessione

      - **hld_s()** : uguale ad `hld()`, però tiene conto della sessione

___
