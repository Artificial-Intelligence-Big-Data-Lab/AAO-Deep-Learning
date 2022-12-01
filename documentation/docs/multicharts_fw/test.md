# Utilizzare la Trading Platform scritta in python.

Nome del file: `SP500_5MIN_ExchangeTime.txt`

In questo test si userà un sistema di trading, scritto in python, sulla serie storica di SP500, in cui si agirà attraverso delle predizioni passate tramite un file csv.

---

## Variabili iniziali.

Per primo si settano le variabili che saranno passate alla classe Trading_System che sono:

  - `contracts` : numero di contratti che verranno scambiati ad ogni trade, in questo caso è settato a `1`.

  - `money`: la quantità di capitale inziale, in questo caso è settato a `100000`.

  - `costs` : costo per ogni operazione, nel nostro caso è settato a `0`.

  - `instrument` : valore che determina se è un azione, un future o un forex, nel nostro caso di sp500 è settato a `2` che indica future.

  - `stop_loss` : il valore massimo di perdita, oltre al quale si esce dal mercato per non subire ulteriori perdite, nel nostro caso è settato a `1100` perchè risulta essere quello ottimale

  - `order_type` : tipo di strategia che puo essere "market", "stop" o "limit", nel nostro caso è settato a `"market"`

---

## Istanza classe Historical e lettura delle predizioni.

In questa fase si richiama il costruttore per istanziare la classe Historical, assegnata alla variabile spx, che conterrà il dataset di SP500. Come parametri vengono passatti una stringa con il nome del dataset e un intero che rappresenta il point_value.

Con la funzione `pandas.read_csv()` prendiamo i dati del csv con le predizioni giornaliere e le mettiamo in un `pandas.DataFrame` assegnata alla variabile `prediction`.

---

## Regole di trading in base alle predizioni.

Prima di istanziare la classe `TradingSystem`, devono essere creati delle pandas.Series, le cui lunghezze sarnno uguali al numero di date presenti nel dataset e saranno considerate come indice, e conterranno le regole da svolgere durante la fase di trading.

Variabili di entrata e uscita:

  - `enter_rules_long` : i valori sono settati a `False`, da questa Series si decide quando entrare long sul mercato.

  - `exit_rules_long` : i valori sono settati a `False`, da questa Series si decide quando uscire dal mercato in strategia long.

  - `enter_rules_short` : uguale al Series long, ma decide per la strategia short.

  - `exit_rules_short` : uguale al Series long, ma decide per la strategia short.

  - `enter_level_long` : decide per quale valore di mercato si effettua l'operazione di entrata con la strategia long, in questo test con le predizioni e `order_type = "market"` essa sara uguale alla colonna di Spx, cioè `spx.open`.

  - `enter_level_short` : uguale alla long, però decide per la strategia short.

Per assegnare alle Series i valori di entrata e uscita dal mercato, viene svolto un ciclo while che andrà avanti per la lunghezza del dataset.
Quindi dentro questi cicli verranno svolte una serie di condizioni per decidere quali valori far assumere alle Series. Essi si basano sui valori della colonna 'ensemble' del DataFrame `predizioni`.
Le condizioni sono basate sugli orari di apertura e chiusura del mercato per il future sp500, che sono:

| Apertura  | Ora   | Chiusura  | Ora   |
| --------- | ----- | --------- | ----- |
| Domenica  | 17:00 | Lunedì    | 16:00 |
| Lunedì    | 17:00 | Martedì   | 16:00 |
| Martedì   | 17:00 | Mercoledì | 16:00 |
| Mercoledì | 17:00 | Giovedì   | 16:00 |
| Giovedì   | 17:00 | Venerdì   | 16:00 |

 Dal momento che le predizioni seguono un andamento giornaliero si entrerà alla mezzanotte per poi uscire alle 23 : 55, nei casi come Domenica e Venerdi si segue l'andamento del mercato.

---

## Istanza TradingSystem, svolgimento trading e stampa report.

  Ora che le pandas Series sono pronte, viene richiamato il costruttore di `TradingSystem` e si passano come parametri le pandas.Series.
  Dal istanza della classe si richiama il metodo `TradingSystem.apply()` a cui verranno passati i settaggi iniziali e l'oggetto spx. Questa funzione fara partire il sistema di trading e alla fine dei calcoli crerà un `Report.csv` con tutti i dettagli delle operazioni.
  Con il metodo `TradingSystem.performance_report()`
  Verranno stampate tutte le informazioni relative al trading svolto e mostrate alcuni grafici risultanti.
