# TradingSystemMonodirectional()

La classe TradingSystemMonodirectional è sotto classe di `TradingSystem`. Effettua le sue stesse operazioni tranne per il fatto che può effettuare solo operazioni long oppure solo operazioni short. Per indicare che tipo di operazioni deve effettuare è presente un parametro `direction`. Se esso è pari a 1 indica che ogni operazione effettuata sarà di tipo long, se è pari a 2 allora saranno di tipo short.

# \_\_init\_\_()

Costruttore della classe. È uguale a quello di Trading system 

**Parametri**:

- **name**: il nome del trading system
- **enter_rules**: una Pandas series che per ogni riga del dataset ci dice se dobbiamo entrare nel mercato
- **exit_rules**: una Pandas series che per ogni riga del dataset ci dice se dobbiamo uscire dal mercato
- **enter_level**: una Pandas series che per ogni riga del dataset individua i prezzi su cui entrerò nel mercato
- **enter_rules_long**: una Pandas series che per ogni riga del dataset ci dice se dobbiamo entrare nel mercato long
- **exit_rules_long**: una Pandas series che per ogni riga del dataset ci dice se dobbiamo uscire dal mercato long
- **enter_level_long**: una Pandas series che per ogni riga del dataset individua i prezzi su cui entrerò nel mercato long
- **enter_rules_short**: una Pandas series che per ogni riga del dataset ci dice se dobbiamo entrare nel mercato short
- **exit_rules_short**: una Pandas series che per ogni riga del dataset ci dice se dobbiamo uscire dal mercato short
- **enter_level_short**: una Pandas series che per ogni riga del dataset individua i prezzi su cui entrerò short nel mercato
- **entry_price**: un Pandas dataframe che indica il prezzo al momento di ingresso
- **market_position**: un Pandas dataframe di interi che quando è pari a 1 indica che siamo nel mercato long, pari a 0 che non siamo nel mercato e pari a -1 che siamo nel mercato short
- **number_of_contracts**: il numero di contratti che dobbiamo acquistare o vendere quando entriamo o usciamo nel mercato
- **open_equity**: un Pandas dataframe che verrà usato per calcolare il drawdown
- **closed_equity**:
- **operations**: un Pandas dataframe che conterrà tutte le operazioni
- **operations_long**: un Pandas dataframe che conterrà tutte le operazioni long
- **operations_short**: un Pandas dataframe che conterrà tutte le operazioni short
- **events_in**: un Pandas dataframe che salva gli eventi in ingresso, quindi le entrate nel mercato
- **events_out**: un Pandas datafarame che salva gli eventi in uscita, quindi le uscite dal mercato
- **stop_loss_active**: un Pandas dataframe di interi che indicherà quando bisogna uscire dal mercato per stop loss
- **time_exit_counter**: un Pandas dataframe di interi che è pari a 0 in tutte le righe in cui non siamo nel mercato, nel momento in cui entriamo per ogni barra questo contatore verrà incrementato. Se supererà un determinato valore allora uscirà dal mercato.
- **time_exit_active**: un Pandas dataframe di interi che indica quando dobbiamo uscire dal mercato per time exit
- **direction**: intero che se è pari a 1 indica che ogni operazione effettuata sarà di tipo long, se è pari a 2 allora saranno di tipo short.

Viene crato un oggetto TradingSystem con il costruttore, passando `name`, `enter_rules_long`, `exit_rules_long`, `enter_level_long`, `enter_rules_short` ,`exit_rules_short`, `enter_level_short`. Gli altri parametri verranno assegnati alle rispettive variabili

# apply()
    
La funzione apply è la stessa di `TradingSystem`, l'unica differenza è che (per esempio per gestire l'ingresso nel mercato) anziché guardare se `enter_rules_long` o `enter_rules_short` sono vere controlla se la `enter_rules` è pari a true. Una volta che quest'ultima si è verificata controlla in base al parametro direction se deve entrare short o long. La stessa cosa viene fatta per tutte gli altri controlli, vengono esaminate innanzitutto le condizioni di ingresso, uscita, stop loss, time exit e così via. Una volta verificate anziché controllare la `market_position` verificano la variabile `direction` e in base ad essa decidono come entrare o uscire dal mercato.