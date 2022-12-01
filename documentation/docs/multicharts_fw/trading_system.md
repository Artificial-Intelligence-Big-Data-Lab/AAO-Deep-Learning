# TradingSystem()

La classe TradingSystem si occupa di gestire le trades, quindi di entrare e uscire nel/dal mercato in base a delle condizioni. Inoltre si occupa di calcolare profitti e perdite delle trades e tutti i relativi dati che possono essere utili, come il drawdown, il net profit e così via. Effettua anche l'export dei dati in un file csv e si occupa di disegnare e stampare i vari grafici, come per esempio il grafico del drawdown, dell'equity curve ecc.

# \_\_init\_\_()

Costruttore della classe

**Parametri**:

- **name**: string
- **enter_rules_long**: Pandas.Series
- **exit_rules_long**: Pandas.Series
- **enter_level_long**: Pandas.Series
- **enter_rules_short**: Pandas.Series
- **exit_rules_short**: Pandas.Series
- **enter_level_short**: Pandas.Series

Nel costruttore verranno inizializzate le seguenti variabili:

- **name**: il nome del trading system
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

# operations_number()

Ritorna il numero di operazioni effettuate.

# operations_number_long()

Ritorna il numero di operazioni long effettuate.

# operations_number_short()

Ritorna il numero di operazioni short effettuate.

# largest_losing_trade()

Ritorna la perdita più grande.

# largest_losing_trade_date()

Ritorna la data della perdita più grande.

# largest_winning_trade()

Ritorna la vincita più grande.

# largest_winning_trade_date()

Ritorna la data della vincita più grande.

# percent_win()

Ritorna la percentuale di vittorie, calcolato tramite il rapporto tra il numero delle trade vincenti e il numero di operazioni totali moltiplicate per 100.

# percent_win_long()

Ritorna la percentuale di vittorie long, calcolato tramite il rapporto tra il numero delle trade long vincenti e il numero di operazioni totali moltiplicate per 100.

# percent_win_short()

Ritorna la percentuale di vittorie short, calcolato tramite il rapporto tra il numero delle trade short vincenti e il numero di operazioni totali moltiplicate per 100.

# percent_loss()

Ritorna la percentuale di peridte, calcolato tramite il rapporto tra il numero delle trade perdenti e il numero di operazioni totali moltiplicate per 100.

# avg_winning_trade()

Ritorna il risultato del rapporto tra il gross profit e il numero di trade vincenti. Rappresenta in media il guadagno che si ha quando si vince una trade.

# avg_winning_trade_long()

Ritorna il risultato del rapporto tra il gross profit long e il numero di trade long vincenti. Rappresenta in media il guadagno che si ha quando si vince una trade long.

# avg_winning_trade_short()

Ritorna il risultato del rapporto tra il gross profit short e il numero di trade short vincenti. Rappresenta in media il guadagno che si ha quando si vince una trade short.

# avg_losing_trade()

Ritorna il risultato del rapporto tra il gross loss e il numero di trade perdenti. Rappresenta in media la perdita che si ha quando si perde una trade.

# avg_losing_trade_long()

Ritorna il risultato del rapporto tra il gross loss long e il numero di trade long perdenti. Rappresenta in media la perdita che si ha quando si perde una trade long.

# avg_losing_trade_short()

Ritorna il risultato del rapporto tra il gross loss short e il numero di trade short perdenti. Rappresenta in media la perdita che si ha quando si perde una trade short.

# reward_risk_ratio()

Ritorna il valore assoluto del rapporto tra la media delle trade vincenti e la media delle trade perdenti.


# reward_risk_ratio_long()

Ritorna il valore assoluto del rapporto tra la media delle trade vincenti long e la media delle trade perdenti long.

# reward_risk_ratio_short()

Ritorna il valore assoluto del rapporto tra la media delle trade vincenti short e la media delle trade perdenti short.

# gross_profit()

Ritorna la somma di tutte le trade che guadagnano.

# gross_profit_long()

Ritorna la somma di tutte le trade long che guadagnano.

# gross_profit_short()

Ritorna la somma di tutte le trade short che guadagnano.

# gross_loss()

Ritorna la somma di tutte le trade che perdono.

# gross_loss_long()

Ritorna la somma di tutte le trade long che perdono.

# gross_loss_short()

Ritorna la somma di tutte le trade short che perdono.

# profit_factor()

Ritorna il profit factor ossia il rapporto tra gross profit e gross loss.

# profit_factor_long()

Ritorna il risultato del rapporto tra gross profit long e gross loss long.

# profit_factor_short()

Ritorna il risultato del rapporto tra gross profit short e gross loss short.

# net_profit()

Ritorna la somma di tutte le trades, sia vincenti che perdenti.

# net_profit_long()

Ritorna la somma di tutte le trades long, sia vincenti che perdenti.

# net_profit_short()

Ritorna la somma di tutte le trades short, sia vincenti che perdenti.

# avg_trade()

Ritorna la somma dei soldi guadagnati o persi dal trade medio. Calcolato dividendo il net profit per il numero di trades.

# avg_trade_long()

Ritorna la somma dei soldi guadagnati o persi dal trade long medio. Calcolato dividendo il net profit long per il numero di trades long.

# avg_trade_short()

Ritorna la somma dei soldi guadagnati o persi dal trade short medio. Calcolato dividendo il net profit short per il numero di trades short.

# drawdown()

Funzione che calcola il drawdown. Il drawdown è la distanza tra l'ultimo picco di equity e il punto più basso registrato.

# max_drawdown()

Ritorna il massimo drawdown. Ossia tra tutte le perdite relative al drawdown ritorna la maggiore.

# max_drawdown_date()

Ritorna la data in cui si è verificato il massimo drawdown

# avg_drawdown()

Ritorna il drawdown medio

# drawdown_statistics()

Ritorna il drawdown in percentili

# apply()
    
Funzione che esegue tutti i calcoli. Il parametro `data` contiene e raccoglie i dati del dataset. `order_type` rappresenta se stiamo effettuando operazioni di tipo **stop**, **market** o **limit**. `instrument` indica su cosa stiamo investendo: 1 - Azioni, 2 - Future, 3 - Forex. Il parametro `stop_loss` indica se siamo in una trade e, se stiamo perdendo più di questo valore allora esce dal mercato. `costs` è il costo per ogni operazione. `contracts` è il numero di contratti che verranno scambiati ad ogni trade. `money` rappresenta la quantità di denaro da investire. `time_exit` è il valore che se il timer della time exit raggiunge fa in modo che venga chiusa la trade automaticamente

**Parametri**

- **data**: Hystorical
- **order_type**: string['stop', 'market', 'limit']
- **instrument**: int[1,2,3]
- **stop_loss**: int
- **costs**: int
- **contracts**: int 
- **money**: int
- **time_exit**: int
---
La funzione all'inizio crea delle Pandas series di zeri di lunghezza pari al data frame `enter_rules_long` (ossia il dataframe che indica quando bisogna entrare sul mercato).

- **equity**: rappresenta i valori della equity line
- **market_position**: indica se siamo entrati nel mercato long (quindi avrà valore 1), se siamo entrati short (valore -1), oppure se non siamo nel mercato (valore 0)
- **number_of_contracts**: numero di contratti che verranno scambiati quando faremo una trade
- **stop_loss_active**: indica se è stata attivata la stop loss
- **operations**: tutte le operazioni che vengono fatte
- **operations_long**: tutte le operazioni long che vengono fatte
- **operations_short**: tutte le operazioni short che vengono fatte
- **time_exit_counter**: nel momento in cui entriamo nel mercato inzia a contare per quante barre restiamo dentro. Se si supera il numero allora si esce
- **time_exit_active**: indica se è stata attivata la time exit

Successivamente i valori delle Pandas series `operations`, `operations_long`, `operations_short` vengono posti a NaN ossia not a number.

Ci sono anche altri parametri: 

- **point_value**: int
- **events_in**: Pandas.Series
- **events_out**: Pandas.Series
- **last_equity**: int

`point_value` indica il costo di un movimento, `events_in` è una Pandas series che contiene tutti gli ingressi nel mercato, `events_out` è una Pandas series che contiente tutte le uscite dal mercato e infine `last_equity` che viene usata nel calcolo della equity line.

---
La parte successiva del codice viene eseguita per ogni barra.
Abbiamo diversi eventi che possono esserci:

**Caso in cui dobbiamo entrare nel mercato**:

Se non siamo già nel mercato e una tra `enter_rules_long` e `enter_rules_short` è vera.
Viene controllato l'`instrument` per determinare il `number_of_contracts`, e successivamente l'`order_type` che assieme ad altre condizioni ci indica se dobbiamo, e in tal caso come, entrare sul mercato.
Se entriamo nel mercato allora aggiorniamo le operazioni, gli `events_in`, la `market_position` e la `last_equity`.

**Quando siamo nel mercato**:

Nel momento in cui siamo sul mercato nelle barre in cui non effettueremo operazioni l'`entry_price`, e `market_position` verranno propagate. Invece l'`equity` andrà aggiornata ogni passo.

**Caso in cui dobbiamo uscire dal mercato**:

Se siamo nel mercato e una tra `exit_rules_long` e `exit_rules_short` è vera e `stop_loss_exit` e `time_exit` sono false.
Vengono scritti gli eventi in `events_out`, aggiornate le operazioni, aggiornata la `market_position`, calcolata l'`equity` e aggiunte eventuali operazioni short o long a `operations_short` o `operations_long`.

**Caso di uscita per stop loss**:

In una prima parte del codice si controlla se siamo nel mercato e se si è verificata la condizione per cui bisogna uscire per stop loss (ossia se stiamo perdendo di più della stop loss). In caso si è verificata vengono scritti gli eventi in `events_out`, aggiornate le operazioni, aggiornata la `market_position`, calcolata l'`equity` e settato il valore di `stop_loss_active` a 1.

Nella seconda parte di codice se non ci sono exit rules vere e `stop_loss_active` è pari a 1, calcola le operazioni e in base alla market position aggiunge eventuali operazioni long o short.

**Caso di uscita per time exit**:

Se siamo nel mercato e si è verificata la condizione per cui bisogna uscire per time exit (ossia se `time_exit_counter` è maggiore della `time_exit`), quindi quando `time_exit_active` è pari a 1.  Vengono scritti gli eventi in `events_out`, aggiornate le operazioni, aggiornata la `market_position`, calcolata l'`equity`.

In una prima parte del codice si controlla se siamo nel mercato e se si è verificata la condizione per cui bisogna uscire per time exit (ossia se `time_exit_counter` è maggiore della `time_exit`). In caso si è verificata vengono scritti gli eventi in `events_out`, aggiornate le operazioni, aggiornata la `market_position`, calcolata l'`equity` e settato il valore di `time_exit_active` a 1.

Nella seconda parte di codice se non ci sono exit rules vere e `time_exit_active` è pari a 1, calcola le operazioni e in base alla market position aggiunge eventuali operazioni long o short.

**Counter per Time Exit**

Nel momento in cui entriamo nel mercato avviamo il contatore della time exit (`time_exit_counter`). Quando usciamo questo parametro viene settato a 0.

Alla fine della funzione vengono aggiornati i parametri inziali con i valori calcolati. E viene fatto l'export

# annual_stats()

Ritorna le statistiche annuali.

# monthly_stats()

Ritorna le statistiche mensili.

# plot_monthly_histogram()

Funzione che disegna e stampa l'istogramma mensile.

# plot_equity()

 Funzione che disegna e stampa il grafico della equity.

# plot_drawdown()

Funzione che disegna e stampa il grafico del drawdown.

# plot_annual_histogram()

Funzione che disegna e stampa l'istogramma annuale.

# plot_monthly_bias_histogram()

Funzione che disegna l'istogramma mensile 

# performance_report()

**Parametri**:

- **costs**: Costo di ogni operazione

Funzione che stampa tutti i dati relativi alla strategia.

# log_export()

**Parametri:**

- **data**: Hystorical

Il parametro data è di tipo storico e contiene tutte le informazioni ottenute dal dataset.
Le colonne da `Date_time` a `DailyHigh[1]` sono tutte prese da questo oggetto della classe Hystorical.
`DailyHigh[1]` e `DailyLow[1]` sono i massimi e i minimi del giorno precedente.
Crea un file csv che conterrà le seguenti colonne:

- Date_time
- Open
- High
- Low
- Close
- Volume
- DailyLow[1]
- DailyHigh[1]
- EnterRulesLong
- ExitRulesLong
- EnterRulesShort
- ExitRulesShort
- Operations
- OperationsLong
- OperationsShort
- Equity
- MarketPosition
- NumberOfContracts
- EntryPrice
- StopLossActive
- Events_In
- Events_Out
- TimeExitCounter
- TimeExitActive