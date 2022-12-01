# Importare dataset SP500 su multicharts

# Titolo
##
###
####


**dfff**
_dsds_


sdsdss

`codice inline`

```python 
var a = 1
for i 
```

- cddcd
- dfdfd
    - dcfdcd 

Nome del file: `SP500_5MIN_ExchangeTime.txt` 

Da Quote Manager: 

1. **Add symbol**:


- Data source: Universal DDE
- Nome: A piacere
- Category: Futures
- CME: Exchange
```

2. **Settings**: 

   ```
   Price scale: 1/100
   Min Mov 25
   BPV 50
   ```

3. **Sessions**: 

   ```
   Sessions Source: Use Custom Session
   Sessions Details: Exchange
   ```
   

| Open      | Time  | Close     | Time  | Session End |
| --------- | ----- | --------- | ----- | ----------- |
| Domenica  | 17:00 | Lunedì    | 16:00 | True        |
| Lunedì    | 17:00 | Martedì   | 16:00 | True        |
| Martedì   | 17:00 | Mercoledì | 16:00 | True        |
| Mercoledì | 17:00 | Giovedì   | 16:00 | True        |
| Giovedì   | 17:00 | Venerdì   | 16:00 | True        |



4. **Tasto destro sul simbolo appena creato -> Import Data -> ASCII**

5. **Selezionare file del dataset:** `SP500_5MIN_ExchangeTime.txt`

6. **Controllare che le colonne siano corrette**

7. **Premere ok**

------

### Ora per mostrare il grafico basterà fare: 

- New Workspace (CTRL + N) 
- Tasto destro -> Chart Window

```
Data Source: Universal DDE
Instrument: il vostro dataset caricato

Su settings, mettete il data range che vi interessa e la risoluzione temporanea
```

# Importare ed eseguire il segnale

### Importare il segnale

Nome del file: `Predizioni4.Strategy.CS`

Da Esplora risorse:

1. **Copiare il file**

2. **Spostarsi nella cartella "C:\ProgramData\TS Support\MultiCharts .NET64\StudyServer\Techniques\CS"**

3. **Incollare il file**

Da Power Language:

Il riquadro a destra di Power Language contiene una cartella di nome Studies

1. **Aprire la cartella**

2. **Aprire la cartella Signals**

3. **Cercare il file "Predizioni4"**

4. **Aprire il file**

5. **Settare i parametri**

	``` 
	- Riga 26: Valore Stop Loss
	- Riga 36: Contiene l'url del server al quale multicharts si connetterà
	- Riga 47: Questa riga serve per decidere se inserire lo stop loss nella strategia o meno. 
			- Se non lo si vuole inserire, bisogna aggiungere '//' facendo diventare la riga 
				da *GenerateStopLoss(StopLoss);* a *//GenerateStopLoss(StopLoss);*
			- Se si vuole lo stop loss la riga deve essere *GenerateStopLoss(StopLoss);*
	``` 

6. **Importare le librerie**
	
	``` 
	Tasto destro su una riga qualsiasi del codice -> References
	Una volta aperta la finestra cliccare su Add Reference
	Selezionare il file `ClassLibrary1.dll`
	Premere su apri
	Premere su close
	``` 

7. **Premere la rotellina blu nella barra superiore**

Se tutto è andato a buon fine nel riquadro a destra, a sinstra del nome del file avremo un cerchio verde con all'interno una spunta bianca

------

### Eseguire il segnale su Multicharts

Da Multicharts:

1. **Tasto destro sul grafico -> insert study**

2. **Selezionare la colonna Signal**

3. **Cercare il file Predizioni4**

4. **Premere su ok**

5. **Si aprirà un riquadro, cliccare su close**

# Visualizzare il report e salvarlo su file

### Per visualizzare il report

Una volta finito il calcolo del segnale

Da Multicharts:

1. **Premere il pulsante a forma di cavallo degli scacchi nella barra superiore (Se ci si passa con il mouse sopra esce la scritta Strategy Performance Report)**

Si aprirà un riquadro dove sarà possibile analizzare tutti i vari i dati della strategia

------

### Per fare l'export

1. **Premere sull'icona di salvataggio**

2. **Selezionare la cartella in cui si vuole salvare il file**

3. **Scegliere il nome del file**

4. **Premere su salva**

5. **Selezionare quali sezioni si vuole esportare**

6. **Cliccare su ok**

7. **Aspettare la fine del report**

Una volta terminato si potrà aprire il file excel appena creato
