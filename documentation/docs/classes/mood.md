# Mood()

**Classe non ultimata, di conseguenza la doc potrebbe essere incompleta o non funzionante**

Questa classe serve per poter ottenere, a partire dai valori `Bearish` e `Bullish` dei twits un indice di mood calcolato sulla base della formula presente nell'articolo PsySignal. 

Il costruttore prende due parametri: `keyword` e `source` che indicano rispettivamente la parola chiave usata per scaricare i twits e la fonte da cui sono stati scaricati. 

E' bene ricordare che questa classe lavora con i twits salvati dai social network `twitter` e `stocktwits` tramite il framework [**Social Crawler**](https://github.com/AsoStrife/PHD-Social-Crawler).


**Parametri**:

- **keyword**: _string_
- **source**: _string_ ['twitter', 'stocktwits']

In fase di inizializzazione, il costruttore controlla che il dataset specificato dalla coppia di parametri `keyword` e `source` esiste in locale. Nel caso in cui l'esito fosse negativo, esegue una query al server MySql remoto e scarica i dati in locale. Questa operazione potrebbe prendere diverso tempo. 


# calculate_mix()
Questo metodo non prende parametri in ingresso. Serve a lanciare a runtime il calcolo dell'indice di mood per ogni giorno del dataset, partendo dal più recente. 

Restituisce un nuovo dataframe strutturato come da esempio. 

**Esempio**:

```
from classes import Mood

mood = Mood.Mood(keyword='SPX', source='stocktwits')

mix = mood.calculate_mix()

print(mix)

------------------
 date     mix
0    2019-04-02  0.5448
1    2019-04-01  0.8230
2    2019-03-31 -0.2772
3    2019-03-30 -0.0957
4    2019-03-29 -0.0638
------------------

```
