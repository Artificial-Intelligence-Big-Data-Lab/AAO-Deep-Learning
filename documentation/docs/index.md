# Intro
Documentazione ufficiale del repository AIBD Market Nets 

## Struttura del progetto

Il repository è strutturato in maniera meticolosa per avere tutto ciò che serve esattamente dove serve.

```
PHD-Market-Nets
│   .gitignore
│   .get-pip.py
|   venv-configuration.txt    
│
└───datasets
│   │   sp500.csv
│   │   bund.csv
│   
└───documentation
|   │   docs
|   │   mkdocs.yml
└───images
│     
│     
└───sql
|   │   dataset_general.sql
│     
└───src
|___|__classes
|   |    |   Gaf.py   
|   |    |   ImagesMerger.py
|   |    |   Market.py
|   |    |   Mood.py
|   |   config
|   |    |   db.py
|   |   utils
|   |    |   csv_to_mysql.py   
|   |   vgg16
|   |    |   small_vgg_16.py   
|   |    |   train_classify_net.py
|   |   __init__.py

│     
└───twits
|   │   stocktwits
|   │   twitter
|_
```

All'interno del repository troveremo le seguenti cartelle con le seguenti funzionalità: 

- **datasets**: in cui verranno salvati automaticamente i datasets su cui si andrà a lavorare. Verranno scaricati in automatico dalla classe Market al primo utilizzo.
- **documentation**: la documentazione del progetto, ovvero questa che si sta leggendo, che è possibile compilare tramite Mkdocs. A fine pagina si trova il link con la relativa documentazione.
- **images**: dove verranno salvate tutte le immagini generate dalla classe Gaf. La struttura interna di questa cartella verrà fornita nella sezione dedicata alla classe Gaf.
- **sql**: contiene uno script di base per aggiungere una tabella al database mysql. Maggiori spiegazioni verranno date nella sezione _"Convertire CSV to Mysql"_.
- **src**: contiene il codice del progetto. Per poter importare/eseguire il codice, inserire i file all'interno di questa cartella. Di base viene fornito un file test.py su cui lavorare. src contiene le seguenti sottocartelle: 
    - **classes**: contiene le classi del progetto. Fondamentali per eseguire tutto il codice.
    - **config**: contiene la configurazione per connettersi al server mysql remoto.
    - **utils**: contiene script generici che possono essere utilizzati per vari scopi.
    - **vgg16**: è la cartella che contiene la CNN con i relativi script utili alla sua esecuzione. 
- **twits**: conterrà i twits scaricati da twitter e stocktwits. Ci saranno le varie sottocartelle al suo interno. 

## Prerequisiti

Per poter far funzionare il repository occorre aver installato sulla propria macchina le seguenti dipendenze: 

- Python3 (non funzionerà con python2)
- Pip
- Virtualenv
- Tkinter
- Git

Per poter installare le dipendenze eseguire i seguenti comandi: 

```bash
sudo apt-get update

# Python3
sudo apt-get install python3.6

# Pip
curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py
python get-pip.py
python -m pip install --upgrade pip

# Virtualenv
sudo pip install virtualenv 

# Tkinter
sudo apt-get install python3-tk

# Git 
apt-get install git-core
```

## Installazione
Il repository privato è raggiungibile all'indirizzo: [https://github.com/AsoStrife/PhD-Market-Nets](https://github.com/AsoStrife/PhD-Market-Nets), solamente post autorizzazione del proprietario (@AsoStrife).


Per clonare il repository: 

```bash
git clone https://github.com/AsoStrife/PhD-Market-Nets.git
```

Una volta clonato il repository, per poter eseguire il codice sorgente è necessario scaricare tutte le dipendenze tramite **pip**. 

Il miglior modo per farlo è utilizzando un virtual environment.

Per creare un virtualenv da Ubuntu: 

```bash
virtualenv --system-site-packages -p python3 ./venv
source ./venv/bin/activate
```

In alternativa, utilizzando **Pycharm** si può creare un virtualenv seguendo la seguente guida: [https://www.jetbrains.com/help/pycharm/creating-virtual-environment.html](https://www.jetbrains.com/help/pycharm/creating-virtual-environment.html).


**Nel caso ci fossero conflitti con l'ambiente virtuale di pycharm e pip questa può essere una soluzione**: 

```bash
python -m pip uninstall pip setuptools

curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py
python get-pip.py
python -m pip install --upgrade pip
```

In entrambi i casi si consiglia come nome della cartella `./venv` in quanto è già inserita nel `.gitignore`.

**Importante**: per poter utilizzare **Tensorflow**, **Tensorflow-GPU** bisogna utilizzare Python 3.6 con pip aggiornato all'ultima versione. 
 

Una volta installato e configurato l'ambiente virtuale, per installare tutte le dipendenze basta eseguire il comando: 

```bash
pip install -r venv-configuration.txt
```

Per aggiornare la lista delle dipendenze, in caso si voglia aggiornare il file _venv-configuration.txt_ e pushare le modifiche nel repository, utilizzare il seguente comando: 

```
pip freeze > venv-configuration.txt
```

# Configurazione del database

Nella path `./config/db.py` è possibile trovare un file in cui sono presenti le stringhe di connessione al database, che viene utilizzato a runtime per generare i dataset in locale. Di default il file è preconfigurato con le credenziali di accesso al server interno del gruppo di ricerca. 

Se non si vuole lavorare con un DB remoto ma uno locale si può sempre cambiare le credenziali, l'importante è poi assicurarsi di avere una copia dei dataset sul DB mysql. Il dump sql attualmente 
pesa 150mb, sarà presto disponibile al download a tutti i membri del gruppo di ricerca autorizzati.

Per potersi connettere al DB è necessario essere all'interno della rete dell'Università. 

```python
import mysql.connector as sql

# Variabile per la connessione al DB temporaneamente settata a None
db_connection = None

# Credenziali per la connessione al DB
db_host = ''
db_user = ''
db_password = ''

datasets_conn = sql.connect(host=db_host, database="datasets", user=db_user, password=db_password)

social_crawler_conn = sql.connect(host=db_host, database="social_crawler", user=db_user, password=db_password)
```

# Mkdocs

Per generare questa documentazione è stato utilizzato [Mkdocs](https://www.mkdocs.org/).
Per compilare in locale la documentazione e visualizzarla senza connessione internet basta installare mkdocs all'interno del proprio sistema con il comando: 

```bash
pip install mkdocs
```

Successivamente, dalla cartella `./documentation` utilizzare i comandi: 

```bash
# Lanciare il sito in localhost all'indirizzo http://127.0.0.1:8000
mkdocs serve 

# Per compilare la soluzione in un sito html statico
mkdocs build
```

## Links

- [GADF & GASF](https://pyts.readthedocs.io/en/latest/auto_examples/image/plot_gaf.html#sphx-glr-auto-examples-image-plot-gaf-py)
- [Lucena PYTS link](https://github.com/johannfaouzi/pyts)
- [Configuring Virtualenv Environment Pycharm](https://www.jetbrains.com/help/pycharm/creating-virtual-environment.html)
- [Mkdocs](https://mkdocs.org)

## Authors

- Andrea Corriga (@AsoStrife)
