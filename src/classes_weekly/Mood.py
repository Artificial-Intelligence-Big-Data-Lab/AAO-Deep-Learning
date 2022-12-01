import os
import sys
import math
import pandas as pd

# Importo il file di configurazione del database
from config import db

'''
 ' This class is used to manage the stocktwits datasets
 ' aggregate in csv, in order to calculate the occurences of moods
 ' per day
 '''
class Mood:
    
    # Define the dataframe
    df = pd.DataFrame()

    # Moods per day DataFrame
    count_moods_per_day = pd.DataFrame()

    # Path locale dei datasets
    twits_path = '../twits/'
    stocktwits_path = twits_path + 'stocktwits/'
    twitter_path = twits_path + 'twitter/'

    acepted_sources = ['twitter', 'stocktwits']

    stocktwits_select = 'stocktwits.id, stwits_id, stwits_created_at, full_text, symbols, likes_count, sentiment, user_id, user_followers_count, user_classification'
    twitter_select = 'twitter.id, tweet_id, twitter_created_at, creation_date, creation_time, full_text, hashtags, lang, retweet_count, favorite_count, possibly_sensitive, sentiment, user_id, user_followers_count, user_verified'

    # Dataframe di lavoro
    df = pd.DataFrame()
    mix = pd.DataFrame(columns=['date', 'mix'])

    db_connection = db.social_crawler_conn
    # Costruttore, setto la connessione al Database
    def __init__(self, keyword, source):
        self.db_connection = db.social_crawler_conn

        self.__read(keyword, source)
        self.calculate_count_moods_per_day()

    '''
    ' Leggo un dataset specificato come parametro
    ' Aggiungo una colonna date_time unendo i campi date e time
    '''
    def __read(self, keyword, source):

        full_file_path = ''

        if source == 'twitter':
            full_file_path = self.twitter_path + keyword + '.csv'
        else:
            full_file_path = self.stocktwits_path + keyword + '.csv'

        if os.path.isfile(full_file_path) is not True:
            self.__initialize(keyword, source, full_file_path)

        self.df = pd.read_csv(full_file_path).set_index('id')

        self.df = self.df.fillna(value={'sentiment': 'null'})

    '''
    ' @private 
    ' Prima di leggere o inizializzare
    ' controllo che il parametro source specificato sia accettato dall'array 
    ' e che non sia un parametro vuoto
    '''
    def __validate(self, source=None):
        # Se non specifico il dataset esco dalla funzione
        if source == None:
            sys.exit("Mood.__validate: Source can't be none")

        if source not in self.acepted_sources:
            sys.exit("Mood.__validate: The " + source + " source is not accepted.")

    '''
    ' @public
    ' Questa funzione viene lanciata se non è presente
    ' il datasets in locale. Viene letto da mysql e salvato
    ' nella cartella a seconda del social network indicato
    ' dal parametro source
    '''
    def __initialize(self, keyword=None, source=None, path=None):
        self.__validate(source=source)

        # Se la cartella del dataset non esiste, la creo a runtime
        if not os.path.isdir(self.stocktwits_path):
            os.makedirs(self.stocktwits_path)

        # Se la cartella del dataset non esiste, la creo a runtime
        if not os.path.isdir(self.twitter_path):
            os.makedirs(self.twitter_path)

        # Leggo da mysql il dataset che mi serve salvarmi in locale
        print("Reading the dataset from Mysql... It can take a while...")

        if source == 'stocktwits':
            df = pd.read_sql('SELECT ' + self.stocktwits_select + ' FROM ' + source + ' LEFT JOIN keywords ON ' + source +'.keyword_id = keywords.id WHERE keyword_name = "' + keyword + '" order by creation_date, creation_time ASC', con=self.db_connection)
        else:
            df = pd.read_sql('SELECT ' + self.twitter_select + ' FROM ' + source + ' LEFT JOIN keywords ON ' + source + '.keyword_id = keywords.id WHERE keyword_name = "' + keyword + '" order by creation_date, creation_time ASC ', con=self.db_connection)

        df = df.set_index('id')
        # Casto il campo Time. Leggendo da mysql viene convertito in timedelta e non in time.
        df['creation_date'] = pd.to_datetime(df['stwits_created_at']).dt.date
        df['creation_time'] = pd.to_datetime(df['stwits_created_at']).dt.time

        # Salvo in csv quando appena letto
        df = df.drop('stwits_created_at', axis=1)
        df.to_csv(path, header=True, index=True, date_format=str)

    # Count foreach day, the number of Bullish, Bearish and Null sentiment
    def calculate_count_moods_per_day(self):
        self.count_moods_per_day = pd.crosstab(self.df['creation_date'], self.df['sentiment'])
        self.count_moods_per_day = self.count_moods_per_day.reset_index()
        self.count_moods_per_day = self.count_moods_per_day.rename(index=str, columns={"Bullish": "bullish", "Bearish": "bearish"})
        
    # Save the mood_per_day dataframe in a csv. If mood_per_day then get_count_moods_per_day() is launched in order to calculate
    # the moods counter
    def save_count_moods_per_day(self, output_path, output_name):
        if self.count_moods_per_day.empty:
            self.calculate_count_moods_per_day()
        self.count_moods_per_day.to_csv(output_path + output_name, sep=',', encoding='utf-8', index=False)

    # Get subset of count_moods_per_day
    def get_subset_count_moods_per_day(self, start_date, end_date):
        # Convert the date to a datetime
        self.count_moods_per_day['date'] = pd.to_datetime(self.count_moods_per_day['date'])  
        # Search mask
        mask = (self.count_moods_per_day['date'] > start_date) & (self.count_moods_per_day['date'] <= end_date)
        # Get the subset of df
        subset = self.count_moods_per_day.loc[mask]
        return subset


    # Get only the twits without "Null" sentiment
    def get_twits_without_null_sentiment(self):
        return self.df[self.df.sentiment != "null"]

    # Get only the twits with "Null" sentiment
    def get_twits_with_null_sentiment(self):
        return self.df[self.df.sentiment == "null"]

    # Get only the twits without "Bearish" sentiment
    def get_twits_with_bearish_sentiment(self):
        return self.df[self.df.sentiment != "bearish"]

    # Get only the twits without "Bullish" sentiment
    def get_twits_with_bullish_sentiment(self):
        return self.df[self.df.sentiment != "bullish"]

    # Get subset of count_moods_per_day
    def get_subset_count_moods_per_day(self, start_date, end_date):
        # Convert the date to a datetime
        self.count_moods_per_day['creation_date'] = pd.to_datetime(self.count_moods_per_day['creation_date'])
        # Search mask
        mask = (self.count_moods_per_day['creation_date'] > start_date) & (self.count_moods_per_day['creation_date'] <= end_date)
        # Get the subset of df
        subset = self.count_moods_per_day.loc[mask]
        return subset

    def calculate_mix_per_range(self, start_date, end_date):
        # 10 giorni scelti per testare la formula
        count_moods = self.get_subset_count_moods_per_day(start_date, end_date)

        xn_list = list()
        yn_list = list()

        # Calcolo xn e yn per i 9 giorni precedenti al 10 giorno
        #for i in range(1, 10):
        for i, value in enumerate(list(count_moods)):
            total_today = count_moods.iloc[i]['bearish'] + count_moods.iloc[i]['bullish'] + count_moods.iloc[i]['null']
            total_yesterday = count_moods.iloc[i - 1]['bearish'] + count_moods.iloc[i - 1]['bullish'] + \
                              count_moods.iloc[i]['null']

            xn = (count_moods.iloc[i]['bullish'] / total_today) - (count_moods.iloc[i - 1]['bullish'] / total_yesterday)
            yn = (count_moods.iloc[i]['bearish'] / total_today) - (count_moods.iloc[i - 1]['bearish'] / total_yesterday)

            xn_list.append(round(xn, 4))
            yn_list.append(round(yn, 4))

        sum_xn_list = sum(xn_list)
        sum_yn_list = sum(yn_list)

        u_x = (1.0 / 9.0) * sum_xn_list
        u_y = (1.0 / 9.0) * sum_yn_list

        bull_today = 0.0
        bear_today = 0.0

        for i, value in enumerate(list(count_moods)):
            bull_today += (xn_list[i] - u_x) ** 2
            bear_today += (yn_list[i] - u_y) ** 2

        bull_today = math.sqrt(bull_today)
        bear_today = math.sqrt(bear_today)

        if bear_today == 0:
            mix = -1.0
        else:
            mix = (bull_today / bear_today) - 1.0

        return round(mix, 4), end_date

    #
    def calculate_mix(self):
        first = self.df.iloc[-1]['creation_date']
        last = self.df.iloc[0]['creation_date']

        start_date = pd.to_datetime(first)
        end_date = pd.to_datetime(last)

        while (start_date - pd.Timedelta(days=10)) >= end_date:
            # Calcolo i dieci giorni prima
            range_min = start_date - pd.Timedelta(days=10)
            range_max = start_date

            # calcolo mix per questo giorno
            mix, date = self.calculate_mix_per_range(range_min, range_max)
            #print("mix: " + str(mix) + " - date: " + str(date))

            self.mix = self.mix.append({'date': date, 'mix': mix}, ignore_index=True)

            # Go on
            start_date = start_date - pd.Timedelta(days=1)

        return self.mix
