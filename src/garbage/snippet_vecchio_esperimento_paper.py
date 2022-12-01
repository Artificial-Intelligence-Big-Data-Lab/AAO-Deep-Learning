''' GENERARE TRAINING, VALIDATION, TEST SET IN MODO AUTOMATICO
TRAINING_SET_SIZE = 10 # YEARS
VALIDATION_SET_SIZE = 1 # MONTHS
TEST_SET_SIZE = 6 # MONTHS

# genero la coppia data inizio - data fine per il training set usando l'offset  specificato
def get_training_set(last_date):
    start_date = (last_date - pd.DateOffset(years=TRAINING_SET_SIZE)).date()
    return [str(start_date), str(last_date)]

# genero la coppia data inizio - data fine per il validation set usando l'offset  specificato
def get_validation_set(last_date):
    start_date = (last_date - pd.DateOffset(months=VALIDATION_SET_SIZE)).date()

    return [str(start_date), str(last_date)]

# genero la coppia data inizio - data fine per il test set usando l'offset  specificato
def get_test_set(last_date):
    start_date = (last_date - pd.DateOffset(months=TEST_SET_SIZE)).date()

    return [str(start_date), str(last_date)]


training_set = []
validation_set = []
test_set = []

# GOLD
#latest_date_test = pd.to_datetime('2018-6-30').date()
#first_day_test = pd.to_datetime('2015-01-31').date()

#SP500 | DAX
latest_date_test = pd.to_datetime('2018-06-30').date()
first_day_test = pd.to_datetime('2010-05-31').date()

print("Generating dates for training...")
while pd.to_datetime(first_day_test) <= pd.to_datetime(latest_date_test):
    # Per il training tolgo la dimensione del test + dimensione del validation + 2 giorni (1giorno di validation + 1 giorno di training)
    latest_training_day = latest_date_test - pd.DateOffset(months=TEST_SET_SIZE) - pd.DateOffset(months=VALIDATION_SET_SIZE) - pd.DateOffset(days=2)
    latest_date_training = latest_training_day.date()
    
    # GOLD
    #if(pd.to_datetime(latest_date_training) < pd.to_datetime('2015-01-26')):
    #    break
    
    # SP500 | DAX
    if(pd.to_datetime(latest_date_training) < pd.to_datetime('2010-04-26')):
        break

    # Per il validation tolgo la dimensione del test + 1 giorno (dovrebbe finire il giorno prima del test)
    latest_validation_day = latest_date_test - pd.DateOffset(months=TEST_SET_SIZE) - pd.DateOffset(days=1)
    latest_date_validation = latest_validation_day.date()

    # creo le liste contenente le coppie per ogni test
    training_set.append(get_training_set(latest_date_training))
    validation_set.append(get_validation_set(latest_date_validation))
    test_set.append(get_test_set(latest_date_test))

    # aggiorno la data di fine per la nuova coppia di test set togliendo l'offset
    # si potrebbe togliere anche un giorno per evitare di avere un giorno in comune per ogni test set
    latest_date_test = (latest_date_test - pd.DateOffset(months=VALIDATION_SET_SIZE)).date()

training_set.reverse()
validation_set.reverse()
test_set.reverse()
'''


''' sp30
input_images_folders = ['merge_sp500/gadf/delta/', 
                        'merge_msft/gadf/delta/', 
                        'merge_amzn/gadf/delta/', 
                        'merge_aapl/gadf/delta/',
                        'merge_googl/gadf/delta/',
                        'merge_goog/gadf/delta/',
                        'merge_brk_b/gadf/delta/',
                        'merge_fb/gadf/delta/',
                        'merge_v/gadf/delta/',
                        'merge_jnj/gadf/delta/',
                        'merge_jpm/gadf/delta/',
                        'merge_xom/gadf/delta/',
                        'merge_wmt/gadf/delta/',
                        'merge_pg/gadf/delta/',
                        'merge_bac/gadf/delta/',
                        'merge_ma/gadf/delta/',
                        'merge_dis/gadf/delta/',
                        'merge_pfe/gadf/delta/',
                        'merge_vz/gadf/delta/',
                        'merge_csco/gadf/delta/',
                        'merge_unh/gadf/delta/',
                        'merge_t/gadf/delta/',
                        'merge_cvx/gadf/delta/',
                        'merge_ko/gadf/delta/',
                        'merge_hd/gadf/delta/',
                        'merge_mrk/gadf/delta/',
                        'merge_wfc/gadf/delta/',
                        'merge_intc/gadf/delta/',
                        'merge_ba/gadf/delta/',
                        'merge_cmcsa/gadf/delta/', 
                        ]
input_datasets = ['sp500', 'msft', 'amzn', 'aapl', 'googl', 'goog', 'brk_b', 'fb', 'v', 'jnj', 'jpm', 'xom', 'wmt', 'pg', 'bac', 'ma', 'dis', 'pfe', 'vz', 'csco', 'unh', 't', 'cvx', 'ko', 'hd', 'mrk', 'wfc', 'intc', 'ba', 'cmcsa']
'''


# FIRST
#training_set = [['2001-06-26', '2011-06-26']]
#validation_set = [ ['2011-06-27', '2011-07-27']]
#test_set = [['2011-07-28', '2012-01-28']]

# walk 0 - walk 44 - walk 83
#training_set =      [['2000-04-26', '2010-04-26'],  ['2003-12-26', '2013-12-26'],  ['2007-03-28', '2017-03-28']]
#validation_set =    [['2010-04-27', '2010-05-27'],  ['2013-12-27', '2014-01-27'],  ['2017-03-29', '2017-04-29']]
#test_set =          [['2010-05-28', '2010-11-28'],  ['2014-01-28', '2014-07-28'],  ['2017-04-30', '2017-10-31']]

#training_set =      [['2004-02-01', '2013-01-31']]
#validation_set =    [['2013-02-01', '2013-07-31']]
#test_set =          [['2013-08-01', '2014-01-31']]

# walk n 25
#training_set =      [['2002-05-26', '2012-05-26']]
#validation_set =    [['2012-05-27', '2012-06-27']]
#test_set =          [['2012-06-28', '2012-12-28']]