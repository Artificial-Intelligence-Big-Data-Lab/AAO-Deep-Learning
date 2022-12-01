import pandas as pd
from classes.Market import Market
from classes.Gaf import Gaf
from classes.ImagesMerger import ImagesMerger
from classes.VggHandler import VggHandler
from classes.ResultsHandler import ResultsHandler

'''
' Preprocessing, credo il DF con un solo mese di dati
'
print('Start downloading sp500')
sp500 = Market(dataset='sp500')
mese = sp500.get_df_by_data_range('2016-01-01', '2016-02-03')
mese.to_csv('../datasets/sp500_mese.csv', header=True, index=True)
'''

'''
' Modulo 1 - Scegli un mercato, stampi un mese di dati con risoluzione 1 ora, 6 ore, 
' 1 giorno, 2 giorni, 7 giorni. Mi mandi il mese di dati raw e i risultati aggregati.
'
print('Start downloading sp500')
sp500 = Market(dataset='sp500')
one_h = sp500.group(freq='1h', nan=False) # un ora
six_h = sp500.group(freq='6h', nan=False) # sei ore
one_d = sp500.group(freq='1d', nan=False) # un giorno
two_d = sp500.group(freq='2d', nan=False) # due giorni
seven_d = sp500.group(freq='7d', nan=False) # sette giorni
one_week = sp500.group(freq='1w', nan=False) # una settimana

one_h.to_csv('../datasets/debug_diego/one_h.csv', header=True, index=True)
six_h.to_csv('../datasets/debug_diego/six_h.csv', header=True, index=True)
one_d.to_csv('../datasets/debug_diego/one_d.csv', header=True, index=True)
two_d.to_csv('../datasets/debug_diego/two_d.csv', header=True, index=True)
seven_d.to_csv('../datasets/debug_diego/seven_d.csv', header=True, index=True)
one_week.to_csv('../datasets/debug_diego/one_week.csv', header=True, index=True)
'''

'''
' Modulo 2 - test su 1 mese con immagini di 10x10 dove primi 10 giorni uguali a ultimi 10+1
'
''Preprocessing

df = pd.read_csv('../datasets/debug_diego/modulo_2/df_one_d_original.csv')

df = df.drop(columns=['open', 'close', 'high', 'low', 'up','volume'])
df = df.set_index('date_time')
df.to_csv('../datasets/debug_diego/modulo_2/df_custom.csv', header=True, index=True)


' Genero le immagini sul custom, quindi la prima e l'ultima sono uguali
df = pd.read_csv('../datasets/debug_diego/modulo_2/df_custom.csv')

df['date_time'] = pd.to_datetime(df['date_time'])

gaf = Gaf()

gaf.run(df=df, dataset_name="debug_diego", subfolder="1day", type='gadf', size=10)
gaf.run(df=df, dataset_name="debug_diego", subfolder="1day", type='gasf', size=10)
'''

'''
' Modulo 3 - esempio di merge
' ho fatto copia e incolla delle immagini per avere diverse risoluzioni copia
'''
#print('Merging images...')
# Unisco  le immagini in una sola immagine 20x20
#imgmerger = ImagesMerger()

#imgmerger.run(input_folders='debug_diego',
#              resolutions=['1day', '11day', '111day', '1111day'],
#              signals=['delta'],
#              positions=[(0, 0), (10, 0), (0, 10), (10, 10)],
#              type='gadf',
#              img_size=[20, 20],
#              output_path="debug_diego_merge")

'''
' Modulo 4
' Stampi training, validation e test con label su un mese con img di 10 righe
'
sp500 = Market(dataset='sp500')

label_current_day = sp500.get_label_current_day(freq='1d', columns=['open', 'close', 'delta']) # un giorno
label_next_day = sp500.get_label_next_day(freq='1d', columns=['open', 'close', 'delta']) # un giorno

label_current_day = label_current_day.rename(index=str, columns={"label": "current_day_label"}).reset_index()
label_next_day = label_next_day.rename(index=str, columns={"label": "next_day_label"}).reset_index()

#print(label_current_day)
#print(label_next_day)

merge = pd.merge(label_current_day, label_next_day, how='inner')

merge.to_csv('../datasets/debug_diego/modulo_4/df_label.csv', header=True, index=True)
'''


'''
' Modulo 5
' proseguire con stampa di walk su un mese di dati con 10-5-5 (tra, eva, test) e 
' mettere risultati di classificazione e label corrette e anche calcolo accuracy
'''

training_set = [
                ['2016-02-01', '2016-02-11'], # walk 1
                ['2016-02-11', '2016-02-21'], # walk 2
                ['2016-02-21', '2016-03-02'], # walk 3
                ]

validation_set = [
                ['2016-02-11', '2016-02-16'], # walk 1
                ['2016-02-21', '2016-02-26'], # walk 2
                ['2016-03-02', '2016-03-07'], # walk 3
        ]

test_set = [
        ['2016-02-16', '2016-02-21'], # walk 1
        ['2016-02-26', '2016-03-02'], # walk 2
        ['2016-03-07', '2016-03-12'], # walk 3
        ]


vgg = VggHandler()

vgg.net_config(epochs=200, number_of_nets=20)

vgg.run_initialize(dataset='sp500',
        input_folder='delta_experiment/gadf/delta/',
        training_set=training_set,
        validation_set=validation_set,
        test_set=test_set,
        input_shape=(40,40,3),
        output_folder='diego_experiment_debug')

#vgg.run()

results_handler = ResultsHandler(experiment_name='diego_experiment_debug', dataset='sp500')

#vgg.get_predictions(set_type='training')
results_handler.generate_ensemble_and_plots(set_type='training')
results_handler.calculate_return(set_type='training')

#vgg.get_predictions(set_type='validation')
results_handler.generate_ensemble_and_plots(set_type='validation')
results_handler.calculate_return(set_type='validation')


#vgg.get_predictions(set_type='test')
results_handler.generate_ensemble_and_plots(set_type='test')
results_handler.calculate_return(set_type='test')