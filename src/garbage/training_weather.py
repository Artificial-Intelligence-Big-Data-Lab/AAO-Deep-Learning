import os 
import pandas as pd 
from classes.VggHandler import VggHandler
from classes.ResultsHandler import ResultsHandler
from classes.Market import Market

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID";
os.environ["CUDA_VISIBLE_DEVICES"]="0";


# FIRST
#training_set = [['2012-10-20', '2016-12-31']]
#validation_set = [ ['2017-01-01', '2017-09-20']]
#test_set = [['2017-09-21', '2017-11-30']]


training_set =      [['2012-10-20', '2015-10-20'],  ['2013-03-20', '2016-03-20'],  ['2013-08-20', '2016-08-20'],  ['2014-01-20', '2017-01-20']]
validation_set =    [['2015-10-21', '2016-03-20'],  ['2016-03-21', '2016-08-20'],  ['2016-08-21', '2017-01-20'],  ['2017-01-21', '2017-06-20']]
test_set =          [['2016-03-21', '2016-08-20'],  ['2016-08-21', '2017-01-20'],  ['2017-01-21', '2017-06-20'],  ['2017-06-21', '2017-11-20']]

''' Converto in date e time la colonna date_time
df = pd.read_csv('../datasets/meteo_formatted.csv')
df = df.set_index('id')

df['date_time'] = pd.to_datetime(df['date_time'], format='%Y-%m-%d %H:%M:%S')

df['date'] = df['date_time'].dt.date
df['time'] = df['date_time'].dt.time
df.to_csv('../datasets/meteo_formatted.csv')
'''


market = Market(dataset='meteo_formatted')
label = market.get_label_next_day_using_close(columns=['date_time', 'delta'])

grouped = self.df.drop(['date', 'time', 'delta'], axis=1).groupby(pd.Grouper(key='date_time', freq=freq), sort=True).agg({
                'open': 'first',
                'close': 'last',
                'close_adj': 'last',
                'high': 'max',
                'low': 'min',
                'up': 'sum',
                'down': 'sum',
                'volume': 'sum'
            })
            
label.to_csv('../datasets/meteo_formatted_with_label.csv')
'''
input_images_folders = ['merge_meteo/gadf/delta/']
input_datasets = ['meteo']


predictions_dataset = 'meteo_formatted'
predictions_images_folder = 'merge_meteo/gadf/delta/'

vgg = VggHandler()

vgg.net_config(epochs=150, number_of_nets=20, save_pkl=True, save_model_history=True, model_history_period=150)

vgg.run_initialize( predictions_dataset=predictions_dataset,
                    predictions_images_folder=predictions_images_folder,

                    input_images_folders=input_images_folders,
                    input_datasets=input_datasets,

                    training_set=training_set,
                    validation_set=validation_set,
                    test_set=test_set,
                    
                    input_shape=(40,40,3),
                    output_folder='meteo_4walks')

 #vgg.run_2D()


results_handler = ResultsHandler(experiment_name='meteo_4walks', dataset='meteo_formatted')

vgg.get_predictions_2D(set_type='validation')

#results_handler.generate_ensemble(set_type='validation')
#results_handler.generate_plots(set_type='validation')
#results_handler.generate_csv_aggregate_by_walk(set_type='validation')
#results_handler.generate_csv_aggregate_unique_walk(set_type='validation')


vgg.get_predictions_2D(set_type='test')
#results_handler.generate_ensemble(set_type='test')
#results_handler.generate_plots(set_type='test')
#results_handler.generate_csv_aggregate_by_walk(set_type='test')
#results_handler.generate_csv_aggregate_unique_walk(set_type='test')
'''