import os
from classes.VggHandler import VggHandler
from classes.ResultsHandler import ResultsHandler
from classes.Market import Market

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID";
os.environ["CUDA_VISIBLE_DEVICES"]="0";

# Training set walks
training_set = [
                ['2000-02-01', '2009-01-30'],
                ['2000-08-01', '2009-07-31'],
                ['2001-02-01', '2010-01-31'],
                ['2001-08-01', '2010-07-30'],
                ['2002-02-01', '2011-01-31'],
                ['2002-08-01', '2011-07-31'],
                ['2003-02-02', '2012-01-31'],
                ['2003-08-01', '2012-07-31'],
                ['2004-02-01', '2013-01-31'],
                ['2004-08-01', '2013-07-31'],
                ['2005-02-01', '2014-01-31']
            ]

# Validation set walks
validation_set = [
                ['2009-02-01', '2009-07-31'],
                ['2009-08-02', '2010-01-31'],
                ['2010-02-01', '2010-07-30'],
                ['2010-08-01', '2011-01-31'],
                ['2011-02-01', '2011-07-31'], 
                ['2011-08-01', '2012-01-31'],
                ['2012-02-01', '2012-07-31'],
                ['2012-08-01', '2013-01-31'],
                ['2013-02-01', '2013-07-31'],
                ['2013-08-01', '2014-01-31'],
                ['2014-02-02', '2014-07-31']   
            ]

# Test set walks
test_set = [
            ['2009-08-02', '2010-01-31'],
            ['2010-02-01', '2010-07-30'],
            ['2010-08-01', '2011-01-31'],
            ['2011-02-01', '2011-07-31'],
            ['2011-08-01', '2012-01-31'], 
            ['2012-02-01', '2012-07-31'],
            ['2012-08-01', '2013-01-31'],
            ['2013-02-01', '2013-07-31'],
            ['2013-08-01', '2014-01-31'],
            ['2014-02-02', '2014-07-31'],
            ['2014-08-01', '2015-01-30'] 
        ]


# The companies that you want to use. The two variables has the same order
input_images_folders = [
                    'merge_sp500/gadf/delta/', 

                    'merge_a/gadf/delta/',
                    'merge_aal/gadf/delta/',
                    'merge_aap/gadf/delta/',
                    'merge_aapl/gadf/delta/',
                    'merge_abbv/gadf/delta/',
                    'merge_abc/gadf/delta/',
                    'merge_abt/gadf/delta/',
                    'merge_acn/gadf/delta/',
                    'merge_adbe/gadf/delta/',
                    'merge_adi/gadf/delta/',
                    'merge_adm/gadf/delta/',
                    'merge_adp/gadf/delta/',
                    'merge_ads/gadf/delta/',
                    'merge_adsk/gadf/delta/',
                    'merge_aee/gadf/delta/',
                    'merge_aep/gadf/delta/',
                    'merge_aes/gadf/delta/',
                    'merge_afl/gadf/delta/',
                    'merge_agn/gadf/delta/',
                    'merge_aig/gadf/delta/',
                    'merge_aiv/gadf/delta/',
                    'merge_amzn/gadf/delta/',
                    'merge_jnj/gadf/delta/',
                    'merge_jpm/gadf/delta/',
                    'merge_ko/gadf/delta/',
                    'merge_mmm/gadf/delta/',
                    'merge_msft/gadf/delta/',


                    ]
input_datasets = [
                    'sp500', 
                    'a', 
                    'aal', 
                    'aap', 
                    'aapl', 
                    'abbv', 
                    'abc', 
                    'abt', 
                    'acn', 
                    'adbe', 
                    'adi', 
                    'adm', 
                    'adp', 
                    'ads', 
                    'adsk', 
                    'aee', 
                    'aep', 
                    'aes', 
                    'afl', 
                    'agn', 
                    'aig', 
                    'aiv', 
                    'amzn', 
                    'jnj', 
                    'jpm', 
                    'ko', 
                    'mmm', 
                    'msft'
                ]

# The market you want to predict
predictions_dataset = 'sp500'
predictions_images_folder = 'merge_sp500/gadf/delta/'

# DEFINE YOUR EXPERIMENT NAME !!!!
experiment_name = 'experiment_anselmo'

vgg = VggHandler()
vgg.net_config(epochs=200, number_of_nets=20, save_pkl=False, save_model_history=True, model_history_period=50)
vgg.run_initialize(predictions_dataset=predictions_dataset,
                    predictions_images_folder=predictions_images_folder,

                    input_images_folders=input_images_folders,
                    input_datasets=input_datasets,

                    training_set=training_set,
                    validation_set=validation_set,
                    test_set=test_set,
                    
                    input_shape=(40,40,3),
                    output_folder=experiment_name) #delta_experiment_third_run


vgg.run_2D()

results_handler = ResultsHandler(experiment_name=experiment_name, dataset='sp500')

vgg.get_predictions_2D(set_type='validation')
results_handler.generate_ensemble(set_type='validation')
results_handler.generate_plots(set_type='validation')
results_handler.generate_csv_aggregate_by_walk(set_type='validation')
results_handler.generate_csv_aggregate_unique_walk(set_type='validation')


vgg.get_predictions_2D(set_type='test')
results_handler.generate_ensemble(set_type='test')
results_handler.generate_plots(set_type='test')
results_handler.generate_csv_aggregate_by_walk(set_type='test')
results_handler.generate_csv_aggregate_unique_walk(set_type='test')
