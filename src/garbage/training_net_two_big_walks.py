import os 
from classes.VggHandler import VggHandler
from classes.ResultsHandler import ResultsHandler
from classes.Market import Market

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID";
os.environ["CUDA_VISIBLE_DEVICES"]="0";

training_set = [['2000-02-01', '2013-12-31'], ['2001-06-01', '2015-5-31']]
validation_set = [ ['2014-01-01', '2014-12-31'], ['2015-06-01', '2016-05-31']]
test_set = [['2015-01-01', '2016-05-31'], ['2016-06-01', '2017-12-31'] ]

training_set = [['2000-02-01', '2013-12-31']]
validation_set = [ ['2014-01-01', '2014-12-31']]
test_set = [['2015-01-01', '2016-05-31']]


#training_set = [['2000-02-01', '2013-12-31']]
#validation_set = [ ['2014-01-01', '2014-12-31']]
#test_set = [['2015-01-01', '2016-05-31']]


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


predictions_dataset = 'sp500'
predictions_images_folder = 'merge_sp500/gadf/delta/'

vgg = VggHandler()

vgg.net_config(epochs=2000, number_of_nets=20, save_pkl=False, save_model_history=True, model_history_period=50)

vgg.run_initialize( predictions_dataset=predictions_dataset,
                    predictions_images_folder=predictions_images_folder,

                    input_images_folders=input_images_folders,
                    input_datasets=input_datasets,

                    training_set=training_set,
                    validation_set=validation_set,
                    test_set=test_set,
                    
                    input_shape=(40,40,3),
                    output_folder='multi_company_exp_after_fix_third_run')

vgg.run()

#vgg.run_again(model_input_folder='multi_company_exp_after_fix')

#results_handler = ResultsHandler(experiment_name='multi_company_exp_after_fix', dataset='sp500')


#vgg.get_predictions(set_type='validation')
#results_handler.generate_ensemble_and_plots(set_type='validation')
#results_handler.calculate_return(set_type='validation')


#vgg.get_predictions(set_type='test')
#results_handler.generate_ensemble_and_plots(set_type='test')
#results_handler.calculate_return(set_type='test')
