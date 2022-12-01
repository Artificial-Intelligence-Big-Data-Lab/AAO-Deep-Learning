from classes.VggHandler import VggHandler
from classes.ResultsHandler import ResultsHandler
from classes.Market import Market

# import os
# os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID";
# os.environ["CUDA_VISIBLE_DEVICES"]="1";

training_set = [['2000-02-01', '2016-07-31'], ['2000-08-01', '2017-01-31'] ]

validation_set = [['2016-08-01', '2017-01-31'], ['2017-02-01', '2017-07-31']]

test_set = [['2017-02-01', '2017-07-31'], ['2017-08-01', '2018-01-31']]


vgg = VggHandler()

vgg.net_config(epochs=5, number_of_nets=20)

vgg.run_initialize(dataset='sp500',
        input_folder='delta_experiment/gadf/delta/',
        training_set=training_set,
        validation_set=validation_set,
        test_set=test_set,
        input_shape=(40,40,3),
        output_folder='experiments_saving_each_epochs')


vgg.run()


'''
results_handler = ResultsHandler(experiment_name='experiments_saving_each_epochs', dataset='sp500') #delta_experiment_third_run

#vgg.get_predictions(set_type='training')
results_handler.generate_ensemble_and_plots(set_type='training')
results_handler.calculate_return(set_type='training')

#vgg.get_predictions(set_type='validation')
results_handler.generate_ensemble_and_plots(set_type='validation')
results_handler.calculate_return(set_type='validation')


#vgg.get_predictions(set_type='test')
results_handler.generate_ensemble_and_plots(set_type='test')
results_handler.calculate_return(set_type='test')
'''