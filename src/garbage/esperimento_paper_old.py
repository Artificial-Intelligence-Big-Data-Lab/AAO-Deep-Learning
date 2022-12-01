import matplotlib
matplotlib.use("Agg")

from classes.VggHandler import VggHandler
from classes.ResultsHandler import ResultsHandler
from classes.Market import Market

import numpy as np
from keras import backend as K
from scipy.spatial.distance import cdist

import os
#os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID";
#os.environ["CUDA_VISIBLE_DEVICES"]="0";


# FIRST
#training_set = [['2012-10-20', '2016-12-31']]
#validation_set = [ ['2017-01-01', '2017-09-20']]
#test_set = [['2017-09-21', '2017-11-30']]

#training_set =      [['2003-08-01', '2012-07-31'],  ['2004-02-01', '2013-01-31'],  ['2004-08-01', '2013-07-31'],  ['2005-02-01', '2014-01-31']]
#validation_set =    [['2012-08-01', '2013-01-31'],  ['2013-02-01', '2013-07-31'],  ['2013-08-01', '2014-01-31'],  ['2014-02-02', '2014-07-31']]
#test_set =          [['2013-02-01', '2013-07-31'],  ['2013-08-01', '2014-01-31'],  ['2014-02-02', '2014-07-31'],  ['2014-08-01', '2015-01-30']]

training_set =      [['2003-08-01', '2012-07-31']]
validation_set =    [['2012-08-01', '2013-01-31']]
test_set =          [['2013-02-01', '2013-07-31']]


#input_images_folders = ['sp500/1day/gadf/delta/']



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
''

######## 200x200 ############
input_images_folders = ['merge/merge_sp500/gadf/delta/']
input_datasets = ['sp500']
# The market you want to predict
predictions_dataset = 'sp500'
#predictions_images_folder = 'sp500/1day/gadf/delta/'
predictions_images_folder = 'merge/merge_sp500/gadf/delta/'
experiment_name = 'exp_3classes_small_net_sp500_test_loss_3'

vgg = VggHandler()

# 2791 samples 1/2 batch 1395  # 1/3 batch 931 |187 1/15 
vgg.net_config(epochs=50, number_of_nets=1, save_pkl=False, save_model_history=True, model_history_period=5, bs=500,init_lr=0.001) 

vgg.run_initialize( predictions_dataset=predictions_dataset,
                    predictions_images_folder=predictions_images_folder,

                    input_images_folders=input_images_folders,
                    input_datasets=input_datasets,

                    training_set=training_set,
                    validation_set=validation_set,
                    test_set=test_set,
                    
                    input_shape=(40,40,3),
                    output_folder=experiment_name)

vgg.run_2D()

'''
def loss_function(y_true,y_pred, weights):

    sess = K.get_session()
    print(sess)

    y_true = y_true.eval(session=sess)
    y_pred = y_pred.eval(session=sess)

    print(y_true)

    loss = np.zeros(y_true.shape[1])

    for i in range(0, y_true.shape[1]):
        loss[i] = np.abs(np.sum((y_true[i] - y_pred[i]) * np.array([0,1,2]))) * weights[i]

    mean_loss = np.sum(loss)

    return mean_loss

def my_function(y_true,y_pred, weights):
    return K.sum(K.log(y_true))

x = np.zeros((3,3))
y = np.zeros((3,3))
w = np.ones(3)

x[0][2] = 1
x[1][1] = 1
x[2][1] = 1

#print(x)

y[0][0] = 1
y[1][1] = 1
y[2][0] = 1

w[1] = 2
w[2] = 5

x = K.constant(x)
y = K.constant(y)


abc = my_function(x, y, w)

print(abc)
print(K.eval(abc))



results_handler = ResultsHandler(experiment_name=experiment_name, dataset='sp500')

vgg.get_predictions_2D(set_type='validation')
results_handler.generate_ensemble(set_type='validation')
results_handler.generate_plots(set_type='validation')
results_handler.generate_csv_aggregate_by_walk(set_type='validation')

vgg.get_predictions_2D(set_type='test')
results_handler.generate_ensemble(set_type='test')
results_handler.generate_plots(set_type='test')
results_handler.generate_csv_aggregate_by_walk(set_type='test')
'''


'''

######## 160x160 ############
vgg = VggHandler()
input_images_folders = ['merge/merge_160x160_sp500/gadf/delta/']
input_datasets = ['sp500']
# The market you want to predict
predictions_dataset = 'sp500'
#predictions_images_folder = 'sp500/1day/gadf/delta/'
predictions_images_folder = 'merge/merge_160x160_sp500/gadf/delta/'
experiment_name = 'exp_v2_anse_sp500_MULTI_RES_160x160_daily_radam'

vgg = VggHandler()

# 2791 samples 1/2 batch 1395  # 1/3 batch 931 |187 1/15 
vgg.net_config(epochs=500, number_of_nets=10, save_pkl=False, save_model_history=True, model_history_period=500, bs=187,init_lr=0.0001) 

vgg.run_initialize( predictions_dataset=predictions_dataset,
                    predictions_images_folder=predictions_images_folder,

                    input_images_folders=input_images_folders,
                    input_datasets=input_datasets,

                    training_set=training_set,
                    validation_set=validation_set,
                    test_set=test_set,
                    
                    input_shape=(160,160,3),
                    output_folder=experiment_name)

vgg.run_2D()


######## 120x120 ############
vgg = VggHandler()
input_images_folders = ['merge/merge_120x120_sp500/gadf/delta/']
input_datasets = ['sp500']
# The market you want to predict
predictions_dataset = 'sp500'
#predictions_images_folder = 'sp500/1day/gadf/delta/'
predictions_images_folder = 'merge/merge_120x120_sp500/gadf/delta/'
experiment_name = 'exp_v2_anse_sp500_MULTI_RES_120x120_daily_radam'

vgg = VggHandler()

# 2791 samples 1/2 batch 1395  # 1/3 batch 931 |187 1/15 
vgg.net_config(epochs=500, number_of_nets=10, save_pkl=False, save_model_history=True, model_history_period=500, bs=187,init_lr=0.0001) 

vgg.run_initialize( predictions_dataset=predictions_dataset,
                    predictions_images_folder=predictions_images_folder,

                    input_images_folders=input_images_folders,
                    input_datasets=input_datasets,

                    training_set=training_set,
                    validation_set=validation_set,
                    test_set=test_set,
                    
                    input_shape=(120,120,3),
                    output_folder=experiment_name)

vgg.run_2D()


######## 80x80 ############
vgg = VggHandler()
input_images_folders = ['merge/merge_80x80_sp500/gadf/delta/']
input_datasets = ['sp500']
# The market you want to predict
predictions_dataset = 'sp500'
#predictions_images_folder = 'sp500/1day/gadf/delta/'
predictions_images_folder = 'merge/merge_80x80_sp500/gadf/delta/'
experiment_name = 'exp_v2_anse_sp500_MULTI_RES_80x80_daily_radam'

vgg = VggHandler()

# 2791 samples 1/2 batch 1395  # 1/3 batch 931 |187 1/15 
vgg.net_config(epochs=500, number_of_nets=10, save_pkl=False, save_model_history=True, model_history_period=500, bs=187,init_lr=0.0001) 

vgg.run_initialize( predictions_dataset=predictions_dataset,
                    predictions_images_folder=predictions_images_folder,

                    input_images_folders=input_images_folders,
                    input_datasets=input_datasets,

                    training_set=training_set,
                    validation_set=validation_set,
                    test_set=test_set,
                    
                    input_shape=(80,80,3),
                    output_folder=experiment_name)

vgg.run_2D()
'''





'''
results_handler = ResultsHandler(experiment_name=experiment_name, dataset='sp500')

vgg.get_predictions_2D(set_type='validation')
results_handler.generate_ensemble(set_type='validation')
results_handler.generate_plots(set_type='validation')
results_handler.generate_csv_aggregate_by_walk(set_type='validation')

vgg.get_predictions_2D(set_type='test')
results_handler.generate_ensemble(set_type='test')
results_handler.generate_plots(set_type='test')
results_handler.generate_csv_aggregate_by_walk(set_type='test')
'''


'''
' 1D STUFFS
''
experiment_name = 'exp_paper_1D'
vgg = VggHandler()
vgg.net_config(epochs=200, number_of_nets=20, save_pkl=False, save_model_history=True, model_history_period=200)
vgg.run_initialize(predictions_dataset=predictions_dataset,
                    predictions_images_folder=predictions_images_folder,

                    input_images_folders=input_images_folders,
                    input_datasets=input_datasets,

                    training_set=training_set,
                    validation_set=validation_set,
                    test_set=test_set,
                    
                    input_shape=(40,40,3),
                    output_folder=experiment_name) #delta_experiment_third_run


#vgg.run_1D()

results_handler = ResultsHandler(experiment_name=experiment_name, dataset='sp500')

vgg.get_predictions_1d(set_type='validation')
results_handler.generate_ensemble(set_type='validation')
results_handler.generate_plots(set_type='validation')
results_handler.generate_csv_aggregate_by_walk(set_type='validation')
results_handler.generate_csv_aggregate_unique_walk(set_type='validation')


vgg.get_predictions_1d(set_type='test')
results_handler.generate_ensemble(set_type='test')
results_handler.generate_plots(set_type='test')
results_handler.generate_csv_aggregate_by_walk(set_type='test')
results_handler.generate_csv_aggregate_unique_walk(set_type='test')
'''