import matplotlib
matplotlib.use("Agg")

from classes.VggHandler import VggHandler
from classes.ResultsHandler import ResultsHandler
from classes.Market import Market

import matplotlib.pyplot as plt

import numpy as np
from keras import backend as K
from scipy.spatial.distance import cdist
from sklearn.metrics import confusion_matrix
#from classes.CustomLoss import pretty_loss

import os
import pandas as pd 
import argparse

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID";
os.environ["CUDA_VISIBLE_DEVICES"]="1";


parser = argparse.ArgumentParser(description='')
parser.add_argument('--start_index_walk', dest='start_index_walk', type=int, default=None, help='To restart the script starting by a specific walk')
arguments = parser.parse_args()

# CONFIGURATION ZONE 
#experiment_name = 'Exp 16 Walks 6mesi , SP500_CET, SGD BS 300, Labeling hold 0.3 (seba)'
#experiment_name = 'Exp 16 Walks 6mesi , SP500_CET, SGD BS 300, Labeling hold 0.3 (seba)'

loss_weight = w_array = np.ones((3,3))
loss_weight[2, 2] = 0
loss_weight[1, 2] = 2
loss_weight[0, 2] = 6

loss_weight[2, 1] = 2
loss_weight[1, 1] = 0
loss_weight[0, 1] = 2

loss_weight[2, 0] = 10
loss_weight[1, 0] = 4
loss_weight[0, 0] = 0
#pretty_loss(loss_weight)



training_set = [
                ['2000-01-01', '2009-12-31'],
                ['2000-01-01', '2010-12-31'],
                ['2000-01-01', '2011-12-31'],
                ['2000-01-01', '2012-12-31'],
                ['2000-01-01', '2013-12-31'],
                ['2000-01-01', '2014-12-31'],
                ['2000-01-01', '2015-12-31'],
                ['2000-01-01', '2016-12-31'],
                ['2000-01-01', '2017-12-31']
                ]

validation_set = [
                ['2010-01-01', '2010-12-31'],
                ['2011-01-01', '2011-12-31'],
                ['2012-01-01', '2012-12-31'],
                ['2013-01-01', '2013-12-31'],
                ['2014-01-01', '2014-12-31'],
                ['2015-01-01', '2015-12-31'],
                ['2016-01-01', '2016-12-31'],
                ['2017-01-01', '2017-12-31'],
                ['2018-01-01', '2018-12-31']

        ]

test_set = [
                ['2011-01-01', '2011-12-31'],
                ['2012-01-01', '2012-12-31'],
                ['2013-01-01', '2013-12-31'],
                ['2014-01-01', '2014-12-31'],
                ['2015-01-01', '2015-12-31'],
                ['2016-01-01', '2016-12-31'],
                ['2017-01-01', '2017-12-31'],
                ['2018-01-01', '2018-12-31'],
                ['2019-01-01', '2019-12-31']
        ]
       
iperparameters = { 
                'experiment_name': '101- Test multivariate SP500 OCHL',
                'epochs' : 500, # n° epoche per il training
                'number_of_nets': 30, # n° di reti da utilizzare per il training l'ensemble
                'save_pkl': False, # salva un file di history del training 
                'save_model_history': False, # booleano per salvare il modello ogni tot epoche
                'model_history_period': 3, # ogni quanto salvare un modello intermedio
                'bs': 192, # dimensione batch size
                'init_lr': 0.00001, # dimensione learning rate
                'return_multiplier': 50, #25 dax, moltiplicatore per convertire i punti di mercato in $
                'loss_function': 'w_categorical_crossentropy', # sparse_categorical_crossentropy | w_categorical_crossentropy
                'loss_weight': loss_weight, # pesi della custom loss
                'validation_thr': 15, # l'intorno da utilizzare per selezionare l'epoca migliore in validation
                'training_set': training_set,
                'validation_set': validation_set,
                'test_set': test_set,
                
                'input_datasets': ['sp500_cet'],
                'predictions_dataset': 'sp500_cet',
                'input_images_folders': ['merge/merge_sp500_cet_multivariate_ochl/gadf/multivariate/'],
                'predictions_images_folder': 'merge/merge_sp500_cet_multivariate_ochl/gadf/multivariate/',
                'input_shape': (40,40,3),
                'stop_loss': 1000, 
                'penalty': 25,
                'hold_labeling': 0.3,
                'use_probabilities': False,
                'verbose': True,
                'multivariate': False
                }

vgg = VggHandler(iperparameters=iperparameters)

vgg.run_2D(start_index_walk=arguments.start_index_walk, gpu_id=os.environ["CUDA_VISIBLE_DEVICES"])

## RESULTS ZONE ###
thr_ensemble_magg = [0.3, 0.33, 0.34, 0.35, 0.36, 0.37, 0.38, 0.39, 0.40, 0.41, 0.42, 0.43, 0.44, 0.45, 0.46, 0.47, 0.48, 0.49, 0.50]
thr_exclusive = [0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.80, 0.85, 0.90, 0.95, 1]
thr_elimination = [] #[0.35, 0.40, 0.45, 0.50, 0.55, 0.60]

rh = ResultsHandler(experiment_name=iperparameters['experiment_name'])
experiment_number = iperparameters['experiment_name'].split('-')[0]

# CALCOLA GLI AVG
rh.generate_json()

# calcolo il csv principale con le triple
rh.generate_triple_csv(remove_nets=False)
# calcolo gli ensemble per le varie soglie partendo dal file delle triple
rh.generate_ensemble(thrs_ensemble_magg=thr_ensemble_magg, thrs_ensemble_exclusive=thr_exclusive, thrs_ensemble_elimination=thr_elimination, remove_nets=False)

# CALCOLA CSV SELECTION
rh.calculate_epoch_selection()

# Per ogni soglia ensemble magg calcolo il file decisioni finale e poi gli excel per ogni report + excel swipe
for thr in thr_ensemble_magg:
        rh.get_final_decision_from_ensemble(type='ensemble_magg', validation_thr=15, validation_metric='romad', epoch_selection_policy='long_short', thr_ensemble_magg=thr, stop_loss=1000, penalty=25)
        rh.get_final_decision_from_ensemble(type='ensemble_magg', validation_thr=15, validation_metric='romad', epoch_selection_policy='long_only', thr_ensemble_magg=thr, stop_loss=1000, penalty=25)
        rh.get_final_decision_from_ensemble(type='ensemble_magg', validation_thr=15, validation_metric='romad', epoch_selection_policy='short_only', thr_ensemble_magg=thr, stop_loss=1000, penalty=25)
        rh.get_report_excel(type='ensemble_magg', epoch_selection_policy='long_short', thr=thr, stop_loss=1000, penalty=25, report_name=experiment_number + '- Rep Ens Magg - Sel. epoca Long Short - SL1000 Pen25 - thr ' + str(thr))
        rh.get_report_excel(type='ensemble_magg', epoch_selection_policy='long_only', thr=thr, stop_loss=1000, penalty=25, report_name=experiment_number + '- Rep Ens Magg - Sel. epoca Long Only - SL1000 Pen25 - thr ' + str(thr))
        rh.get_report_excel(type='ensemble_magg', epoch_selection_policy='short_only', thr=thr, stop_loss=1000, penalty=25, report_name=experiment_number + '- Rep Ens Magg - Sel. epoca Short Only - SL1000 Pen25 - thr ' + str(thr))

rh.get_report_excel_swipe(thrs_ensemble_magg=thr_ensemble_magg, epoch_selection_policy='long_short', type='ensemble_magg', stop_loss=1000, penalty=25,
        report_name=experiment_number + '- Rep Swipe Ens Magg - Sel. epoca Long Short - SL1000 Pen25')

rh.get_report_excel_swipe(thrs_ensemble_magg=thr_ensemble_magg, epoch_selection_policy='long_only', type='ensemble_magg', stop_loss=1000, penalty=25,
        report_name=experiment_number + '- Rep Swipe Ens Magg - Sel. epoca Long Only - SL1000 Pen25')

rh.get_report_excel_swipe(thrs_ensemble_magg=thr_ensemble_magg, epoch_selection_policy='short_only', type='ensemble_magg', stop_loss=1000, penalty=25,
        report_name=experiment_number + '- Rep Swipe Ens Magg - Sel. epoca Short Only - SL1000 Pen25')

# Per ogni soglia ensemble exclusive calcolo il file decisioni finale e poi gli excel per ogni report + excel swipe
for thr in thr_exclusive:
        rh.get_final_decision_from_ensemble(type='ensemble_exclusive', validation_thr=15, validation_metric='romad',
                epoch_selection_policy='long_only', thr_ensemble_exclusive=thr, stop_loss=1000, penalty=25)
        
        rh.get_final_decision_from_ensemble(type='ensemble_exclusive_short', validation_thr=15, validation_metric='romad',
                        epoch_selection_policy='short_only', thr_ensemble_exclusive=thr, stop_loss=1000, penalty=25)

        rh.get_report_excel(type='ensemble_exclusive', epoch_selection_policy='long_only', thr=thr, stop_loss=1000, penalty=25, 
                report_name=experiment_number + "- Rep Ens Excl - Sel. epoca Long Only - SL1000 Pen25 - thr " + str(thr))

        rh.get_report_excel(type='ensemble_exclusive_short', epoch_selection_policy='short_only', thr=thr, stop_loss=1000, penalty=25, 
                        report_name=experiment_number + "- Rep Ens Excl Short- Sel. epoca Short Only - SL1000 Pen25 - thr " + str(thr))

rh.get_report_excel_swipe(thrs_ensemble_exclusive=thr_exclusive, epoch_selection_policy='long_only', type='ensemble_exclusive', stop_loss=1000, penalty=25,
        report_name=experiment_number + '- Rep Swipe Ens Excl - Sel. epoca Long Only - SL1000 Pen25')

rh.get_report_excel_swipe(thrs_ensemble_exclusive=thr_exclusive, epoch_selection_policy='short_only', type='ensemble_exclusive_short', stop_loss=1000, penalty=25,
        report_name=experiment_number + '- Rep Swipe Ens Excl Short - Sel. epoca Short Only - SL1000 Pen25')


'''
## ZONA REPORT IS 1 - IS 2

# IS 1 - 2 - SP500
IS_1 = ['2009-08-02', '2014-01-31']
IS_2 = ['2014-02-02', '2017-01-31']
OOS = ['2017-02-01', '2019-10-30']

#IS_1 = ['2009-08-02', '2013-12-31']
#IS_2 = ['2014-01-02', '2016-12-30']


# ENSEMBLE MAGG
rh.get_report_excel_swipe(thrs_ensemble_magg=thr_ensemble_magg, epoch_selection_policy='long_short', type='ensemble_magg', stop_loss=1000, penalty=25, subfolder_is='Swipe IS 1', subsample=IS_1,
        report_name=experiment_number + '- Rep Swipe Ens Magg - IS 1 - Sel. epoca Long Short - SL1000 Pen25')

rh.get_report_excel_swipe(thrs_ensemble_magg=thr_ensemble_magg, epoch_selection_policy='long_only', type='ensemble_magg', stop_loss=1000, penalty=25, subfolder_is='Swipe IS 1', subsample=IS_1,
        report_name=experiment_number + '- Rep Swipe Ens Magg - IS 1 - Sel. epoca Long Only - SL1000 Pen25')

rh.get_report_excel_swipe(thrs_ensemble_magg=thr_ensemble_magg, epoch_selection_policy='short_only', type='ensemble_magg', stop_loss=1000, penalty=25, subfolder_is='Swipe IS 1', subsample=IS_1,
        report_name=experiment_number + '- Rep Swipe Ens Magg - IS 1 - Sel. epoca Short Only - SL1000 Pen25')

rh.get_report_excel_swipe(thrs_ensemble_magg=thr_ensemble_magg, epoch_selection_policy='long_short', type='ensemble_magg', stop_loss=1000, penalty=25, subfolder_is='Swipe IS 2', subsample=IS_2,
        report_name=experiment_number + '- Rep Swipe Ens Magg - IS 2 - Sel. epoca Long Short - SL1000 Pen25')

rh.get_report_excel_swipe(thrs_ensemble_magg=thr_ensemble_magg, epoch_selection_policy='long_only', type='ensemble_magg', stop_loss=1000, penalty=25, subfolder_is='Swipe IS 2', subsample=IS_2,
        report_name=experiment_number + '- Rep Swipe Ens Magg - IS 2 - Sel. epoca Long Only - SL1000 Pen25')

rh.get_report_excel_swipe(thrs_ensemble_magg=thr_ensemble_magg, epoch_selection_policy='short_only', type='ensemble_magg', stop_loss=1000, penalty=25, subfolder_is='Swipe IS 2', subsample=IS_2, 
        report_name=experiment_number + '- Rep Swipe Ens Magg - IS 2 - Sel. epoca Short Only - SL1000 Pen25')

# EXCLUSIVE
rh.get_report_excel_swipe(thrs_ensemble_exclusive=thr_exclusive, epoch_selection_policy='long_only', type='ensemble_exclusive', stop_loss=1000, penalty=25, subfolder_is='Swipe IS 1', subsample=IS_1,
        report_name=experiment_number + '- Rep Swipe Ens Excl - IS 1 - Sel. epoca Long Only - SL1000 P25')

rh.get_report_excel_swipe(thrs_ensemble_exclusive=thr_exclusive, epoch_selection_policy='long_only', type='ensemble_exclusive', stop_loss=1000, penalty=25, subfolder_is='Swipe IS 2', subsample=IS_2,
        report_name=experiment_number + '- Rep Swipe Ens Excl - IS 2 - Sel. epoca Long Only - SL1000 P25')


# OOS
rh.get_report_excel(type='ensemble_exclusive', epoch_selection_policy='long_only', thr=0.45, stop_loss=1000, penalty=25, subfolder_is='Report OOS', subsample=OOS,
        report_name=experiment_number + '- Rep Ens Excl - OOS - Sel. epoca Long Only - SL1000 Pen25 - thr ' + str(0.45))
'''