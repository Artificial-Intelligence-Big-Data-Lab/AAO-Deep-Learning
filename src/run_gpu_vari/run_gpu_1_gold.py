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

loss_weight = w_array = np.ones((3,3))
loss_weight[2, 2] = 0
loss_weight[1, 2] = 2
loss_weight[0, 2] = 6

loss_weight[2, 1] = 2
loss_weight[1, 1] = 0
loss_weight[0, 1] = 2

loss_weight[2, 0] = 6
loss_weight[1, 0] = 2
loss_weight[0, 0] = 0


training_set = [ ['2008-01-01', '2018-05-31'] ]
validation_set = [ ['2018-06-01', '2018-12-31'] ]
test_set = [ ['2019-01-01', '2019-06-30'] ]

#TMP__075 - gold and gold silver - pesi 27
iperparameters = { 
                'experiment_name': '074 - gold - pesi bilanciati controprova exp 73',
                'epochs' : 500, # n° epoche per il training
                'number_of_nets': 300, # n° di reti da utilizzare per il training l'ensemble
                'save_pkl': False, # salva un file di history del training 
                'save_model_history': False, # booleano per salvare il modello ogni tot epoche
                'model_history_period': 3, # ogni quanto salvare un modello intermedio
                'bs': 192, # dimensione batch size
                'init_lr': 0.000005, # dimensione learning rate
                'return_multiplier': 100, #25 dax, #100 gold, 50 sp500, moltiplicatore per convertire i punti di mercato in $
                'loss_function': 'w_categorical_crossentropy', # sparse_categorical_crossentropy | w_categorical_crossentropy
                'loss_weight': loss_weight, # pesi della custom loss
                'validation_thr': 15, # l'intorno da utilizzare per selezionare l'epoca migliore in validation
                'training_set': training_set,
                'validation_set': validation_set,
                'test_set': test_set,
                
                'input_datasets': ['gold_cet'],
                'predictions_dataset': 'gold_cet',
                'input_images_folders': ['merge/merge_gold_cet/gadf/delta_current_day/'],
                'predictions_images_folder': 'merge/merge_gold_cet/gadf/delta_current_day/',
                'input_shape': (40,40,3),
                'stop_loss': 1000, 
                'penalty': 25,
                'hold_labeling': 0.3,
                'use_probabilities': False,
                'verbose': True
            }

vgg = VggHandler(iperparameters=iperparameters)

vgg.run_2D(start_index_walk=arguments.start_index_walk, gpu_id=os.environ["CUDA_VISIBLE_DEVICES"])

## RESULTS ZONE ###
thr_ensemble_magg = [0.33, 0.34, 0.35, 0.36, 0.37, 0.38, 0.39, 0.40, 0.41, 0.42, 0.43, 0.44, 0.45, 0.46, 0.47, 0.48, 0.49, 0.50]
thr_exclusive = [0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60]
thr_elimination = [] #[0.35, 0.40, 0.45, 0.50, 0.55, 0.60]

rh = ResultsHandler(experiment_name=iperparameters['experiment_name'])
experiment_number = iperparameters['experiment_name'].split('-')[0]

rh.generate_json()
rh.calculate_epoch_selection(test_penalty=30, test_stop_loss=1000, long_min_thr=5, long_step=5, short_min_thr=10, short_step=10)
# calcolo il csv principale con le triple
rh.generate_triple_csv(remove_nets=False)
# calcolo gli ensemble per le varie soglie partendo dal file delle triple
rh.generate_ensemble(thrs_ensemble_magg=thr_ensemble_magg, thrs_ensemble_exclusive=thr_exclusive, thrs_ensemble_elimination=thr_elimination, remove_nets=False)


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
        rh.get_report_excel(type='ensemble_exclusive', epoch_selection_policy='long_only', thr=thr, stop_loss=1000, penalty=25, 
                report_name=experiment_number + "- Rep Ens Excl - Sel. epoca Long Only - SL1000 Pen25 - thr " + str(thr))

rh.get_report_excel_swipe(thrs_ensemble_exclusive=thr_exclusive, epoch_selection_policy='long_only', type='ensemble_exclusive', stop_loss=1000, penalty=25,
        report_name=experiment_number + '- Rep Swipe Ens Excl - Sel. epoca Long Only - SL1000 Pen25')
