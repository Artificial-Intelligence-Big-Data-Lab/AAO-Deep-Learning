import matplotlib
matplotlib.use("Agg")

from classes.VggHandler import VggHandler
from classes.ResultsHandler import ResultsHandler
from classes.Market import Market
from classes.Helper import generate_results

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
os.environ["CUDA_VISIBLE_DEVICES"]="0";


parser = argparse.ArgumentParser(description='')
parser.add_argument('--start_index_walk', dest='start_index_walk', type=int, default=None, help='To restart the script starting by a specific walk')
arguments = parser.parse_args()

# CONFIGURATION ZONE 
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

training_set = [['2000-01-01', '2009-12-31'], 
        ['2000-01-01', '2010-12-31'], 
        ['2000-01-01', '2011-12-31'], 
        ['2000-01-01', '2012-12-31'], 
        ['2000-01-01', '2013-12-31'], 
        ['2000-01-01', '2014-12-31'], 
        ['2000-01-01', '2015-12-31'], 
        ['2000-01-01', '2016-12-31'], 
        ['2000-01-01', '2017-12-31'], 
        ['2000-01-01', '2018-12-31'] 
 ]

validation_set = [['2010-01-01', '2010-12-31'], 
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

test_set = [['2011-01-01', '2011-12-31'], 
        ['2012-01-01', '2012-12-31'], 
        ['2013-01-01', '2013-12-31'], 
        ['2014-01-01', '2014-12-31'], 
        ['2015-01-01', '2015-12-31'], 
        ['2016-01-01', '2016-12-31'], 
        ['2017-01-01', '2017-12-31'], 
        ['2018-01-01', '2018-12-31'], 
        ['2019-01-01', '2019-12-31'], 
        ['2020-01-01', '2020-07-31']
]

iperparameters = { 
                'experiment_name': '128 - SP500 walk 1 anno sino al 2020 con nuovi dati.',
                'epochs' : 500, # n° epoche per il training
                'number_of_nets': 100, # n° di reti da utilizzare per il training l'ensemble
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
                'input_images_folders': ['merge/merge_sp500_vix_skwe_multivariate/gadf/multivariate/'],
                'predictions_images_folder': 'merge/merge_sp500_vix_skwe_multivariate/gadf/multivariate/',
                'input_shape': (40,40,3),
                'input_shape_single_signal': (20, 20), # only multivariate
                'stop_loss': 1000, 
                'penalty': 25,
                'hold_labeling': 0.3,
                'use_probabilities': False,
                'verbose': True,
                'multivariate': True,
                'description': '<p>Primo esperimento utilizzando più fonti di dati con la rete multivariate.</p>\
<p>Vengono usate le seguenti img 20x20: 1h  nel blocco superiore sinistro e 8h nel blocco superiore destro di sp500.<br>\
Vix daily nel blocco inferiore sinistro e skew daily nel blocco inferiore destro.</p>'
                }

#vgg = VggHandler(iperparameters=iperparameters)

#vgg.run_2D(start_index_walk=arguments.start_index_walk, gpu_id=os.environ["CUDA_VISIBLE_DEVICES"])

generate_results(experiment_name=iperparameters['experiment_name'], single_net=True)