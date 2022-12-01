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

training_set = [['2000-01-01', '2010-02-28'], 
    ['2000-01-01', '2010-12-31'], 
    ['2000-01-01', '2011-10-31']
 ]

validation_set = [['2010-03-01', '2010-12-31'], 
    ['2011-01-01', '2011-10-31'], 
    ['2011-11-01', '2012-08-31']
 ]

test_set = [['2011-01-01', '2011-10-31'], 
    ['2011-11-01', '2012-08-31'], 
    ['2012-09-01', '2013-06-30']
]


iperparameters = { 
                'experiment_name': '00 - Debug new code TMP',
                'epochs' : 100, # n° epoche per il training
                'number_of_nets': 10, # n° di reti da utilizzare per il training l'ensemble
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
                'input_images_folders': ['sp500_cet/1hour/gadf/delta_current_day/'],
                'predictions_images_folder': 'sp500_cet/1hour/gadf/delta_current_day/',
                'input_shape': (20,20,3),
                'stop_loss': 1000, 
                'penalty': 25,
                'hold_labeling': 0.3,
                'use_probabilities': False,
                'verbose': True,
                'multivariate': False
                }

vgg = VggHandler(iperparameters=iperparameters)

#vgg.run_2D(start_index_walk=arguments.start_index_walk, gpu_id=os.environ["CUDA_VISIBLE_DEVICES"])

generate_results(experiment_name=iperparameters['experiment_name'], single_net=True)