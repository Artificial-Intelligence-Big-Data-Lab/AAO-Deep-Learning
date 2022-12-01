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
loss_weight[0, 2] = 3

loss_weight[2, 1] = 2
loss_weight[1, 1] = 0
loss_weight[0, 1] = 2

loss_weight[2, 0] = 10
loss_weight[1, 0] = 4
loss_weight[0, 0] = 0

#pretty_loss(loss_weight)

training_set = [ ['1998-01-01', '2002-12-31'] ]
validation_set = [ ['2003-01-01', '2006-12-31'] ]
test_set = [ ['2007-01-01', '2012-12-31'] ]

iperparameters = { 
                'experiment_name': 'Applied Intelligence - IBM',
                'epochs' : 500, # n° epoche per il training
                'number_of_nets': 30, # n° di reti da utilizzare per il training l'ensemble
                'save_pkl': False, # salva un file di history del training 
                'save_model_history': False, # booleano per salvare il modello ogni tot epoche
                'model_history_period': 3, # ogni quanto salvare un modello intermedio
                'bs': 192, # dimensione batch size
                'init_lr': 0.00001, # dimensione learning rate
                'return_multiplier': 1, #25 dax, moltiplicatore per convertire i punti di mercato in $
                'loss_function': 'w_categorical_crossentropy', # sparse_categorical_crossentropy | w_categorical_crossentropy
                'loss_weight': loss_weight, # pesi della custom loss
                'validation_thr': 15, # l'intorno da utilizzare per selezionare l'epoca migliore in validation
                'training_set': training_set,
                'validation_set': validation_set,
                'test_set': test_set,
                
                'input_datasets': ['ibm'],
                'predictions_dataset': 'ibm',
                'input_images_folders': ['merge/merge_ibm/gadf/delta_current_day/'],
                'predictions_images_folder': 'merge/merge_ibm/gadf/delta_current_day/',
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