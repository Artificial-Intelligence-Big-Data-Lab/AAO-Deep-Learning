import matplotlib
matplotlib.use("Agg")

#from classes.VggHandler import VggHandler
from classes.VggBinaryHandler import VggBinaryHandler
from classes.ClassifierHandler import ClassifierHandler
from classes.ResultsHandler import ResultsHandler
from classes.BinaryTrading import BinaryTrading
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
loss_weight = w_array = np.ones((2,2))
loss_weight[0, 1] = 1
loss_weight[1, 1] = 0
loss_weight[0, 0] = 0
loss_weight[1, 0] = 1


#pretty_loss(loss_weight)

training_set = [
        ['2011-01-01', '2015-12-31']  
 ]

validation_set = [
        ['2016-01-01', '2016-12-31']

 ]

test_set = [
        ['2017-01-01', '2017-12-31']
]

iperparameters = { 
                'experiment_name': '53 - Lexicons IBM - IMG Grayscale - High Volatility 5 - 7 day prediction - lr 0.01 - superlexico filtrando con delta 2perc news dna',
                'epochs' : 300, # n° epoche per il training
                'number_of_nets': 5, # n° di reti da utilizzare per il training l'ensemble
                'save_pkl': False, # salva un file di history del training 
                'save_model_history': False, # booleano per salvare il modello ogni tot epoche
                'model_history_period': 3, # ogni quanto salvare un modello intermedio
                'bs': 128, # dimensione batch size
                'init_lr': 0.01 , # dimensione learning rate
                'return_multiplier': 50, #25 dax, moltiplicatore per convertire i punti di mercato in $
                'loss_function': 'sparse_categorical_crossentropy', # sparse_categorical_crossentropy | w_categorical_crossentropy
                'loss_weight': loss_weight, # pesi della custom loss
                'validation_thr': 15, # l'intorno da utilizzare per selezionare l'epoca migliore in validation
                'training_set': training_set,
                'validation_set': validation_set,
                'test_set': test_set,
                'input_datasets': ['ibm_cet'],
                'predictions_dataset': 'ibm_cet',
                'input_images_folders': ['lexicons/IBM/for_cnn_model_filter_256_256_grayscale_filtro_delta_news'],
                'predictions_images_folder': 'lexicons/IBM/for_cnn_model_filter_256_256_grayscale_filtro_delta_news',
                #'input_images_folders': ['lexicons/for_cnn_model_filter_256_256_opacity/'],
                #'predictions_images_folder': 'lexicons/for_cnn_model_filter_256_256_opacity/',

                'input_shape': (256, 256, 3),
                'stop_loss': 1000, 
                'penalty': 25,
                'thr_binary_labeling': 5, #binary
                'is_binary': True,
                'hold_labeling': 0.3,
                'use_probabilities': False,
                'verbose': True,
                'multivariate': False,
                'balance_binary': True,
                'experiment_path': '/media/unica/HDD 9TB Raid0 - 1/experiments-binary-lexiconcs/', 
                #'experiment_path': 'D:/PhD-Market-Nets/experiments-binary-lexiconcs/',
                'description': '<p></p>'
}

vgg = VggBinaryHandler(iperparameters=iperparameters)
#vgg = VggHandler(iperparameters=iperparameters)
vgg.run_2D(start_index_walk=arguments.start_index_walk, gpu_id=os.environ["CUDA_VISIBLE_DEVICES"])
#vgg.run_1D(dataset='sp500_cet_WITH_PATTERN_FORTE', start_index_walk=arguments.start_index_walk, gpu_id=os.environ["CUDA_VISIBLE_DEVICES"])


if 'experiment_path' in iperparameters: 
        generate_results(experiment_name=iperparameters['experiment_name'], experiment_path=iperparameters['experiment_path'], single_net=True)
else: 
        generate_results(experiment_name=iperparameters['experiment_name'], single_net=False)  


#bt = BinaryTrading(experiment_name=iperparameters['experiment_name'], experiment_path=iperparameters['experiment_path'])
#bt.run()