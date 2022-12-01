import matplotlib
matplotlib.use("Agg")

from classes.VggBinaryHandler import VggBinaryHandler
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




#experiment_name = '001 - SP500 walk 1 anno sino al 2020 sp500 e vix BINARY'
#experiment_name = '002 - SP500 walk 1 anno sino al 2020 sp500 e vix BINARY'
#experiment_name = '006 - SP500 walk 2 anni sino al 2020 sp500 e vix BINARY PESI Bilanciati'
#experiment_path = 'D:/PhD-Market-Nets/experiments-binary/'
experiment_path = '/media/unica/HDD 9TB Raid0 - 1/experiments-binary/'

experiments = [
    #'001 - SP500 walk 1 anno sino al 2020 sp500 e vix BINARY', 
    #'002 - SP500 walk 1 anno sino al 2020 sp500 e vix BINARY', 
    #'004 - SP500 walk 2 anni sino al 2020 sp500 e vix BINARY PESI IDLE',
    '005 - SP500 walk 2 anno sino al 2020 sp500 e vix BINARY con Pesi LONG', 
    '006 - SP500 walk 2 anni sino al 2020 sp500 e vix BINARY PESI Bilanciati', 
    '007 - SP500 walk 2 anno sino al 2020 sp500 e vix BINARY con Pesi IDLE 2', 
    '008 - SP500 walk 2 anno sino al 2020 sp500 e vix BINARY con Pesi IDLE 3', 
    '009 - SP500 walk 2 anno sino al 2020 sp500 e vix BINARY con Pesi IDLE 4', 
    '010 - SP500 walk 2 anno sino al 2020 sp500 e vix BINARY con Pesi IDLE 5', 
    '011 - SP500 walk 2 anno sino al 2020 sp500 e vix BINARY con Pesi IDLE 6',
    '012 - SP500 walk 2 anno sino al 2020 sp500 e vix BINARY con Pesi LONG 2', # seconda run terminale
    '013 - SP500 walk 2 anno sino al 2020 sp500 e vix BINARY con Pesi LONG 3',
    '014 - SP500 walk 2 anno sino al 2020 sp500 e vix BINARY con Pesi LONG 4',
    '015 - SP500 walk 2 anno sino al 2020 sp500 e vix BINARY con Pesi LONG 5', 
    '016 - SP500 walk 2 anno sino al 2020 sp500 e vix BINARY con Pesi Bilanciati 2', # terza run
    '017 - SP500 walk 2 anno sino al 2020 sp500 e vix BINARY con Bilanciati 3',
]
for experiment in experiments: 
    print("Lancio generazione risultati per", experiment)
    generate_results(experiment_name=experiment, experiment_path=experiment_path, single_net=True)
