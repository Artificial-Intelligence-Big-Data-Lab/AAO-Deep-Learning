import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import sys
import cv2
import time
import pickle
import datetime
import matplotlib
matplotlib.use("Agg")
import numpy as np
import pandas as pd
from datetime import timedelta

#matplotlib.use("Agg")
import matplotlib.pyplot as plt

from classes.Market import Market
from classes.StopLossCustom import stop_loss_custom

from cycler import cycler

from keras.models import load_model
from vgg16.SmallerVGGNet import SmallerVGGNet
from sklearn.preprocessing import LabelBinarizer, MultiLabelBinarizer
from keras.optimizers import Adam, RMSprop, SGD, Nadam
from keras_radam import RAdam
from keras import backend as K
from keras.callbacks import ModelCheckpoint
from keras.callbacks import Callback
import tensorflow as tf
import sklearn.metrics as sklm  
from sklearn.metrics import confusion_matrix

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID";
os.environ["CUDA_VISIBLE_DEVICES"]="1";

model_path = '/home/unica/PhD-Market-Nets/experiments/exp_BH_walk_1_mese_SGD_BS_500_SP500_thr0.3/models/'
model_filename = 'walk_32_net_1.model'
model_filename = 'walk_32_net_6.model'
model_filename = 'walk_32_net_10.model'

model = load_model(model_path + model_filename)

# walk 32
image_name = '2013-02-22.png'
x = np.zeros(shape=(1, 40, 40, 3), dtype=np.uint8)

x[0] = cv2.imread('/home/unica/PhD-Market-Nets/images/merge/merge_sp500_cet/gadf/delta/' + image_name)

preds = model.predict_classes(x)

print(preds)