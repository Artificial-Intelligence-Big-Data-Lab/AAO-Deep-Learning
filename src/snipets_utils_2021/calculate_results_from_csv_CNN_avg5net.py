import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import sys
import cv2
import time
import pickle
import datetime
import functools
import numpy as np
import pandas as pd
from datetime import timedelta
from classes.Market import Market
from classes.Metrics import Metrics
from classes.Set import Set
from classes.Measures import Measures
from classes.Utils import create_folder
from cycler import cycler
import sklearn.metrics as sklm  
import json
import platform
import statistics

# classification moment
from sklearn import model_selection
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, balanced_accuracy_score
from classes.Utils import df_date_merger_binary

### CALCOLO LA BALANCED ACCURACY E PRECISION DAI CSV DELLE PREDIZIONI DELLE CNN


def print_cm(cm, labels, hide_zeroes=False, hide_diagonal=False, hide_threshold=None):
        columnwidth = max([len(x) for x in labels] + [5])  # 5 is value length
        empty_cell = " " * columnwidth
        
        # Begin CHANGES
        fst_empty_cell = (columnwidth-3)//2 * " " + "t/p" + (columnwidth-3)//2 * " "
        
        if len(fst_empty_cell) < len(empty_cell):
            fst_empty_cell = " " * (len(empty_cell) - len(fst_empty_cell)) + fst_empty_cell
        # Print header
        print("    " + fst_empty_cell, end=" ")
        # End CHANGES
        
        for label in labels:
            print("%{0}s".format(columnwidth) % label, end=" ")
            
        print()
        # Print rows
        for i, label1 in enumerate(labels):
            print("    %{0}s".format(columnwidth) % label1, end=" ")
            for j in range(len(labels)):
                cell = "%{0}.3f".format(columnwidth) % cm[i, j]
                if hide_zeroes:
                    cell = cell if float(cm[i, j]) != 0 else empty_cell
                if hide_diagonal:
                    cell = cell if i != j else empty_cell
                if hide_threshold:
                    cell = cell if cm[i, j] > hide_threshold else empty_cell
                print(cell, end=" ")
            print()
    
experiments = [
    #'001 - 20Sample 1h - BS 128 LR 0.0001 SGD',
    #'002 - 96Sample 5min -  BS 128 LR 0.0001 SGD', 
    #'003 - 20Sample 1h - BS 128 LR 0.00001 SGD' #v2
    #'004 - 96Sample 5min -  BS 128 LR 0.00001 SGD' #v2
    #'005 - 20Sample 1h - BS 128 LR 0.001 SGD', #v3
    #'007 - 96Sample 5min -  BS 128 LR 0.001 SGD' #v3
    #'006 - 20Sample 1h - BS 128 LR 0.0005 SGD' #v4
    '008 - 96Sample 5min -  BS 128 LR 0.0005 SGD' #v4
]

nets = ['0', '1', '2', '3', '4']
walks = ['walk_0', 'walk_1', 'walk_2']

delta_precision_down = []
label_precision_down = []
balanced_accuracies = []
delta_avg_precision = []
label_avg_precision = []
perc_operazioni = []
num_operazioni = []


market = Market(dataset='sp500_cet')
market = market.get_binary_labels(freq='1d', columns=['delta_current_day', 'delta_next_day', 'close'], thr=-0.5).reset_index()


for experiment_name in experiments: 
    for walk in walks: 
        print(experiment_name, walk)
        print("Net\t\t Delta Down %\t\t Label Down %\t\t Balanced Accuracy %\t\t Delta AVG Precision\t\t Label Avg Precision\t\tOperazioni % / N?? operazioni")

        for net in nets:
            
            df = pd.read_csv('D:/PhD-Market-Nets/experiments-binary-a-50/' + experiment_name + '/predictions/predictions_during_training/test/' + walk + '/net_' + net + '.csv')

            df = df[['date_time', 'epoch_1000']]
            df = df.rename(columns={"epoch_1000": "decision"})

            df['date_time'] = df['date_time'].shift(-1)
            df = df.dropna()

            df = df_date_merger_binary(df=df.copy(), thr=-0.5, columns=['delta_current_day', 'delta_next_day', 'open', 'close', 'high', 'low'], dataset='sp500_cet')
            
            results = Measures.get_equity_return_mdd_romad(df=df.copy(), multiplier=50, type='long_only', penalty=25, stop_loss=1000, delta_to_use='delta_current_day', compact_results=True)
            
            label_precisions = confusion_matrix(df['label_current_day'].tolist(), df['decision'].tolist(), normalize='pred', labels = [0, 1])
            label_random_precision = Measures.get_binary_coverage(y=df['label_current_day'].tolist())

            coverage = Measures.get_binary_coverage(y=df['decision'].tolist())

            # Precision delta + random delta
            delta_precision = Measures.get_binary_delta_precision(y=df['decision'].tolist(), delta=df['delta_current_day'].tolist(), delta_val=-25)

            balanced_accuracy = balanced_accuracy_score(df['label_current_day'].tolist(), df['decision'].tolist())

            delta_precision_down.append(delta_precision['down'])
            label_precision_down.append(label_precisions[0][0] * 100)
            balanced_accuracies.append(balanced_accuracy * 100)
            delta_avg_precision.append((delta_precision['down'] + delta_precision['up']) / 2)
            label_avg_precision.append(((label_precisions[0][0] + label_precisions[1][1]) / 2) * 100)
            perc_operazioni.append(coverage['down_perc'])
            num_operazioni.append(coverage['down_count'])

            print(  net, "\t\t",
                    round(delta_precision['down'], 2), "/", round(delta_precision['random_down'], 2), "\t\t",
                    round(label_precisions[0][0] * 100, 2), "/", round(label_random_precision['down_perc'], 2), "\t\t",
                    round(balanced_accuracy * 100, 2), "\t\t\t\t", 
                    #round(((delta_precision['down'] / delta_precision['random_down']) - 1 ) * 100, 2), "\t\t",
                    #round((((label_precisions[0][0] * 100) / label_random_precision['down_perc']) - 1 ) * 100, 2), "\t\t",
                    round((delta_precision['down'] + delta_precision['up']) / 2, 2), "\t\t\t\t",
                    round(((label_precisions[0][0] + label_precisions[1][1]) / 2) * 100, 2), "\t\t\t\t",
                    coverage['down_perc'], "/", coverage['down_count'],
            )

        print(  "Avg", "\t\t",
                round(statistics.mean(delta_precision_down), 2), "/", round(delta_precision['random_down'], 2), "\t\t",
                round(statistics.mean(label_precision_down), 2), "/", round(label_random_precision['down_perc'], 2), "\t\t",
                round(statistics.mean(balanced_accuracies), 2), "\t\t\t\t", 
                round(statistics.mean(delta_avg_precision), 2), "\t\t\t\t",
                round(statistics.mean(label_avg_precision), 2), "\t\t\t\t",
                statistics.mean(perc_operazioni), "/", statistics.mean(num_operazioni),
            )
    print("\n")