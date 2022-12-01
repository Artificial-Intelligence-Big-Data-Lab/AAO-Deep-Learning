from classes.Market import Market
from classes.Measures import Measures
from classes.Utils import df_date_merger
import os
import json

import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score
from cycler import cycler

experiment_name = '005 - pesiloss5 - 16gennaio'
index_walk = 0
net = 0
type = 'validation'
MULTIPLIER = 50 


df = pd.read_csv('../experiments/' + experiment_name + '/predictions/predictions_during_training/' + type + '/walk_' + str(index_walk) + '/net_' + str(net)  + '.csv')

# mergio con le label, così ho un subset del df con le date che mi servono e la predizione 
df_merge_with_label = df_date_merger(df=df, columns=['date_time', 'delta_next_day', 'close', 'open'], dataset='sp500_cet')


y_pred = df_merge_with_label['epoch_794'].tolist()
delta = df_merge_with_label['delta_next_day'].tolist()

ls_equity_line, ls_global_return, ls_mdd, ls_romad, ls_i, ls_j  = Measures.get_equity_return_mdd_romad(y_pred=y_pred, delta=delta, multiplier=MULTIPLIER, type='long_short')
lh_equity_line, lh_global_return, lh_mdd, lh_romad, lh_i, lh_j  = Measures.get_equity_return_mdd_romad(y_pred=y_pred, delta=delta, multiplier=MULTIPLIER, type='long_only')
sh_equity_line, sh_global_return, sh_mdd, sh_romad, sh_i, sh_j  = Measures.get_equity_return_mdd_romad(y_pred=y_pred, delta=delta, multiplier=MULTIPLIER, type='short_only')

long, short, hold, general = Measures.get_precision_count_coverage(y_pred=y_pred, delta=delta)
long_poc, short_poc = Measures.get_precision_over_coverage(y_pred=y_pred, delta=delta)

print(short)