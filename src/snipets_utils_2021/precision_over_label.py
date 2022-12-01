import pandas as pd 
import numpy as np 
from classes.Measures import Measures
from classes.Utils import natural_keys, df_date_merger, df_date_merger_binary, do_plot, revert_probabilities
import matplotlib.pyplot as plt
from cycler import cycler
import os 

def plot(net_json): 
    plt.figure(figsize=(30,18))
    
    # plot the training loss and accuracy
    plt.style.use("ggplot")
    plt.rcParams['axes.prop_cycle'] = cycler(color='bgrcmyk')

    plt.xlabel("Epoch #")
    plt.ylabel("Per-class val. precision")

    plt.plot(np.arange(0, epochs), net_json['longs_precisions'], label="Long precision", color="green")
    plt.plot(np.arange(0, epochs), net_json['longs_label_coverage'], label="Long Coverage", color="darkgreen", linewidth=5, linestyle='--')

    plt.plot(np.arange(0, epochs), net_json['shorts_precisions'], label="Short precision", color="red")
    plt.plot(np.arange(0, epochs), net_json['shorts_label_coverage'], label="Short Coverage", color="darkred", linewidth=5, linestyle='--')

    plt.plot(np.arange(0, epochs), net_json['holds_precisions'], label="Short precision", color="gray")
    plt.plot(np.arange(0, epochs), net_json['holds_label_coverage'], label="Short Coverage", color="darkgray", linewidth=5, linestyle='--')

    plt.show()

def get_date_epochs_walk(path, walk):
    date_list = []
    epochs_list = []

    full_path = path + walk + '/'

    for filename in os.listdir(full_path):
        df = pd.read_csv(full_path + filename)
        date_list = df['date_time'].tolist()
        epochs_list = df.columns.values
        # rimuovo il primo ed il secondo elemento che sono rispettivamente col0, e date_time
        epochs_list = np.delete(epochs_list, [0, 1])
        break

    return np.array(date_list), np.array(epochs_list)

walks = [0, 1,2,3,4,5,6,7,8,9]
epochs = 500
index_net = 1

base_path = 'D:/PhD-Market-Nets/experiments-binary/001 - SP500 walk 1 anno sino al 2020 sp500 e vix BINARY/'
validation_path = 'D:/PhD-Market-Nets/experiments-binary/001 - SP500 walk 1 anno sino al 2020 sp500 e vix BINARY/predictions/predictions_during_training/validation/'
test_path = 'D:/PhD-Market-Nets/experiments-binary/001 - SP500 walk 1 anno sino al 2020 sp500 e vix BINARY/predictions/predictions_during_training/test/'

for type in ['test', 'validation']:
    for index_walk in walks: 
        walk_str = 'walk_' + str(index_walk)

        if type == 'validation':
            date_list, epochs_list = get_date_epochs_walk(path=validation_path, walk=walk_str)
        if type == 'test': 
            date_list, epochs_list = get_date_epochs_walk(path=test_path, walk=walk_str)

        net = 'net_' + str(index_net) + '.csv'
        # leggo le predizioni fatte con l'esnemble
        df = pd.read_csv(base_path + 'predictions/predictions_during_training/' + type + '/walk_' + str(index_walk) + '/' + net)

        # mergio con le label, così ho un subset del df con le date che mi servono e la predizione 
        df_merge_with_label = df_date_merger_binary(df=df, columns=['date_time', 'delta_next_day', 'delta_current_day', 'close', 'open', 'high', 'low'], dataset='sp500_cet', thr=-0.5)

        #df_merge_with_label['date_time'] = df_merge_with_label['date_time'].shift(-1)
        df_merge_with_label = df_merge_with_label.drop(df.index[0])
        df_merge_with_label = df_merge_with_label.drop_duplicates(subset='date_time', keep="first")


        # PRECISIONI E LINEA RETTA DEL BILANCIAMENTO DELLE CLASSI
        longs_precisions = []
        shorts_precisions = []
        holds_precisions = []

        longs_label_coverage = []
        shorts_label_coverage = []
        holds_label_coverage = []

        for epoch in range(1, len(epochs_list) + 1): 
            df_epoch_rename = df_merge_with_label
            df_epoch_rename = df_epoch_rename.rename(columns={'epoch_' + str(epoch): 'decision'})

            long, short, hold = Measures.get_precision_over_label(df=df_epoch_rename.copy(), label_to_use='label_next_day')

            longs_precisions.append(long['precision'])
            shorts_precisions.append(short['precision'])
            holds_precisions.append(hold['precision'])

            longs_label_coverage.append(long['random_perc'])
            shorts_label_coverage.append(short['random_perc'])
            holds_label_coverage.append(hold['random_perc'])


    
        net_json = {
            "longs_precisions": longs_precisions,
            "shorts_precisions": shorts_precisions,
            "holds_precisions": holds_precisions,

            "longs_label_coverage": longs_label_coverage,
            "shorts_label_coverage": shorts_label_coverage,
            "holds_label_coverage": holds_label_coverage,
        }

        plot(net_json)


