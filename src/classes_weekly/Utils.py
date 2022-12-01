import os
import re
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
from cycler import cycler
from classes.Market import Market

'''
'
'''
def atoi(text):
    return int(text) if text.isdigit() else text

def natural_keys(text):
    return [atoi(c) for c in re.split(r'(\d+)', text)]

'''
'
'''
def df_date_merger(df, columns, thr_hold, dataset='sp500_cet'): 
    # dataset wrapper
    dataset = Market(dataset=dataset)
    dataset_label = dataset.get_label(freq='1d', columns=columns, thr=thr_hold)
    dataset_label = dataset_label.reset_index()
    dataset_label['date_time'] = dataset_label['date_time'].astype(str)

    df['date_time'] = df['date_time'].astype(str)
    df_merge = pd.merge(df, dataset_label, how="inner")

    return df_merge

'''
'
'''
def create_folder(path):
    # creo la path finale     
    if not os.path.isdir(path):
        os.makedirs(path)

'''
'
'''
def convert_probabilities(y_pred_prob):
    return str(np.round(float(y_pred_prob[0]), 2)) + ";" + str(np.round(float(y_pred_prob[1]),2)) + ";" + str(np.round(float(y_pred_prob[2]) ,2))

'''
'
'''
def do_plot(metrics, walk, epochs, main_path, experiment_name, net='average', type='validation'):
    plt.figure(figsize=(30,18))
    
    if net == 'average':
        plt.figtext(0.1, 0.98, "Averages | WALK #" + str(walk), fontsize='xx-large')
    else:
        plt.figtext(0.1, 0.98, "SINGLE NET " + str(net) + " | WALK #" + str(walk), fontsize='xx-large')

    # plot the training loss and accuracy
    plt.style.use("ggplot")
    plt.rcParams['axes.prop_cycle'] = cycler(color='bgrcmyk')
    plt.subplot(2, 3, 1)

    plt.xlabel("Epoch #")
    plt.ylabel("Romad")
    plt.plot(np.arange(0, epochs), metrics['ls_romads'], label="ROMAD Long+Short")
    plt.plot(np.arange(0, epochs), metrics['lh_romads'], label="ROMAD Long-only")
    plt.plot(np.arange(0, epochs), metrics['sh_romads'], label="ROMAD Short-only")
    plt.plot(np.arange(0, epochs), metrics['bh_romads'], label="Return Buy&Hold (USD)", color="orange")
    plt.plot(np.arange(0, epochs), np.zeros(epochs), color="black")
    plt.legend(loc="upper left")

    plt.subplot(2, 3, 2)
    plt.xlabel("Epoch #")
    plt.ylabel("Return (USD)")
    
    plt.plot(np.arange(0, epochs), metrics['ls_returns'], label="Return Long+Short")
    plt.plot(np.arange(0, epochs), metrics['lh_returns'], label="Return Long-only")
    plt.plot(np.arange(0, epochs), metrics['sh_returns'], label="Return Short-only")
    plt.plot(np.arange(0, epochs), metrics['bh_returns'], label="Return B&H", color="orange")
    plt.plot(np.arange(0, epochs), np.zeros(epochs), color="black")
    plt.legend(loc="upper left")

    plt.subplot(2, 3, 3)
    plt.xlabel("Epoch #")
    plt.ylabel("Mdd (USD)")
    
    plt.plot(np.arange(0, epochs), metrics['ls_mdds'], label="Max drawdown Long+Short")
    plt.plot(np.arange(0, epochs), metrics['lh_mdds'], label="Max drawdown Long-only")
    plt.plot(np.arange(0, epochs), metrics['sh_mdds'], label="Max drawdon Short-only")
    plt.plot(np.arange(0, epochs), metrics['bh_mdds'], label="Return B&H", color="orange")
    plt.plot(np.arange(0, epochs), np.zeros(epochs), color="black")
    plt.legend(loc="upper left")

    plt.subplot(2, 3, 4)
    plt.xlabel("Epoch #")
    plt.ylabel("Per-class val. precision")

    plt.plot(np.arange(0, epochs), metrics['longs_precisions'], label="Long precision", color="green")
    plt.plot(np.arange(0, epochs), metrics['longs_label_coverage'], label="Long Coverage", color="darkgreen", linewidth=5, linestyle='--')

    plt.plot(np.arange(0, epochs), metrics['shorts_precisions'], label="Short precision", color="red")
    plt.plot(np.arange(0, epochs), metrics['shorts_label_coverage'], label="Short Coverage", color="darkred", linewidth=5, linestyle='--')

    plt.plot(np.arange(0, epochs), np.full((epochs, 1), 0.5), color="black")
    plt.legend(loc="upper left")

    plt.subplot(2, 3, 5)
    plt.xlabel("Epoch #")
    plt.ylabel("Perc Operations")
    plt.plot(np.arange(0, epochs), metrics['long_operations'], label='% of long operations', color="green")
    plt.plot(np.arange(0, epochs), metrics['hold_operations'], label="% of hold operations", color="grey")
    plt.plot(np.arange(0, epochs), metrics['short_operations'], label="% of short operations", color="red")
    plt.legend(loc="upper left")

    plt.subplot(2, 3, 6)
    plt.xlabel("Epoch #")
    plt.ylabel("Perc Operations")
    plt.plot(np.arange(0, epochs), metrics['longs_poc'], label='Long precision over coverage', color="green")
    plt.plot(np.arange(0, epochs), metrics['shorts_poc'], label="Short precision over coverage", color="red")
    plt.plot(np.arange(0, epochs), np.zeros(epochs), color="black")
    plt.legend(loc="upper left")

    #plt.show()
    #plt.close('all')

    if type is 'validation': 
        type = 'a_validation'

    if net == 'average':
        # Se non esiste la cartella, la creo
        if not os.path.isdir(main_path + experiment_name + '/accuracy_loss_plots/average/'):
            os.makedirs(main_path + experiment_name + '/accuracy_loss_plots/average/')
        plt.savefig(main_path + experiment_name + '/accuracy_loss_plots/average/walk_' + str(walk) + '_AVG_ ' + type + '.png')
    else:
        # Se non esiste la cartella, la creo
        if not os.path.isdir(main_path + experiment_name + '/accuracy_loss_plots/walk_' + str(walk) + '/'):
            os.makedirs(main_path + experiment_name + '/accuracy_loss_plots/walk_' + str(walk) + '/')
        plt.savefig(main_path + experiment_name + '/accuracy_loss_plots/walk_' + str(walk) + '/net_' + str(net) + '_' + type + '.png')

    plt.close('all')