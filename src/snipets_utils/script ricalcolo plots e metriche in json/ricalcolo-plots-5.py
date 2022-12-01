from classes.Market import Market
from classes.Measures import Measures
from classes.Utils import df_date_merger
import os
import json

import numpy as np 
from numpy import inf
import numpy as geek 
import pandas as pd 
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score
from cycler import cycler

EPOCHS = 800
MAX_WALK = 4
MULTIPLIER = 50
NETS = 30


experiments = [
                #'022 - diminuita class weight', # 4 walks 30 reti,
                '003 - pesiloss3 - 4walk - 16gennaio'
            ]


'''
experiments = [
                    '001 - ExpSalva_5_3_15gennaio', 
                    '002 - ExpSalva_5_3_15gennaio',
                    '005 - pesiloss5 - 16gennaio',
                    '006 - 17gennaio',
                    '007 - 17gennaio',
                    '008 - 18gennaio',
                    '010 - 21gennaio',
                    '011 - 22gennaio - class weight',
                    '012 - prova loss - primo',
                    '013 - prova loss - secondo',
                    '015 - 28gennaio - ripetizione 012'
                ]
'''
def do_plot(net, walk, type):
    plt.figure(figsize=(30,18))
    
    plt.figtext(0.1, 0.98, "SINGLE NET " + str(net) + " | WALK #" + str(walk), fontsize='xx-large')
    #plt.figtext(0.1, 0.96, "Training set: " + str(self.training_set[index_walk][0]) + " - " + str(self.training_set[index_walk][1]), fontsize='x-large')
    #plt.figtext(0.1, 0.94, "Validing set: " + str(self.validation_set[index_walk][0]) + " - " + str(self.validation_set[index_walk][1]), fontsize='x-large')
    #plt.figtext(0.1, 0.92, "Epochs: " + str(EPOCHS) + " - Nets: " + str(self.number_of_nets) + " - Batch Size: " + str(self.bs), fontsize='x-large')

    # plot the training loss and accuracy
    plt.style.use("ggplot")
    plt.rcParams['axes.prop_cycle'] = cycler(color='bgrcmyk')
    plt.subplot(2, 3, 1)

    plt.xlabel("Epoch #")
    plt.ylabel("Romad")
    plt.plot(np.arange(0, EPOCHS), ls_romads, label="ROMAD Long+Short")
    plt.plot(np.arange(0, EPOCHS), lh_romads, label="ROMAD Long-only")
    plt.plot(np.arange(0, EPOCHS), sh_romads, label="ROMAD Short-only")
    plt.plot(np.arange(0, EPOCHS), bh_romads, label="Return Buy&Hold (USD)", color="orange")
    plt.plot(np.arange(0, EPOCHS), np.zeros(EPOCHS), color="black")
    plt.legend(loc="upper left")

    plt.subplot(2, 3, 2)
    plt.xlabel("Epoch #")
    plt.ylabel("Return (USD)")
    
    plt.plot(np.arange(0, EPOCHS), ls_returns, label="Return Long+Short")
    plt.plot(np.arange(0, EPOCHS), lh_returns, label="Return Long-only")
    plt.plot(np.arange(0, EPOCHS), sh_returns, label="Return Short-only")
    plt.plot(np.arange(0, EPOCHS), bh_returns, label="Return B&H", color="orange")
    plt.plot(np.arange(0, EPOCHS), np.zeros(EPOCHS), color="black")
    plt.legend(loc="upper left")

    plt.subplot(2, 3, 3)
    plt.xlabel("Epoch #")
    plt.ylabel("Mdd (USD)")
    
    plt.plot(np.arange(0, EPOCHS), ls_mdds, label="Max drawdown Long+Short")
    plt.plot(np.arange(0, EPOCHS), lh_mdds, label="Max drawdown Long-only")
    plt.plot(np.arange(0, EPOCHS), sh_mdds, label="Max drawdon Short-only")
    plt.plot(np.arange(0, EPOCHS), bh_mdds, label="Return B&H", color="orange")
    plt.plot(np.arange(0, EPOCHS), np.zeros(EPOCHS), color="black")
    plt.legend(loc="upper left")

    plt.subplot(2, 3, 4)
    plt.xlabel("Epoch #")
    plt.ylabel("Per-class val. precision")

    plt.plot(np.arange(0, EPOCHS), longs_precisions, label="Long precision", color="green")
    plt.plot(np.arange(0, EPOCHS), longs_label_coverage, label="Long Coverage", color="darkgreen", linewidth=5, linestyle='--')

    plt.plot(np.arange(0, EPOCHS), shorts_precisions, label="Short precision", color="red")
    plt.plot(np.arange(0, EPOCHS), shorts_label_coverage, label="Short Coverage", color="darkred", linewidth=5, linestyle='--')

    plt.plot(np.arange(0, EPOCHS), np.full((EPOCHS, 1), 0.5), color="black")
    plt.legend(loc="upper left")

    plt.subplot(2, 3, 5)
    plt.xlabel("Epoch #")
    plt.ylabel("Perc Operations")
    plt.plot(np.arange(0, EPOCHS), long_operations, label='% of long operations', color="green")
    plt.plot(np.arange(0, EPOCHS), hold_operations, label="% of hold operations", color="grey")
    plt.plot(np.arange(0, EPOCHS), short_operations, label="% of short operations", color="red")
    plt.legend(loc="upper left")

    
    plt.subplot(2, 3, 6)
    plt.xlabel("Epoch #")
    plt.ylabel("Perc Operations")
    plt.plot(np.arange(0, EPOCHS), longs_poc, label='Long precision over coverage', color="green")
    plt.plot(np.arange(0, EPOCHS), shorts_poc, label="Short precision over coverage", color="red")
    plt.plot(np.arange(0, EPOCHS), np.zeros(EPOCHS), color="black")
    plt.legend(loc="upper left")
    
    #plt.show()
    #plt.close('all')

    if type is 'validation': 
        type = 'a_validation'

    # Se non esiste la cartella, la creo
    if not os.path.isdir('../experiments/' + experiment_name + '/accuracy_loss_plots/walk_' + str(walk) + '/'):
        os.makedirs('../experiments/' + experiment_name + '/accuracy_loss_plots/walk_' + str(walk) + '/')


    plt.savefig('../experiments/' + experiment_name + '/accuracy_loss_plots/walk_' + str(walk) + '/net_' + str(net) + '_' + type + '.png')
    
    plt.close('all')


def do_plot_avg(walk, type):
    plt.figure(figsize=(30,18))
    
    plt.figtext(0.1, 0.98, "Averages | WALK #" + str(walk), fontsize='xx-large')
    #plt.figtext(0.1, 0.96, "Training set: " + str(self.training_set[index_walk][0]) + " - " + str(self.training_set[index_walk][1]), fontsize='x-large')
    #plt.figtext(0.1, 0.94, "Validing set: " + str(self.validation_set[index_walk][0]) + " - " + str(self.validation_set[index_walk][1]), fontsize='x-large')
    #plt.figtext(0.1, 0.92, "Epochs: " + str(EPOCHS) + " - Nets: " + str(self.number_of_nets) + " - Batch Size: " + str(self.bs), fontsize='x-large')

    # plot the training loss and accuracy
    plt.style.use("ggplot")
    plt.rcParams['axes.prop_cycle'] = cycler(color='bgrcmyk')
    plt.subplot(2, 3, 1)

    plt.xlabel("Epoch #")
    plt.ylabel("Romad")
    plt.plot(np.arange(0, EPOCHS), avg_ls_romads, label="ROMAD Long+Short")
    plt.plot(np.arange(0, EPOCHS), avg_lh_romads, label="ROMAD Long-only")
    plt.plot(np.arange(0, EPOCHS), avg_sh_romads, label="ROMAD Short-only")
    plt.plot(np.arange(0, EPOCHS), avg_bh_romads, label="Return Buy&Hold (USD)", color="orange")
    plt.plot(np.arange(0, EPOCHS), np.zeros(EPOCHS), color="black")
    plt.legend(loc="upper left")

    plt.subplot(2, 3, 2)
    plt.xlabel("Epoch #")
    plt.ylabel("Return (USD)")
    
    plt.plot(np.arange(0, EPOCHS), avg_ls_returns, label="Return Long+Short")
    plt.plot(np.arange(0, EPOCHS), avg_lh_returns, label="Return Long-only")
    plt.plot(np.arange(0, EPOCHS), avg_sh_returns, label="Return Short-only")
    plt.plot(np.arange(0, EPOCHS), avg_bh_returns, label="Return B&H", color="orange")
    plt.plot(np.arange(0, EPOCHS), np.zeros(EPOCHS), color="black")
    plt.legend(loc="upper left")

    plt.subplot(2, 3, 3)
    plt.xlabel("Epoch #")
    plt.ylabel("Mdd (USD)")
    
    plt.plot(np.arange(0, EPOCHS), avg_ls_mdds, label="Max drawdown Long+Short")
    plt.plot(np.arange(0, EPOCHS), avg_lh_mdds, label="Max drawdown Long-only")
    plt.plot(np.arange(0, EPOCHS), avg_sh_mdds, label="Max drawdon Short-only")
    plt.plot(np.arange(0, EPOCHS), avg_bh_mdds, label="Return B&H", color="orange")
    plt.plot(np.arange(0, EPOCHS), np.zeros(EPOCHS), color="black")
    plt.legend(loc="upper left")

    plt.subplot(2, 3, 4)
    plt.xlabel("Epoch #")
    plt.ylabel("Per-class val. precision")

    plt.plot(np.arange(0, EPOCHS), avg_longs_precisions, label="Long precision", color="green")
    plt.plot(np.arange(0, EPOCHS), avg_label_longs_coverage, label="Long Coverage", color="darkgreen", linewidth=5, linestyle='--')

    plt.plot(np.arange(0, EPOCHS), avg_shorts_precisions, label="Short precision", color="red")
    plt.plot(np.arange(0, EPOCHS), avg_label_shorts_coverage, label="Short Coverage", color="darkred", linewidth=5, linestyle='--')

    plt.plot(np.arange(0, EPOCHS), np.full((EPOCHS, 1), 0.5), color="black")
    plt.legend(loc="upper left")

    plt.subplot(2, 3, 5)
    plt.xlabel("Epoch #")
    plt.ylabel("Perc Operations")
    plt.plot(np.arange(0, EPOCHS), avg_long_operations, label='% of long operations', color="green")
    plt.plot(np.arange(0, EPOCHS), avg_hold_operations, label="% of hold operations", color="grey")
    plt.plot(np.arange(0, EPOCHS), avg_short_operations, label="% of short operations", color="red")
    plt.legend(loc="upper left")

    plt.subplot(2, 3, 6)
    plt.xlabel("Epoch #")
    plt.ylabel("Perc Operations")
    plt.plot(np.arange(0, EPOCHS), avg_longs_poc, label='Long precision over coverage', color="green")
    plt.plot(np.arange(0, EPOCHS), avg_shorts_poc, label="Short precision over coverage", color="red")
    plt.plot(np.arange(0, EPOCHS), np.zeros(EPOCHS), color="black")
    plt.legend(loc="upper left")

    #plt.show()
    #plt.close('all')

    if type is 'validation': 
        type = 'a_validation'

    # Se non esiste la cartella, la creo
    if not os.path.isdir('../experiments/' + experiment_name + '/accuracy_loss_plots/average/'):
        os.makedirs('../experiments/' + experiment_name + '/accuracy_loss_plots/average/')

    plt.savefig('../experiments/' + experiment_name + '/accuracy_loss_plots/average/walk_' + str(walk) + '_AVG_ ' + type + '.png')
    
    plt.close('all')
################################################# 
#################################################

for experiment_name in experiments: 
    for index_walk in range(0, MAX_WALK):
        for type in ['validation', 'test']: 
            avg_ls_returns = []
            avg_lh_returns = []
            avg_sh_returns = []
            avg_bh_returns = [] # BH

            # MDDS
            avg_ls_mdds = []
            avg_lh_mdds = []
            avg_sh_mdds = []
            avg_bh_mdds = [] # BH

            # ROMADS
            avg_ls_romads = []
            avg_lh_romads = []
            avg_sh_romads = []
            avg_bh_romads = []

            avg_longs_precisions = []
            avg_shorts_precisions = []
            
            avg_label_longs_coverage = []
            avg_label_shorts_coverage = []

            # NUOVO TEST AVG POC
            avg_longs_poc = []
            avg_shorts_poc = []

            # RETURNS
            all_ls_returns = np.zeros(shape=(NETS, EPOCHS))
            all_lh_returns = np.zeros(shape=(NETS, EPOCHS))
            all_sh_returns = np.zeros(shape=(NETS, EPOCHS))
            all_bh_return = np.zeros(shape=(NETS, EPOCHS)) #BH

            # ROMADS
            all_ls_romads = np.zeros(shape=(NETS, EPOCHS))
            all_lh_romads = np.zeros(shape=(NETS, EPOCHS))
            all_sh_romads = np.zeros(shape=(NETS, EPOCHS))
            all_bh_romads = np.zeros(shape=(NETS, EPOCHS)) #BH

            # MDDS
            all_ls_mdds = np.zeros(shape=(NETS, EPOCHS))
            all_lh_mdds = np.zeros(shape=(NETS, EPOCHS))
            all_sh_mdds = np.zeros(shape=(NETS, EPOCHS))
            all_bh_mdds = np.zeros(shape=(NETS, EPOCHS)) # BH

            # PRECISIONI E LINEA RETTA DEL BILANCIAMENTO DELLE CLASSI
            all_longs_precisions = np.zeros(shape=(NETS, EPOCHS))
            all_shorts_precisions = np.zeros(shape=(NETS, EPOCHS))
            all_labels_longs_coverage = np.zeros(shape=(NETS, EPOCHS))
            all_labels_shorts_coverage = np.zeros(shape=(NETS, EPOCHS))

            # % di operazioni fatte 
            all_long_operations = np.zeros(shape=(NETS, EPOCHS))
            all_short_operations = np.zeros(shape=(NETS, EPOCHS))
            all_hold_operations = np.zeros(shape=(NETS, EPOCHS))

            # Precision over coverage
            all_longs_poc = np.zeros(shape=(NETS, EPOCHS))
            all_shorts_poc = np.zeros(shape=(NETS, EPOCHS))

            for index_net in range(0, NETS):
                net = 'net_' + str(index_net) + '.csv'
                # leggo le predizioni fatte con l'esnemble
                df = pd.read_csv('../experiments/' + experiment_name + '/predictions/predictions_during_training/' + type + '/walk_' + str(index_walk) + '/' + net)

                # mergio con le label, così ho un subset del df con le date che mi servono e la predizione 
                df_merge_with_label = df_date_merger(df=df, columns=['date_time', 'delta_next_day', 'close', 'open'], dataset='sp500_cet')

                # RETURNS 
                ls_returns = []
                lh_returns = []
                sh_returns = []
                bh_returns = [] # BH

                # ROMADS
                ls_romads = []
                lh_romads = []
                sh_romads = []
                bh_romads = [] # BH

                # MDDS
                ls_mdds = []
                lh_mdds = []
                sh_mdds = []
                bh_mdds = [] # BH

                # PRECISIONI E LINEA RETTA DEL BILANCIAMENTO DELLE CLASSI
                longs_precisions = []
                shorts_precisions = []
                longs_label_coverage = []
                shorts_label_coverage = []

                # % DI OPERAZIONI FATTE
                long_operations = []
                short_operations = []
                hold_operations = []

                # POC
                longs_poc = []
                shorts_poc = []
                

                label_coverage = Measures.get_delta_coverage(delta=df_merge_with_label['delta_next_day'].tolist())

                bh_equity_line, bh_global_return, bh_mdd, bh_romad, bh_i, bh_j  = Measures.get_return_mdd_romad_bh(close=df_merge_with_label['close'].tolist(), multiplier=MULTIPLIER)

                # calcolo il return per un epoca
                for epoch in range(1, EPOCHS + 1): 
                    y_pred = df_merge_with_label['epoch_' + str(epoch)].tolist()
                    delta = df_merge_with_label['delta_next_day'].tolist()

                    ls_equity_line, ls_global_return, ls_mdd, ls_romad, ls_i, ls_j  = Measures.get_equity_return_mdd_romad(y_pred=y_pred, delta=delta, multiplier=MULTIPLIER, type='long_short')
                    lh_equity_line, lh_global_return, lh_mdd, lh_romad, lh_i, lh_j  = Measures.get_equity_return_mdd_romad(y_pred=y_pred, delta=delta, multiplier=MULTIPLIER, type='long_only')
                    sh_equity_line, sh_global_return, sh_mdd, sh_romad, sh_i, sh_j  = Measures.get_equity_return_mdd_romad(y_pred=y_pred, delta=delta, multiplier=MULTIPLIER, type='short_only')

                    long, short, hold, general = Measures.get_precision_count_coverage(y_pred=y_pred, delta=delta)
                    long_poc, short_poc = Measures.get_precision_over_coverage(y_pred=y_pred, delta=delta)

                    # RETURNS 
                    ls_returns.append(ls_global_return)
                    lh_returns.append(lh_global_return)
                    sh_returns.append(sh_global_return)
                    bh_returns.append(bh_global_return) # BH

                    # ROMADS
                    ls_romads.append(ls_romad)
                    lh_romads.append(lh_romad)
                    sh_romads.append(sh_romad)
                    bh_romads.append(bh_romad) # BH

                    # MDDS
                    ls_mdds.append(ls_mdd)
                    lh_mdds.append(lh_mdd)
                    sh_mdds.append(sh_mdd)
                    bh_mdds.append(bh_mdd) # BH

                    # PRECISIONI E LINEA RETTA DEL BILANCIAMENTO DELLE CLASSI
                    longs_precisions.append(long['precision'])
                    shorts_precisions.append(short['precision'])
                    longs_label_coverage.append(label_coverage['long'])
                    shorts_label_coverage.append(label_coverage['short'])

                    # % di operazioni fatte
                    long_operations.append(long['coverage'])
                    short_operations.append(short['coverage'])
                    hold_operations.append(hold['coverage'])

                    # POC
                    longs_poc.append(long_poc)
                    shorts_poc.append(short_poc)

                net_json = {
                    "ls_returns": ls_returns,
                    "lh_returns": lh_returns,
                    "sh_returns": sh_returns,
                    "bh_returns": bh_returns,

                    "ls_romads": ls_romads,
                    "lh_romads": lh_romads,
                    "sh_romads": sh_romads,
                    "bh_romads": bh_romads,

                    "ls_mdds": ls_mdds,
                    "lh_mdds": lh_mdds,
                    "sh_mdds": sh_mdds,
                    "bh_mdds": bh_mdds,

                    "longs_precisions": longs_precisions,
                    "shorts_precisions": shorts_precisions,
                    "longs_label_coverage": longs_label_coverage,
                    "shorts_label_coverage": shorts_label_coverage,

                    "long_operations": long_operations,
                    "short_operations": short_operations,
                    "hold_operations": hold_operations,

                    "longs_poc": longs_poc,
                    "shorts_poc": shorts_poc           
                }

                

                output_path = '../experiments/' + experiment_name + '/calculated_metrics/' + type + '/walk_' + str(index_walk) + '/' 

                if not os.path.isdir(output_path):
                    os.makedirs(output_path)

                with open(output_path + 'net_' + str(index_net) + '.json', 'w') as json_file:
                    json.dump(net_json, json_file, indent=4)
                    #json.dump(net_json, json_file)
                
                # PLOT SINGOLA RETE
                do_plot(walk=index_walk, net=index_net, type=type)
                print(experiment_name + ' | ' + type + " - Salvate le metriche per walk n° ", index_walk, " rete: ", net)

                # RETURNS
                all_ls_returns[index_net] = ls_returns
                all_lh_returns[index_net] = lh_returns
                all_sh_returns[index_net] = sh_returns
                all_bh_return[index_net] = bh_returns # BH

                # ROMADS
                all_ls_romads[index_net] = ls_romads
                all_lh_romads[index_net] = lh_romads
                all_sh_romads[index_net] = sh_romads
                all_bh_romads[index_net] = bh_romads # BH

                # MDDS
                all_ls_mdds[index_net] = ls_mdds
                all_lh_mdds[index_net] = lh_mdds
                all_sh_mdds[index_net] = sh_mdds
                all_bh_mdds[index_net] = bh_mdds #BH

                # PRECISIONI E LINEA RETTA DEL BILANCIAMENTO DELLE CLASSI
                all_longs_precisions[index_net] = longs_precisions
                all_shorts_precisions[index_net] = shorts_precisions
                all_labels_longs_coverage[index_net] = longs_label_coverage
                all_labels_shorts_coverage[index_net] = shorts_label_coverage

                # % di operazioni fatte
                all_long_operations[index_net] = long_operations
                all_short_operations[index_net] = short_operations
                all_hold_operations[index_net] = hold_operations

                all_longs_poc[index_net] = longs_poc
                all_shorts_poc[index_net] = shorts_poc

            # RETURNS
            avg_ls_returns = np.around(np.average(all_ls_returns, axis=0), decimals=3)
            avg_lh_returns = np.around(np.average(all_lh_returns, axis=0), decimals=3)
            avg_sh_returns = np.around(np.average(all_sh_returns, axis=0), decimals=3)
            avg_bh_returns = np.average(all_bh_return, axis=0) # BH

            # MDDS
            avg_ls_mdds = np.around(np.average(all_ls_mdds, axis=0), decimals=3)
            avg_lh_mdds = np.around(np.average(all_lh_mdds, axis=0), decimals=3)
            avg_sh_mdds = np.around(np.average(all_sh_mdds, axis=0), decimals=3)
            avg_bh_mdds = np.average(all_bh_mdds, axis=0) # BH

            # ROMADS
            avg_ls_romads = np.divide(avg_ls_returns, avg_ls_mdds, out=np.zeros_like(avg_ls_returns), where=avg_ls_mdds!=0)
            avg_lh_romads = np.divide(avg_lh_returns, avg_lh_mdds, out=np.zeros_like(avg_ls_returns), where=avg_ls_mdds!=0)
            avg_sh_romads = np.divide(avg_sh_returns, avg_sh_mdds, out=np.zeros_like(avg_ls_returns), where=avg_ls_mdds!=0)
            avg_bh_romads = np.divide(avg_bh_returns, avg_bh_mdds)

            # rimuovo i nan dai romads
            avg_ls_romads = np.around(np.nan_to_num(avg_ls_romads), decimals=3)
            avg_lh_romads = np.around(np.nan_to_num(avg_lh_romads), decimals=3)
            avg_sh_romads = np.around(np.nan_to_num(avg_sh_romads), decimals=3)
            avg_sh_romads[~np.isfinite(avg_sh_romads)] = 0

            avg_longs_precisions = np.around(np.average(all_longs_precisions, axis=0), decimals=3)
            avg_shorts_precisions = np.around(np.average(all_shorts_precisions, axis=0), decimals=3)

            avg_label_longs_coverage = np.around(np.average(all_labels_longs_coverage, axis=0), decimals=3)
            avg_label_shorts_coverage = np.around(np.average(all_labels_shorts_coverage, axis=0), decimals=3)

            # NUOVO TEST AVG POC
            avg_longs_poc = np.around(np.divide(avg_longs_precisions, avg_label_longs_coverage), decimals=3)
            avg_shorts_poc = np.around(np.divide(avg_shorts_precisions, avg_label_shorts_coverage), decimals=3)

            avg_longs_poc = (avg_longs_poc - 1 ) * 100
            
            for avg_id, avg in enumerate(avg_longs_poc):
                if avg_longs_poc[avg_id] < -30:
                    avg_longs_poc[avg_id] = -30
                if avg_longs_poc[avg_id] > 30:
                    avg_longs_poc[avg_id] = 30

            avg_shorts_poc = (avg_shorts_poc - 1 ) * 100

            for avg_id, avg in enumerate(avg_shorts_poc):
                if avg_shorts_poc[avg_id] < -30:
                    avg_shorts_poc[avg_id] = -30
                if avg_shorts_poc[avg_id] > 30:
                    avg_shorts_poc[avg_id] = 30

            avg_long_operations = np.average(all_long_operations, axis=0)
            avg_short_operations= np.average(all_short_operations, axis=0)
            avg_hold_operations = np.average(all_hold_operations, axis=0)



            avg_json = {
                    "ls_returns": avg_ls_returns.tolist(),
                    "lh_returns": avg_lh_returns.tolist(),
                    "sh_returns": avg_sh_returns.tolist(),
                    "bh_returns": avg_bh_returns.tolist(), # BH

                    "ls_romads": avg_ls_romads.tolist(),
                    "lh_romads": avg_lh_romads.tolist(),
                    "sh_romads": avg_sh_romads.tolist(),
                    "bh_romads": avg_bh_romads.tolist(), # BH

                    "ls_mdds": avg_ls_mdds.tolist(),
                    "lh_mdds": avg_lh_mdds.tolist(),
                    "sh_mdds": avg_sh_mdds.tolist(),
                    "bh_mdds": avg_bh_mdds.tolist(), # BH

                    "longs_precisions": avg_longs_precisions.tolist(),
                    "shorts_precisions": avg_shorts_precisions.tolist(),
                    "longs_label_coverage": avg_label_longs_coverage.tolist(),
                    "shorts_label_coverage": avg_label_shorts_coverage.tolist(),

                    "long_operations": avg_long_operations.tolist(),
                    "short_operations": avg_short_operations.tolist(),
                    "hold_operations": avg_hold_operations.tolist(),

                    "longs_poc": avg_longs_poc.tolist(),
                    "shorts_poc": avg_shorts_poc.tolist(),
                }

            avg_output_path = '../experiments/' + experiment_name + '/calculated_metrics/' + type + '/average/' 

            if not os.path.isdir(avg_output_path):
                os.makedirs(avg_output_path)

            with open(avg_output_path + 'walk_' + str(index_walk) + '.json', 'w') as json_file:
                json.dump(avg_json, json_file, indent=4)
                #json.dump(net_json, json_file)
            
            print(experiment_name + ' | ' + type + " - Salvate le metriche AVG per walk n° ", index_walk)

            do_plot_avg(walk=index_walk, type=type)
        