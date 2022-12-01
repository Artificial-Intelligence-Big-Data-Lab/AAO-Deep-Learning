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

import os
import pandas as pd 
import argparse
import json

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID";
os.environ["CUDA_VISIBLE_DEVICES"]="0";

## RESULTS ZONE ###
thr_ensemble_magg = [0.3, 0.33, 0.34, 0.35, 0.36, 0.37, 0.38, 0.39, 0.40, 0.41, 0.42, 0.43, 0.44, 0.45, 0.46, 0.47, 0.48, 0.49, 0.50]
thr_exclusive = [0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.80, 0.85, 0.90, 0.95, 1]

thr_elimination = [] #[0.35, 0.40, 0.45, 0.50, 0.55, 0.60]

#experiments = ['069 - gold and gold silver - pesi 27']


#experiments = ['072 - gold - pesi 27 controprova di exp 071']

#experiments = ['073 - gold and gold silver - pesi bilanciati']

experiments = [ 
                '080 - Test multivariate SP500 con tutti i blocchi',
                '081 - Test multivariate SP500 senza blocco 1d',
                '082 - Test multivariate SP500 senza blocco 8h',
                '083 - Test multivariate SP500 senza blocco 4h',
                '084 - Test multivariate SP500 senza blocco 1h',
                '085 - Test multivariate SP500 senza blocchi 1d 8h',
                '086 - Test multivariate SP500 senza blocchi 1h 4h',
                '087 - Test univariate SP500 unico blocco 1h',
                '088 - Test univariate SP500 unico blocco 4h',
                '089 - Test multivariate SP500 unico blocco 8h'
                '090 - Test univariate SP500 unico blocco 1d',
                '091 - Test univariate SP500 unico blocco 1h 40x40',
                '092 - Test multivariate SP500 tutti i blocchi delta percentuale'
        ]

experiments = [ 
                '080 - Test multivariate SP500 con tutti i blocchi',
                '085 - Test multivariate SP500 senza blocchi 1d 8h',
                '087 - Test univariate SP500 unico blocco 1h',
                '092 - Test multivariate SP500 tutti i blocchi delta percentuale'
        ]

experiments = ['112 - SP500 univariate blocco unico 1h 20x20 100 reti walk 10 mesi allineati','113 - SP500 univariate blocco unico 1h 20x20 100 reti walk 8 mesi allineati']
experiments = ['00 - Debug new code TMP']

for experiment in experiments:
        rh = ResultsHandler(experiment_name=experiment)

        experiment_number = experiment.split('-')[0]
        
        generate_results(experiment_name=experiment, single_net=True)
        #rh.generate_json(single_net=True)

        '''
        rh.get_csv_alg4(validation_thr=15, validation_metric='romad', epoch_selection_policy='long_short',  stop_loss=1000, penalty=25)
        rh.get_csv_alg4(validation_thr=15, validation_metric='romad', epoch_selection_policy='long_only',  stop_loss=1000, penalty=25)
        rh.get_csv_alg4(validation_thr=15, validation_metric='romad', epoch_selection_policy='short_only',  stop_loss=1000, penalty=25)
        rh.get_final_decision_alg4(ensemble_type='ensemble_magg', thrs=thr_ensemble_magg, epoch_selection_policy='long_short')
        rh.get_final_decision_alg4(ensemble_type='ensemble_exclusive', thrs=thr_exclusive, epoch_selection_policy='long_only')
        rh.get_final_decision_alg4(ensemble_type='ensemble_exclusive_short', thrs=thr_exclusive, epoch_selection_policy='short_only')
        
        #rh.generate_ensemble(thrs_ensemble_magg=[], thrs_ensemble_exclusive=thr_exclusive, thrs_ensemble_elimination=[], remove_nets=False)
        

        for thr in thr_ensemble_magg:
                #rh.get_final_decision_from_ensemble(type='ensemble_magg', validation_thr=15, validation_metric='romad', epoch_selection_policy='long_short', thr_ensemble_magg=thr, stop_loss=1000, penalty=25)
                #rh.get_final_decision_from_ensemble(type='ensemble_magg', validation_thr=15, validation_metric='romad', epoch_selection_policy='long_only', thr_ensemble_magg=thr, stop_loss=1000, penalty=25)
                #rh.get_final_decision_from_ensemble(type='ensemble_magg', validation_thr=15, validation_metric='romad', epoch_selection_policy='short_only', thr_ensemble_magg=thr, stop_loss=1000, penalty=25)
                rh.get_report_excel(type='ensemble_magg', epoch_selection_policy='long_short', thr=thr, stop_loss=1000, penalty=25, report_name=experiment_number + '- Rep Ens Magg - Sel. epoca Long Short - SL1000 Pen25 - thr ' + str(thr), decision_folder='test_alg4', subfolder_is='ALG 4 0-200')
                #rh.get_report_excel(type='ensemble_magg', epoch_selection_policy='long_only', thr=thr, stop_loss=1000, penalty=25, report_name=experiment_number + '- Rep Ens Magg - Sel. epoca Long Only - SL1000 Pen25 - thr ' + str(thr), decision_folder='test_alg4')
                #rh.get_report_excel(type='ensemble_magg', epoch_selection_policy='short_only', thr=thr, stop_loss=1000, penalty=25, report_name=experiment_number + '- Rep Ens Magg - Sel. epoca Short Only - SL1000 Pen25 - thr ' + str(thr), decision_folder='test_alg4')
        
        for thr in thr_exclusive:
                #rh.get_final_decision_from_ensemble(type='ensemble_exclusive', validation_thr=15, validation_metric='romad',
                #        epoch_selection_policy='long_only', thr_ensemble_exclusive=thr, stop_loss=1000, penalty=25)

                #rh.get_final_decision_from_ensemble(type='ensemble_exclusive_short', validation_thr=15, validation_metric='romad',
                #        epoch_selection_policy='short_only', thr_ensemble_exclusive=thr, stop_loss=1000, penalty=25)
                
                rh.get_report_excel(type='ensemble_exclusive', epoch_selection_policy='long_only', thr=thr, stop_loss=1000, penalty=25, 
                report_name=experiment_number + "- Rep Ens Excl - Sel. epoca Long Only - SL1000 Pen25 - thr " + str(thr), decision_folder='test_alg4', subfolder_is='ALG 4 0-200')

                rh.get_report_excel(type='ensemble_exclusive_short', epoch_selection_policy='short_only', thr=thr, stop_loss=1000, penalty=25, 
                        report_name=experiment_number + "- Rep Ens Excl Short- Sel. epoca Short Only - SL1000 Pen25 - thr " + str(thr), decision_folder='test_alg4', subfolder_is='ALG 4 0-200')
                

        rh.get_report_excel_swipe(thrs_ensemble_exclusive=thr_exclusive, epoch_selection_policy='long_only', type='ensemble_exclusive', stop_loss=1000, penalty=25,
                report_name=experiment_number + '- Test - Rep Swipe Ens Excl - Sel. epoca Long Only - SL1000 Pen25', decision_folder='test_alg4', subfolder_is='ALG 4 0-200')

        rh.get_report_excel_swipe(thrs_ensemble_exclusive=thr_exclusive, epoch_selection_policy='short_only', type='ensemble_exclusive_short', stop_loss=1000, penalty=25,
                report_name=experiment_number + '- Test - Rep Swipe Ens Excl - Sel. epoca Short Only - SL1000 Pen25', decision_folder='test_alg4', subfolder_is='ALG 4 0-200')

        rh.get_report_excel_swipe(thrs_ensemble_magg=thr_ensemble_magg, epoch_selection_policy='long_short', type='ensemble_magg', stop_loss=1000, penalty=25,
                report_name=experiment_number + '- Test - Rep Swipe Ens Magg - Sel. epoca Long Short - SL1000 Pen25', decision_folder='test_alg4', subfolder_is='ALG 4 0-200')
        '''

        '''

        #rh.generate_json()

        #rh.calculate_epoch_selection()

        #long_metrics = ['por15_grp20', 'por15_grp50', 'por15_grp100', 'por10_grp20', 'por10_grp50', 'por10_grp100', 'por5_grp20', 'por5_grp50', 'por5_grp100']
        #sort_metrics = ['por30_grp20','por30_grp50','por30_grp100','por20_grp20','por20_grp50','por20_grp100','por10_grp20','por10_grp50','por10_grp100']
        
        #pors = [2, 4, 6, 8, 10, 12]
        #for por in pors: 
        #for metric in long_metrics:
        
        #rh.get_report_excel(type='selection', metric='valid_romad', second_metric=[], epoch_selection_policy='long', 
        #        stop_loss=1000, penalty=25, report_name=experiment_number + "- Rep Selection " + "Por tra 7 e 10% Romad Magg 0.8 Cove 035 065 - Ordered by valid_cove ASC - 10 reti ensemble")

        #rh.get_report_excel(type='selection', metric='valid_romad', second_metric=[], epoch_selection_policy='long_short', 
        #       stop_loss=1000, penalty=25, report_name=experiment_number + "- valid_cove min 0.45 por tra 0 e 10 romad magg 0 - walk_1 - long_only - soglia 0.3") # - 


        #rh.get_report_excel(type='selection', metric='valid_romad', second_metric="valid_cove", epoch_selection_policy='long', stop_loss=1000, penalty=30, report_name=experiment_number + "- Rep Selection " + "valid romad Romad BH + Por 10" + " - Long Only - SL1000 Pen30")
        #h.get_report_excel(type='selection', metric='valid_romad', second_metric="valid_cove", epoch_selection_policy='short', stop_loss=1000, penalty=30, report_name=experiment_number + "- Rep Selection " + "valid romad Romad BH + Por 10" + " - Short Only - SL1000 Pen30")
        '''
        
        '''
        #rh.calculate_custom_metrics(penalty=0, stop_loss=0, start_by_walk=20, end_at_walk=20)
        #rh.calculate_custom_metrics(penalty=0, stop_loss=0)
        #rh.generate_json_avg(type='validation')
        #rh.generate_json_avg(type='test')
        #rh.generate_json_loss() # viene chiamato anche dentro calculate_custom_metrics() ma si può usare anche separatamente
        #rh.generate_json_accuracy()
        
        # calcolo il csv principale con le triple
        rh.generate_triple_csv(remove_nets=False, prob_mode='none', prob_thr=0.8) # prob_mode none | long_short_thr
        # calcolo gli ensemble per le varie soglie partendo dal file delle triple
        rh.generate_ensemble(thrs_ensemble_magg=thr_ensemble_magg, thrs_ensemble_exclusive=thr_exclusive, thrs_ensemble_elimination=thr_elimination, remove_nets=False)

        
        # Per ogni soglia ensemble magg calcolo il file decisioni finale e poi gli excel per ogni report + excel swipe
        for thr in thr_ensemble_magg:
                rh.get_final_decision_from_ensemble(type='ensemble_magg', validation_thr=15, validation_metric='romad', epoch_selection_policy='long_short', thr_ensemble_magg=thr, stop_loss=0, penalty=0)
                rh.get_final_decision_from_ensemble(type='ensemble_magg', validation_thr=15, validation_metric='romad', epoch_selection_policy='long_only', thr_ensemble_magg=thr, stop_loss=0, penalty=0)
                rh.get_final_decision_from_ensemble(type='ensemble_magg', validation_thr=15, validation_metric='romad', epoch_selection_policy='short_only', thr_ensemble_magg=thr, stop_loss=0, penalty=0)
                rh.get_report_excel(type='ensemble_magg', epoch_selection_policy='long_short', thr=thr, stop_loss=1000, penalty=25, report_name=experiment_number + '- Rep Ens Magg - Sel. epoca Long Short - SL1000 Pen25 - thr ' + str(thr))
                rh.get_report_excel(type='ensemble_magg', epoch_selection_policy='long_only', thr=thr, stop_loss=1000, penalty=25, report_name=experiment_number + '- Rep Ens Magg - Sel. epoca Long Only - SL1000 Pen25 - thr ' + str(thr))
                rh.get_report_excel(type='ensemble_magg', epoch_selection_policy='short_only', thr=thr, stop_loss=1000, penalty=25, report_name=experiment_number + '- Rep Ens Magg - Sel. epoca Short Only - SL1000 Pen25 - thr ' + str(thr))
        


        rh.get_report_excel_swipe(thrs_ensemble_magg=thr_ensemble_magg, epoch_selection_policy='long_short', type='ensemble_magg', stop_loss=1000, penalty=25,
                report_name=experiment_number + '- Rep Swipe Ens Magg - Sel. epoca Long Short - SL1000 Pen25')

        rh.get_report_excel_swipe(thrs_ensemble_magg=thr_ensemble_magg, epoch_selection_policy='long_only', type='ensemble_magg', stop_loss=1000, penalty=25,
               report_name=experiment_number + '- Rep Swipe Ens Magg - Sel. epoca Long Only - SL1000 Pen25')

        rh.get_report_excel_swipe(thrs_ensemble_magg=thr_ensemble_magg, epoch_selection_policy='short_only', type='ensemble_magg', stop_loss=1000, penalty=25,
                report_name=experiment_number + '- Rep Swipe Ens Magg - Sel. epoca Short Only - SL1000 Pen25')
        '''

        '''
        # Per ogni soglia ensemble exclusive calcolo il file decisioni finale e poi gli excel per ogni report + excel swipe
        ''
        for thr in thr_exclusive:
                rh.get_final_decision_from_ensemble(type='ensemble_exclusive', validation_thr=15, validation_metric='romad',
                        epoch_selection_policy='long_only', thr_ensemble_exclusive=thr, stop_loss=0, penalty=0)
                rh.get_report_excel(type='ensemble_exclusive', epoch_selection_policy='long_only', thr=thr, stop_loss=1000, penalty=25, 
                       report_name=experiment_number + "- Rep Ens Excl - Sel. epoca Long Only - SL1000 Pen25 - thr " + str(thr))
        
        rh.get_report_excel_swipe(thrs_ensemble_exclusive=thr_exclusive, epoch_selection_policy='long_only', type='ensemble_exclusive', stop_loss=1000, penalty=25,
                report_name=experiment_number + '- Rep Swipe Ens Excl - Sel. epoca Long Only - SL1000 Pen25')
        '''

        '''
        
        ## ZONA REPORT IS 1 - IS 2
        ''
        # IS 1 - 2 - SP500
        IS_1 = ['2009-08-02', '2014-01-31']
        IS_2 = ['2014-02-02', '2017-01-31']
        OOS = ['2017-02-01', '2019-10-30']

        #IS_1 = ['2009-08-02', '2013-12-31']
        #IS_2 = ['2014-01-02', '2016-12-30']

        
        # ENSEMBLE MAGG
        rh.get_report_excel_swipe(thrs_ensemble_magg=thr_ensemble_magg, epoch_selection_policy='long_short', type='ensemble_magg', stop_loss=1000, penalty=25, subfolder_is='Swipe IS 1', subsample=IS_1,
                report_name=experiment_number + '- Rep Swipe Ens Magg - IS 1 - Sel. epoca Long Short - SL1000 Pen25')

        rh.get_report_excel_swipe(thrs_ensemble_magg=thr_ensemble_magg, epoch_selection_policy='long_only', type='ensemble_magg', stop_loss=1000, penalty=25, subfolder_is='Swipe IS 1', subsample=IS_1,
                report_name=experiment_number + '- Rep Swipe Ens Magg - IS 1 - Sel. epoca Long Only - SL1000 Pen25')

        rh.get_report_excel_swipe(thrs_ensemble_magg=thr_ensemble_magg, epoch_selection_policy='short_only', type='ensemble_magg', stop_loss=1000, penalty=25, subfolder_is='Swipe IS 1', subsample=IS_1,
                report_name=experiment_number + '- Rep Swipe Ens Magg - IS 1 - Sel. epoca Short Only - SL1000 Pen25')
        
        rh.get_report_excel_swipe(thrs_ensemble_magg=thr_ensemble_magg, epoch_selection_policy='long_short', type='ensemble_magg', stop_loss=1000, penalty=25, subfolder_is='Swipe IS 2', subsample=IS_2,
                report_name=experiment_number + '- Rep Swipe Ens Magg - IS 2 - Sel. epoca Long Short - SL1000 Pen25')

        rh.get_report_excel_swipe(thrs_ensemble_magg=thr_ensemble_magg, epoch_selection_policy='long_only', type='ensemble_magg', stop_loss=1000, penalty=25, subfolder_is='Swipe IS 2', subsample=IS_2,
                report_name=experiment_number + '- Rep Swipe Ens Magg - IS 2 - Sel. epoca Long Only - SL1000 Pen25')

        rh.get_report_excel_swipe(thrs_ensemble_magg=thr_ensemble_magg, epoch_selection_policy='short_only', type='ensemble_magg', stop_loss=1000, penalty=25, subfolder_is='Swipe IS 2', subsample=IS_2, 
                report_name=experiment_number + '- Rep Swipe Ens Magg - IS 2 - Sel. epoca Short Only - SL1000 Pen25')
        
        # EXCLUSIVE
        rh.get_report_excel_swipe(thrs_ensemble_exclusive=thr_exclusive, epoch_selection_policy='long_only', type='ensemble_exclusive', stop_loss=1000, penalty=25, subfolder_is='Swipe IS 1', subsample=IS_1,
                report_name=experiment_number + '- Rep Swipe Ens Excl - IS 1 - Sel. epoca Long Only - SL1000 P25')

        rh.get_report_excel_swipe(thrs_ensemble_exclusive=thr_exclusive, epoch_selection_policy='long_only', type='ensemble_exclusive', stop_loss=1000, penalty=25, subfolder_is='Swipe IS 2', subsample=IS_2,
                report_name=experiment_number + '- Rep Swipe Ens Excl - IS 2 - Sel. epoca Long Only - SL1000 P25')
        

        # OOS
        rh.get_report_excel(type='ensemble_exclusive', epoch_selection_policy='long_only', thr=0.45, stop_loss=1000, penalty=25, subfolder_is='Report OOS', subsample=OOS,
                report_name=experiment_number + '- Rep Ens Excl - OOS - Sel. epoca Long Only - SL1000 Pen25 - thr ' + str(0.45))

        rh.get_report_excel(type='ensemble_exclusive', epoch_selection_policy='long_only', thr=0.5, stop_loss=1000, penalty=25, subfolder_is='Report OOS', subsample=OOS,
                report_name=experiment_number + '- Rep Ens Excl - OOS - Sel. epoca Long Only - SL1000 Pen25 - thr ' + str(0.5))

        rh.get_report_excel(type='ensemble_exclusive', epoch_selection_policy='long_only', thr=0.55, stop_loss=1000, penalty=25, subfolder_is='Report OOS', subsample=OOS,
                report_name=experiment_number + '- Rep Ens Excl - OOS - Sel. epoca Long Only - SL1000 Pen25 - thr ' + str(0.55))
        '''