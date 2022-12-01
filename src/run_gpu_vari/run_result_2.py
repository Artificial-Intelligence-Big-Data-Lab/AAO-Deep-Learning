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

import os
import pandas as pd 
import argparse
import json

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID";
os.environ["CUDA_VISIBLE_DEVICES"]="0";

## RESULTS ZONE ###
thr_ensemble_magg = [0.33, 0.34, 0.35, 0.36, 0.37, 0.38, 0.39, 0.40, 0.41, 0.42, 0.43, 0.44, 0.45, 0.46, 0.47, 0.48, 0.49, 0.50]
thr_exclusive = [0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60]
thr_elimination = [] #[0.35, 0.40, 0.45, 0.50, 0.55, 0.60]

experiments = ['042_a - NO thr - 9 con prob']
experiments = ['042_f - 0.98 thr - 9 con prob']

#experiments = ['009 - 18gennaio - replica exp 004 fullbatch', '047 - Ripetizione 9 MULTIVARIATO - SP500']

for experiment in experiments:
        rh = ResultsHandler(experiment_name=experiment)

        experiment_number = experiment.split('-')[0]
        #rh.calculate_custom_metrics(penalty=0, stop_loss=0)
        #rh.generate_json_loss() # viene chiamato anche dentro calculate_custom_metrics() ma si può usare anche separatamente

        
        # calcolo il csv principale con le triple
        #rh.generate_triple_csv(remove_nets=False, prob_mode='none', prob_thr=0.98)
        # calcolo gli ensemble per le varie soglie partendo dal file delle triple
        #rh.generate_ensemble(thrs_ensemble_magg=thr_ensemble_magg, thrs_ensemble_exclusive=thr_exclusive, thrs_ensemble_elimination=thr_elimination, remove_nets=False)

        '''
        # Per ogni soglia ensemble magg calcolo il file decisioni finale e poi gli excel per ogni report + excel swipe
        for thr in thr_ensemble_magg:
                rh.get_final_decision_from_ensemble(type='ensemble_magg', validation_thr=15, validation_metric='romad', epoch_selection_policy='long_short', thr_ensemble_magg=thr, stop_loss=1000, penalty=25)
                rh.get_final_decision_from_ensemble(type='ensemble_magg', validation_thr=15, validation_metric='romad', epoch_selection_policy='long_only', thr_ensemble_magg=thr, stop_loss=1000, penalty=25)
                rh.get_final_decision_from_ensemble(type='ensemble_magg', validation_thr=15, validation_metric='romad', epoch_selection_policy='short_only', thr_ensemble_magg=thr, stop_loss=1000, penalty=25)
                rh.get_report_excel(type='ensemble_magg', epoch_selection_policy='long_short', thr=thr, stop_loss=1000, penalty=25, report_name=experiment_number + '- Rep Ens Magg - Sel. epoca Long Short - SL1000 Pen25 - thr ' + str(thr))
                rh.get_report_excel(type='ensemble_magg', epoch_selection_policy='long_only', thr=thr, stop_loss=1000, penalty=25, report_name=experiment_number + '- Rep Ens Magg - Sel. epoca Long Only - SL1000 Pen25 - thr ' + str(thr))
                rh.get_report_excel(type='ensemble_magg', epoch_selection_policy='short_only', thr=thr, stop_loss=1000, penalty=25, report_name=experiment_number + '- Rep Ens Magg - Sel. epoca Short Only - SL1000 Pen25 - thr ' + str(thr))

        rh.get_report_excel_swipe(thrs_ensemble_magg=thr_ensemble_magg, epoch_selection_policy='long_short', type='ensemble_magg', stop_loss=1000, penalty=25,
                report_name=experiment_number + '- Rep Swipe Ens Magg - Sel. epoca Long Short - SL1000 Pen25')

        rh.get_report_excel_swipe(thrs_ensemble_magg=thr_ensemble_magg, epoch_selection_policy='long_only', type='ensemble_magg', stop_loss=1000, penalty=25,
               report_name=experiment_number + '- Rep Swipe Ens Magg - Sel. epoca Long Only - SL1000 Pen25')

        rh.get_report_excel_swipe(thrs_ensemble_magg=thr_ensemble_magg, epoch_selection_policy='short_only', type='ensemble_magg', stop_loss=1000, penalty=25,
                report_name=experiment_number + '- Rep Swipe Ens Magg - Sel. epoca Short Only - SL1000 Pen25')
        
        # Per ogni soglia ensemble exclusive calcolo il file decisioni finale e poi gli excel per ogni report + excel swipe
        '''
        for thr in thr_exclusive:
                rh.get_final_decision_from_ensemble(type='ensemble_exclusive', validation_thr=15, validation_metric='romad',
                        epoch_selection_policy='long_only', thr_ensemble_exclusive=thr, stop_loss=1000, penalty=25)
                rh.get_report_excel(type='ensemble_exclusive', epoch_selection_policy='long_only', thr=thr, stop_loss=1000, penalty=25, 
                       report_name=experiment_number + "- Rep Ens Excl - Sel. epoca Long Only - SL1000 Pen25 - thr " + str(thr))

        rh.get_report_excel_swipe(thrs_ensemble_exclusive=thr_exclusive, epoch_selection_policy='long_only', type='ensemble_exclusive', stop_loss=1000, penalty=25,
                report_name=experiment_number + '- Rep Swipe Ens Excl - Sel. epoca Long Only - SL1000 Pen25')
        
         
        
        ## ZONA REPORT IS 1 - IS 2
        
        # IS 1 - 2 - SP500
        IS_1 = ['2009-08-02', '2014-01-31']
        IS_2 = ['2014-02-02', '2017-01-31']
        OOS = ['2017-02-01', '2019-10-30']

        #IS_1 = ['2009-08-02', '2013-12-31']
        #IS_2 = ['2014-01-02', '2016-12-30']

        '''
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
        '''
        # EXCLUSIVE
        rh.get_report_excel_swipe(thrs_ensemble_exclusive=thr_exclusive, epoch_selection_policy='long_only', type='ensemble_exclusive', stop_loss=1000, penalty=25, subfolder_is='Swipe IS 1', subsample=IS_1,
                report_name=experiment_number + '- Rep Swipe Ens Excl - IS 1 - Sel. epoca Long Only - SL1000 P25')

        rh.get_report_excel_swipe(thrs_ensemble_exclusive=thr_exclusive, epoch_selection_policy='long_only', type='ensemble_exclusive', stop_loss=1000, penalty=25, subfolder_is='Swipe IS 2', subsample=IS_2,
                report_name=experiment_number + '- Rep Swipe Ens Excl - IS 2 - Sel. epoca Long Only - SL1000 P25')
        

        # OOS
        rh.get_report_excel(type='ensemble_exclusive', epoch_selection_policy='long_only', thr=0.45, stop_loss=1000, penalty=25, subfolder_is='Report OOS', subsample=OOS,
                report_name=experiment_number + '- Rep Ens Excl - OOS - Sel. epoca Long Only - SL1000 Pen25 - thr ' + str(0.45))
        