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
thr_exclusive = [0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.80, 0.85, 0.90, 0.95, 1]

thr_elimination = [] #[0.35, 0.40, 0.45, 0.50, 0.55, 0.60]


experiments = [ '085 - Test multivariate SP500 senza blocchi 1d 8h' ]
experiments = ['100 - Analisi coverage - Ripetizione 93 - Unico blocco 1h']

for experiment in experiments:
        rh = ResultsHandler(experiment_name=experiment)

        experiment_number = experiment.split('-')[0]

        #rh.generate_ensemble(thrs_ensemble_magg=[], thrs_ensemble_exclusive=thr_exclusive, thrs_ensemble_elimination=[], remove_nets=False)
        
        '''
        for thr in thr_ensemble_magg:
                #rh.get_final_decision_from_ensemble(type='ensemble_magg', validation_thr=15, validation_metric='romad', epoch_selection_policy='long_short', thr_ensemble_magg=thr, stop_loss=1000, penalty=25)
                #rh.get_final_decision_from_ensemble(type='ensemble_magg', validation_thr=15, validation_metric='romad', epoch_selection_policy='long_only', thr_ensemble_magg=thr, stop_loss=1000, penalty=25)
                #rh.get_final_decision_from_ensemble(type='ensemble_magg', validation_thr=15, validation_metric='romad', epoch_selection_policy='short_only', thr_ensemble_magg=thr, stop_loss=1000, penalty=25)
                rh.get_report_excel(type='ensemble_magg', epoch_selection_policy='long_short', thr=thr, stop_loss=1000, penalty=25, report_name=experiment_number + '- Rep Ens Magg - Sel. epoca Long Short - SL1000 Pen25 - thr ' + str(thr))
                rh.get_report_excel(type='ensemble_magg', epoch_selection_policy='long_only', thr=thr, stop_loss=1000, penalty=25, report_name=experiment_number + '- Rep Ens Magg - Sel. epoca Long Only - SL1000 Pen25 - thr ' + str(thr))
                rh.get_report_excel(type='ensemble_magg', epoch_selection_policy='short_only', thr=thr, stop_loss=1000, penalty=25, report_name=experiment_number + '- Rep Ens Magg - Sel. epoca Short Only - SL1000 Pen25 - thr ' + str(thr))

        for thr in thr_exclusive:
                #rh.get_final_decision_from_ensemble(type='ensemble_exclusive', validation_thr=15, validation_metric='romad',
               #         epoch_selection_policy='long_only', thr_ensemble_exclusive=thr, stop_loss=1000, penalty=25)

                #rh.get_final_decision_from_ensemble(type='ensemble_exclusive_short', validation_thr=15, validation_metric='romad',
                #        epoch_selection_policy='short_only', thr_ensemble_exclusive=thr, stop_loss=1000, penalty=25)

                rh.get_report_excel(type='ensemble_exclusive', epoch_selection_policy='long_only', thr=thr, stop_loss=1000, penalty=25, 
                report_name=experiment_number + "- Rep Ens Excl - Sel. epoca Long Only - SL1000 Pen25 - thr " + str(thr))

                rh.get_report_excel(type='ensemble_exclusive_short', epoch_selection_policy='short_only', thr=thr, stop_loss=1000, penalty=25, 
                        report_name=experiment_number + "- Rep Ens Excl Short- Sel. epoca Short Only - SL1000 Pen25 - thr " + str(thr))        
        '''
        '''
        # validation
        rh.get_report_excel_swipe(thrs_ensemble_exclusive=thr_exclusive, epoch_selection_policy='long_only', type='ensemble_exclusive', stop_loss=1000, penalty=25,
                report_name=experiment_number + '- Valid - Rep Swipe Ens Excl - Sel. epoca Long Only - SL1000 Pen25', decision_folder='validation')
        rh.get_report_excel_swipe(thrs_ensemble_exclusive=thr_exclusive, epoch_selection_policy='short_only', type='ensemble_exclusive_short', stop_loss=1000, penalty=25,
                report_name=experiment_number + '- Valid - Rep Swipe Ens Excl - Sel. epoca Short Only - SL1000 Pen25', decision_folder='validation')
        rh.get_report_excel_swipe(thrs_ensemble_magg=thr_ensemble_magg, epoch_selection_policy='long_short', type='ensemble_magg', stop_loss=1000, penalty=25,
                report_name=experiment_number + '- Valid - Rep Swipe Ens Magg - Sel. epoca Long Short - SL1000 Pen25', decision_folder='validation')
        

        # test
        rh.get_report_excel_swipe(thrs_ensemble_exclusive=thr_exclusive, epoch_selection_policy='long_only', type='ensemble_exclusive', stop_loss=1000, penalty=25,
                report_name=experiment_number + '- Test - Rep Swipe Ens Excl - Sel. epoca Long Only - SL1000 Pen25', decision_folder='test')
        rh.get_report_excel_swipe(thrs_ensemble_exclusive=thr_exclusive, epoch_selection_policy='short_only', type='ensemble_exclusive_short', stop_loss=1000, penalty=25,
                report_name=experiment_number + '- Test - Rep Swipe Ens Excl - Sel. epoca Short Only - SL1000 Pen25', decision_folder='test')
        rh.get_report_excel_swipe(thrs_ensemble_magg=thr_ensemble_magg, epoch_selection_policy='long_short', type='ensemble_magg', stop_loss=1000, penalty=25,
                report_name=experiment_number + '- Test - Rep Swipe Ens Magg - Sel. epoca Long Short - SL1000 Pen25', decision_folder='test')
        

        for walk in range(0, 9):
                # validation
                rh.get_report_excel_swipe_per_walk(thrs_ensemble_exclusive=thr_exclusive, epoch_selection_policy='long_only', type='ensemble_exclusive', stop_loss=1000, penalty=25, subfolder='valid_swipe_per_walk/Exlusive Long',
                        report_name=experiment_number + '- Valid - Rep Swipe Ens Excl Walk ' + str(walk) + ' - Sel. epoca Long Only - SL1000 Pen25', decision_folder='validation', walk=walk)
                
                rh.get_report_excel_swipe_per_walk(thrs_ensemble_exclusive=thr_exclusive, epoch_selection_policy='short_only', type='ensemble_exclusive_short', stop_loss=1000, penalty=25, subfolder='valid_swipe_per_walk/Exlusive Short',
                      report_name=experiment_number + '- Valid - Rep Swipe Ens Excl Short Walk ' + str(walk) + ' - Sel. epoca Short Only - SL1000 Pen25', decision_folder='validation', walk=walk)

                rh.get_report_excel_swipe_per_walk(thrs_ensemble_magg=thr_ensemble_magg, epoch_selection_policy='long_short', type='ensemble_magg', stop_loss=1000, penalty=25, subfolder='valid_swipe_per_walk/Magg',
                        report_name=experiment_number + '- Valid - Rep Swipe Ens Magg Walk ' + str(walk) + ' - Sel. epoca Long Short - SL1000 Pen25', decision_folder='validation', walk=walk)

                # test
                rh.get_report_excel_swipe_per_walk(thrs_ensemble_exclusive=thr_exclusive, epoch_selection_policy='long_only', type='ensemble_exclusive', stop_loss=1000, penalty=25, subfolder='test_swipe_per_walk/Exlusive Long',
                        report_name=experiment_number + '- Test - Rep Swipe Ens Excl Walk ' + str(walk) + ' - Sel. epoca Long Only - SL1000 Pen25', decision_folder='test', walk=walk)
                
                rh.get_report_excel_swipe_per_walk(thrs_ensemble_exclusive=thr_exclusive, epoch_selection_policy='short_only', type='ensemble_exclusive_short', stop_loss=1000, penalty=25, subfolder='test_swipe_per_walk/Exlusive Short',
                      report_name=experiment_number + '- Test - Rep Swipe Ens Excl Short Walk ' + str(walk) + ' - Sel. epoca Short Only - SL1000 Pen25', decision_folder='test', walk=walk)

                rh.get_report_excel_swipe_per_walk(thrs_ensemble_magg=thr_ensemble_magg, epoch_selection_policy='long_short', type='ensemble_magg', stop_loss=1000, penalty=25, subfolder='test_swipe_per_walk/Magg',
                        report_name=experiment_number + '- Test - Rep Swipe Ens Magg Walk ' + str(walk) + ' - Sel. epoca Long Short - SL1000 Pen25', decision_folder='test', walk=walk)
        '''
        

        #rh.get_thr_per_walk_by_cvg(type='ensemble_exclusive', thrs_ensemble_exclusive=thr_exclusive, epoch_selection_policy='long_only', stop_loss=1000, penalty=25, remove_nets=False, target_cvg=0.5)
        #rh.get_thr_per_walk_by_cvg(type='ensemble_exclusive_short', thrs_ensemble_exclusive=thr_exclusive, epoch_selection_policy='short_only', stop_loss=1000, penalty=25, remove_nets=False, target_cvg=0.3)
        
        thrss = [11, 6, 7, 10, 7]

        rh.get_report_with_cvg_sel(report_name="100 - Report con selezione cvg 0.5", type='ensemble_exclusive', 
                epoch_selection_policy='long_only', thrs=thr_exclusive, selected_thrs=thrss)


        input()


        '''
        # LONG
        thrs_per_03 = [6, 11, 12, 12, 7, 7, 9, 7, 11]
        thrs_per_04 = [6, 11, 12, 10, 6, 6, 9, 6, 10]
        thrs_per_05 = [5, 11, 8, 10, 11, 5, 8, 5, 10]
        thrs_per_06 = [4, 10, 8, 11, 10, 4, 8, 5, 10]
        thrs_per_07 = [3, 10, 8, 9, 4, 3, 8, 9, 10]
        thrs_per_08 = [2, 9, 8, 9, 3, 2, 8, 3, 5]
        thrs_per_09 = [1, 9, 7, 3, 1, 1, 7, 1, 9]
        
        # SHORT
        thrs_per_03 = [7, 6, 6, 5, 4, 7, 6, 5, 4]
        thrs_per_04 = [6, 6, 5, 4, 4, 6, 5, 4, 4]
        thrs_per_05 = [4, 5, 4, 3, 3, 6, 5, 3, 3]
        thrs_per_06 = [3, 4, 3, 3, 2, 5, 4, 2, 3]
        thrs_per_07 = [3, 4, 3, 2, 2, 2, 4, 2, 2]
        thrs_per_08 = [2, 3, 2, 1, 0, 1, 1, 1, 0]
        thrs_per_09 = [1, 3, 0, 1, 1, 0, 0, 0, 0]

        #rh.get_report_with_cvg_sel(report_name="085 - Report con selezione cvg 0.3", type='ensemble_exclusive', epoch_selection_policy='long_only', thrs=thr_exclusive, selected_thrs=thrs_per_03)
        #rh.get_report_with_cvg_sel(report_name="085 - Report con selezione cvg 0.4", type='ensemble_exclusive', epoch_selection_policy='long_only', thrs=thr_exclusive, selected_thrs=thrs_per_04)
        #rh.get_report_with_cvg_sel(report_name="085 - Report con selezione cvg 0.5", type='ensemble_exclusive', epoch_selection_policy='long_only', thrs=thr_exclusive, selected_thrs=thrs_per_05)
        #rh.get_report_with_cvg_sel(report_name="085 - Report con selezione cvg 0.6", type='ensemble_exclusive', epoch_selection_policy='long_only', thrs=thr_exclusive, selected_thrs=thrs_per_06)
        #rh.get_report_with_cvg_sel(report_name="085 - Report con selezione cvg 0.7", type='ensemble_exclusive', epoch_selection_policy='long_only', thrs=thr_exclusive, selected_thrs=thrs_per_07)
        #rh.get_report_with_cvg_sel(report_name="085 - Report con selezione cvg 0.8", type='ensemble_exclusive', epoch_selection_policy='long_only', thrs=thr_exclusive, selected_thrs=thrs_per_08)
        #rh.get_report_with_cvg_sel(report_name="085 - Report con selezione cvg 0.9", type='ensemble_exclusive', epoch_selection_policy='long_only', thrs=thr_exclusive, selected_thrs=thrs_per_09)
        
        rh.get_report_with_cvg_sel(report_name="085 - Report con selezione cvg 0.3", type='ensemble_exclusive_short', epoch_selection_policy='short_only', thrs=thr_exclusive, selected_thrs=thrs_per_03)
        rh.get_report_with_cvg_sel(report_name="085 - Report con selezione cvg 0.4", type='ensemble_exclusive_short', epoch_selection_policy='short_only', thrs=thr_exclusive, selected_thrs=thrs_per_04)
        rh.get_report_with_cvg_sel(report_name="085 - Report con selezione cvg 0.5", type='ensemble_exclusive_short', epoch_selection_policy='short_only', thrs=thr_exclusive, selected_thrs=thrs_per_05)
        rh.get_report_with_cvg_sel(report_name="085 - Report con selezione cvg 0.6", type='ensemble_exclusive_short', epoch_selection_policy='short_only', thrs=thr_exclusive, selected_thrs=thrs_per_06)
        rh.get_report_with_cvg_sel(report_name="085 - Report con selezione cvg 0.7", type='ensemble_exclusive_short', epoch_selection_policy='short_only', thrs=thr_exclusive, selected_thrs=thrs_per_07)
        rh.get_report_with_cvg_sel(report_name="085 - Report con selezione cvg 0.8", type='ensemble_exclusive_short', epoch_selection_policy='short_only', thrs=thr_exclusive, selected_thrs=thrs_per_08)
        rh.get_report_with_cvg_sel(report_name="085 - Report con selezione cvg 0.9", type='ensemble_exclusive_short', epoch_selection_policy='short_only', thrs=thr_exclusive, selected_thrs=thrs_per_09)
        '''