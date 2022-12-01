from classes.ResultsHandler import ResultsHandler

def generate_results(experiment_name='', experiment_path='', single_net=True): 
        thr_ensemble_magg = [0.3, 0.33, 0.34, 0.35, 0.36, 0.37, 0.38, 0.39, 0.40, 0.41, 0.42, 0.43, 0.44, 0.45, 0.46, 0.47, 0.48, 0.49, 0.50]
        thr_exclusive = [0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.80, 0.85, 0.90, 0.95, 1]
        thr_elimination = [] #[0.35, 0.40, 0.45, 0.50, 0.55, 0.60]

        rh = ''
        if experiment_path != '':
                rh = ResultsHandler(experiment_name=experiment_name, experiment_path=experiment_path)
        else: 
             rh = ResultsHandler(experiment_name=experiment_name)

        experiment_number = experiment_name.split('-')[0]

        
        # Calcola i json. Single_net=true ricalcola anche quelli delle reti singole 
        rh.generate_json(single_net=single_net)

        '''
        # calcolo il csv principale con le triple utili per gli ensemble
        rh.generate_triple_csv(remove_nets=False)
        
        
        # ALG4 METHODS
        rh.get_csv_alg4(validation_thr=15, validation_metric='romad', epoch_selection_policy='long_short',  stop_loss=1000, penalty=25)
        rh.get_csv_alg4(validation_thr=15, validation_metric='romad', epoch_selection_policy='long_only',  stop_loss=1000, penalty=25)
        rh.get_csv_alg4(validation_thr=15, validation_metric='romad', epoch_selection_policy='short_only',  stop_loss=1000, penalty=25)

        rh.get_final_decision_alg4(ensemble_type='ensemble_magg', thrs=thr_ensemble_magg, epoch_selection_policy='long_short')
        rh.get_final_decision_alg4(ensemble_type='ensemble_magg', thrs=thr_ensemble_magg, epoch_selection_policy='long_only')
        rh.get_final_decision_alg4(ensemble_type='ensemble_magg', thrs=thr_ensemble_magg, epoch_selection_policy='short_only')

        rh.get_final_decision_alg4(ensemble_type='ensemble_exclusive', thrs=thr_exclusive, epoch_selection_policy='long_only')
        rh.get_final_decision_alg4(ensemble_type='ensemble_exclusive_short', thrs=thr_exclusive, epoch_selection_policy='short_only')
        
        
        # calcolo gli ensemble per le varie soglie partendo dal file delle triple
        rh.generate_ensemble(thrs_ensemble_magg=thr_ensemble_magg, thrs_ensemble_exclusive=thr_exclusive, thrs_ensemble_elimination=thr_elimination, remove_nets=False)
        
        # CALCOLA CSV SELECTION
        #rh.calculate_epoch_selection()
        
        # Per ogni soglia ensemble magg calcolo il file decisioni finale e poi gli excel per ogni report + excel swipe
        for thr in thr_ensemble_magg:
                #ALG3
                rh.get_final_decision_from_ensemble(type='ensemble_magg', validation_thr=15, validation_metric='romad', epoch_selection_policy='long_short', thr_ensemble_magg=thr, stop_loss=1000, penalty=25)
                rh.get_final_decision_from_ensemble(type='ensemble_magg', validation_thr=15, validation_metric='romad', epoch_selection_policy='long_only', thr_ensemble_magg=thr, stop_loss=1000, penalty=25)
                rh.get_final_decision_from_ensemble(type='ensemble_magg', validation_thr=15, validation_metric='romad', epoch_selection_policy='short_only', thr_ensemble_magg=thr, stop_loss=1000, penalty=25)
                
                rh.get_report_excel(type='ensemble_magg', epoch_selection_policy='long_short', thr=thr, stop_loss=1000, penalty=25, 
                        report_name=experiment_number + '- Test - Rep Ens Magg - Sel. epoca Long Short - SL1000 Pen25 - thr ' + str(thr), subfolder_is='ALG 3/Ens Magg')

                rh.get_report_excel(type='ensemble_magg', epoch_selection_policy='long_only', thr=thr, stop_loss=1000, penalty=25, 
                        report_name=experiment_number + '- Test - Rep Ens Magg - Sel. epoca Long Only - SL1000 Pen25 - thr ' + str(thr), subfolder_is='ALG 3/Ens Magg')

                rh.get_report_excel(type='ensemble_magg', epoch_selection_policy='short_only', thr=thr, stop_loss=1000, penalty=25, 
                        report_name=experiment_number + '- Test - Rep Ens Magg - Sel. epoca Short Only - SL1000 Pen25 - thr ' + str(thr), subfolder_is='ALG 3/Ens Magg')

                #ALG4
                rh.get_report_excel(type='ensemble_magg', epoch_selection_policy='long_short', thr=thr, stop_loss=1000, penalty=25, 
                        report_name=experiment_number + '- Test - Rep Ens Magg - Sel. epoca Long Short - SL1000 Pen25 - thr ' + str(thr), decision_folder='test_alg4', subfolder_is='ALG 4/Ens Magg')

                rh.get_report_excel(type='ensemble_magg', epoch_selection_policy='long_only', thr=thr, stop_loss=1000, penalty=25, 
                        report_name=experiment_number + '- Test - Rep Ens Magg - Sel. epoca Long Only - SL1000 Pen25 - thr ' + str(thr), decision_folder='test_alg4', subfolder_is='ALG 4/Ens Magg')

                rh.get_report_excel(type='ensemble_magg', epoch_selection_policy='short_only', thr=thr, stop_loss=1000, penalty=25, 
                        report_name=experiment_number + '- Test - Rep Ens Magg - Sel. epoca Short Only - SL1000 Pen25 - thr ' + str(thr), decision_folder='test_alg4', subfolder_is='ALG 4/Ens Magg')

        # SWIPE MAGG ALG 3    
        rh.get_report_excel_swipe(thrs_ensemble_magg=thr_ensemble_magg, epoch_selection_policy='long_short', type='ensemble_magg', stop_loss=1000, penalty=25,
                report_name=experiment_number + '- Test - Rep Swipe Ens Magg - Sel. epoca Long Short - SL1000 Pen25', subfolder_is='ALG 3')

        rh.get_report_excel_swipe(thrs_ensemble_magg=thr_ensemble_magg, epoch_selection_policy='long_only', type='ensemble_magg', stop_loss=1000, penalty=25,
                report_name=experiment_number + '- Test - Rep Swipe Ens Magg - Sel. epoca Long Only - SL1000 Pen25', subfolder_is='ALG 3')

        rh.get_report_excel_swipe(thrs_ensemble_magg=thr_ensemble_magg, epoch_selection_policy='short_only', type='ensemble_magg', stop_loss=1000, penalty=25,
                report_name=experiment_number + '- Test - Rep Swipe Ens Magg - Sel. epoca Short Only - SL1000 Pen25', subfolder_is='ALG 3')

        # SWIPE MAGG ALG 4
        rh.get_report_excel_swipe(thrs_ensemble_magg=thr_ensemble_magg, epoch_selection_policy='long_short', type='ensemble_magg', stop_loss=1000, penalty=25,
                report_name=experiment_number + '- Test - Rep Swipe Ens Magg - Sel. epoca Long Short - SL1000 Pen25', decision_folder='test_alg4', subfolder_is='ALG 4')

        rh.get_report_excel_swipe(thrs_ensemble_magg=thr_ensemble_magg, epoch_selection_policy='long_only', type='ensemble_magg', stop_loss=1000, penalty=25,
                report_name=experiment_number + '- Test - Rep Swipe Ens Magg - Sel. epoca Long Only - SL1000 Pen25', decision_folder='test_alg4', subfolder_is='ALG 4')

        rh.get_report_excel_swipe(thrs_ensemble_magg=thr_ensemble_magg, epoch_selection_policy='short_only', type='ensemble_magg', stop_loss=1000, penalty=25,
                report_name=experiment_number + '- Test - Rep Swipe Ens Magg - Sel. epoca Short Only - SL1000 Pen25', decision_folder='test_alg4', subfolder_is='ALG 4')
                
        # Per ogni soglia ensemble exclusive calcolo il file decisioni finale e poi gli excel per ogni report + excel swipe
        for thr in thr_exclusive:
                #ALG3
                rh.get_final_decision_from_ensemble(type='ensemble_exclusive', validation_thr=15, validation_metric='romad',
                        epoch_selection_policy='long_only', thr_ensemble_exclusive=thr, stop_loss=1000, penalty=25)

                rh.get_final_decision_from_ensemble(type='ensemble_exclusive_short', validation_thr=15, validation_metric='romad',
                                epoch_selection_policy='short_only', thr_ensemble_exclusive=thr, stop_loss=1000, penalty=25)

                
                rh.get_report_excel(type='ensemble_exclusive', epoch_selection_policy='long_only', thr=thr, stop_loss=1000, penalty=25, 
                        report_name=experiment_number + "- Test - Rep Ens Excl - Sel. epoca Long Only - SL1000 Pen25 - thr " + str(thr), subfolder_is='ALG 3/Ens Excl Long')

                rh.get_report_excel(type='ensemble_exclusive_short', epoch_selection_policy='short_only', thr=thr, stop_loss=1000, penalty=25, 
                                report_name=experiment_number + "- Test - Rep Ens Excl Short- Sel. epoca Short Only - SL1000 Pen25 - thr " + str(thr), subfolder_is='ALG 3/Ens Excl Short')

                # ALG4
                rh.get_report_excel(type='ensemble_exclusive', epoch_selection_policy='long_only', thr=thr, stop_loss=1000, penalty=25, 
                        report_name=experiment_number + "- Test - Rep Ens Excl - Sel. epoca Long Only - SL1000 Pen25 - thr " + str(thr), decision_folder='test_alg4', subfolder_is='ALG 4/Ens Excl Long')

                rh.get_report_excel(type='ensemble_exclusive_short', epoch_selection_policy='short_only', thr=thr, stop_loss=1000, penalty=25, 
                        report_name=experiment_number + "- Test - Rep Ens Excl Short- Sel. epoca Short Only - SL1000 Pen25 - thr " + str(thr), decision_folder='test_alg4', subfolder_is='ALG 4/Ens Excl Short')

        # ALG3
        rh.get_report_excel_swipe(thrs_ensemble_exclusive=thr_exclusive, epoch_selection_policy='long_only', type='ensemble_exclusive', stop_loss=1000, penalty=25,
                report_name=experiment_number + '- Test - Rep Swipe Ens Excl - Sel. epoca Long Only - SL1000 Pen25', subfolder_is='ALG 3')

        rh.get_report_excel_swipe(thrs_ensemble_exclusive=thr_exclusive, epoch_selection_policy='short_only', type='ensemble_exclusive_short', stop_loss=1000, penalty=25,
                report_name=experiment_number + '- Test - Rep Swipe Ens Excl Short - Sel. epoca Short Only - SL1000 Pen25', subfolder_is='ALG 3')


        # SWIPE ALG 4 
        rh.get_report_excel_swipe(thrs_ensemble_exclusive=thr_exclusive, epoch_selection_policy='long_only', type='ensemble_exclusive', stop_loss=1000, penalty=25,
                        report_name=experiment_number + '- Test - Rep Swipe Ens Excl - Sel. epoca Long Only - SL1000 Pen25', decision_folder='test_alg4', subfolder_is='ALG 4')

        rh.get_report_excel_swipe(thrs_ensemble_exclusive=thr_exclusive, epoch_selection_policy='short_only', type='ensemble_exclusive_short', stop_loss=1000, penalty=25,
                report_name=experiment_number + '- Test - Rep Swipe Ens Excl - Sel. epoca Short Only - SL1000 Pen25', decision_folder='test_alg4', subfolder_is='ALG 4')
        '''