from classes.ResultsHandlerNuovo import ResultsHandlerNuovo
import time
validation_thr = 15

experiments = ['009 - 18gennaio - replica exp 004 fullbatch']
#experiments = ['015 - 28gennaio - ripetizione 012']
#experiments = ['015 - 28gennaio - ripetizione 012']
#experiments = ['005 - pesiloss5 - 16gennaio']

thr_ensemble_magg = [0.33, 0.34, 0.35, 0.36, 0.37, 0.38, 0.39, 0.40, 0.41, 0.42, 0.43, 0.44, 0.45, 0.46, 0.47, 0.48, 0.49, 0.50]
thr_exclusive = [0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60]
thr_elimination = [] #[0.35, 0.40, 0.45, 0.50, 0.55, 0.60]

for experiment_name in experiments:
    rh = ResultsHandlerNuovo(experiment_name=experiment_name)
    
        
    # PROVA STOP LOSS 
    #rh.get_report_excel(type='ensemble_magg', thr=0.35, report_name='SENZA LOSSS - Report ensemble_magg - thr ' + str(0.35) + ' - penalty')
    start = time.time()
    #rh.get_final_decision_from_ensemble(type='ensemble_magg', validation_thr=15, validation_metric='romad', thr_ensemble_magg=0.35)
    rh.get_report_excel(type='ensemble_magg', thr=0.35, stop_loss=1000, penalty=32,
                        report_name='provastoplossss - Report ensemble_magg - thr ' + str(0.35) + ' - penalty')
    end = time.time()
    print("Tempo di esecuzione: " + "{:.3f}".format(end-start))

    #rh.generate_triple_csv(remove_nets=False)

    #rh.generate_ensemble(thrs_ensemble_magg=thr_ensemble_magg, thrs_ensemble_exclusive=thr_exclusive, thrs_ensemble_elimination=thr_elimination, remove_nets=False)

    #for thr in thr_ensemble_magg:
    #    rh.get_final_decision_from_ensemble(type='ensemble_magg', validation_thr=15, validation_metric='romad', thr_ensemble_magg=thr)
    #    rh.get_report_excel(type='ensemble_magg', thr=thr, report_name= experiment_name + ' - Report ensemble_magg - thr ' + str(thr) + ' - penalty')
    #rh.get_report_excel_swipe(thrs_ensemble_magg=thr_ensemble_magg, type='ensemble_magg', report_name= experiment_name + ' - Report Swipe ensemble_magg - penalty')

    #for thr in thr_exclusive:
    #    rh.get_final_decision_from_ensemble(type='ensemble_exclusive', validation_thr=50, validation_metric='romad', thr_ensemble_exclusive=thr)
    #    rh.get_report_excel(type='ensemble_exclusive', thr=thr, report_name= experiment_name + "Report ensemble_exclusive - thr " + str(thr) + ' - penalty')
    #rh.get_report_excel_swipe(thrs_ensemble_exclusive=thr_exclusive, type='ensemble_exclusive', report_name= experiment_name + ' - Report Swipe ensemble exclusive - penalty')

    #rh.get_report_excel_swipe(thr_exclusive=thr_exclusive, type='ensemble_exclusive', report_name= experiment_name + ' - Report Swipe ensemble_exclusive - penalty')

    #rh.run_ensemble()
    #thrs_magg = [0.33, 0.34, 0.35, 0.36, 0.37, 0.38, 0.39, 0.40, 0.41, 0.42, 0.43, 0.44, 0.45, 0.46, 0.47, 0.48, 0.49, 0.50]
    #rh.run_ensemble(thrs=thrs_magg, remove_nets=False)
    
    #for thr in thrs_magg:
    #    rh.get_final_decision_from_ensemble(type='ensemble_magg', validation_thr=validation_thr, validation_metric='romad', ensemble_thr=thr)
    #    rh.get_report_excel(type='ensemble_magg', ensemble_thr=thr, report_name= experiment_name + ' - Report ensemble_magg - thr ' + str(thr) + ' - penalty')


    #thrs_exclusive = [0.35, 0.40, 0.45, 0.50, 0.55, 0.60]    
    #thrs_exclusive = [0.30, 0.31, 0.32, 0.33, 0.34]    
    #thrs_exclusive = [0.33]    
    #for thr in thrs_exclusive:
    #    rh.get_final_decision_from_ensemble(type='ensemble_el_exclusive', validation_thr=validation_thr, validation_metric='romad', perc_agreement=thr)
    #    rh.get_report_excel(type='ensemble_magg', ensemble_thr=thr, report_name= experiment_name + ' - PROVA EQUITY ' + str(thr) + ' - penalty')
