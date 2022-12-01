from classes.ResultsHandler import ResultsHandler

validation_thr = 15

experiments = [ '015 - 28gennaio - ripetizione 012' ]


for experiment_name in experiments:
    rh = ResultsHandler(experiment_name=experiment_name)

    thrs_magg = [0.33, 0.34, 0.35, 0.36, 0.37, 0.38, 0.39, 0.40]
    #thrs_magg = [0.45, 0.46, 0.47, 0.48, 0.49, 0.50]
    #rh.run_ensemble(thrs=thrs_magg, remove_nets=False)
    
    for thr in thrs_magg:
        rh.get_final_decision_from_ensemble(type='ensemble_magg', validation_thr=validation_thr, validation_metric='romad', ensemble_thr=thr)
        rh.get_report_excel(type='ensemble_magg', ensemble_thr=thr, report_name= experiment_name + ' - Report ensemble_magg - thr ' + str(thr) + ' - penalty')


    thrs_exclusive = [0.35, 0.40, 0.45, 0.50, 0.55, 0.60]    
    for thr in thrs_exclusive:
        rh.get_final_decision_from_ensemble(type='ensemble_el_exclusive', validation_thr=validation_thr, validation_metric='romad', perc_agreement=thr)
        rh.get_report_excel(type='ensemble_el_exclusive', perc_agreement=thr, report_name= experiment_name + ' - Report ensemble_el_exclusive - perc agreement ' + str(thr) + ' - penalty')