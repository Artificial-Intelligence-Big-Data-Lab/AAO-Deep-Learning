from classes.ResultsHandler import ResultsHandler

validation_thr = 15

experiments = [
        #'015 - 28gennaio - ripetizione 012',
        #'001 - ExpSalva_5_3_15gennaio',
        #'002 - ExpSalva_5_3_15gennaio',
        #'003 - pesiloss3 - 4walk - 16gennaio',
        #'004 - pesiloss3 - 16gennaio',
        #'005 - pesiloss5 - 16gennaio',
        #'006 - 17gennaio',
        #'007 - 17gennaio',
        #'008 - 18gennaio',
        #'010 - 21gennaio',
        #'011 - 22gennaio - class weight',
        #'012 - prova loss - primo',
        ##'013 - prova loss - secondo',
        #'016 - Class weight pro-short', 
        #'017 - Class weight pro-short - loss pesante',
        #'018 - Class weight 4-1-1 - loss bilanciata soft', 
        #'019 - Class weight 8-1-1 - loss bilanciata soft', 
        #'021 - ripetizione 020 corretto',
        '027 - ripetizione 026',
        '026 - BIG - test lungo',
        '025 - BIG rilanciato - validation singolo, loss 15, no class_w - 90 reti',
        '024 - BIG corretto - validation singolo, loss 15, no class_w - 100 reti',
        '022 - diminuita class weight',
    ]

for experiment_name in experiments:
    rh = ResultsHandler(experiment_name=experiment_name)

    thrs_magg = [0.33, 0.34, 0.35, 0.36, 0.37, 0.38, 0.39, 0.40]
    #thrs_magg = [0.41, 0.42, 0.43, 0.44, 0.45]
    #thrs_magg = [0.46, 0.47, 0.48, 0.49, 0.50]
    rh.run_ensemble(thrs=thrs_magg, remove_nets=False)
    
    for thr in thrs_magg:
        rh.get_final_decision_from_ensemble(type='ensemble_magg', validation_thr=validation_thr, validation_metric='romad', ensemble_thr=thr)
        rh.get_report_excel(type='ensemble_magg', ensemble_thr=thr, report_name= experiment_name + ' - Report ensemble_magg - thr ' + str(thr) + ' - penalty')


    #thrs_exclusive = [0.35, 0.40, 0.45, 0.50, 0.55, 0.60]    
    #for thr in thrs_exclusive:
    #    rh.get_final_decision_from_ensemble(type='ensemble_el_exclusive', validation_thr=validation_thr, validation_metric='romad', perc_agreement=thr)
    #    rh.get_report_excel(type='ensemble_el_exclusive', perc_agreement=thr, report_name= experiment_name + ' - Report ensemble_el_exclusive - perc agreement ' + str(thr) + ' - penalty')