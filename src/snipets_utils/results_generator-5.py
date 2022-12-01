from classes.ResultsHandler import ResultsHandler

validation_thr = 15

experiments = [
        #'026 - BIG - test lungo',
        '001 - ExpSalva_5_3_15gennaio',
        '002 - ExpSalva_5_3_15gennaio',
        '003 - pesiloss3 - 4walk - 16gennaio',
        '026 - BIG - test lungo',
        '010 - 21gennaio',
        '011 - 22gennaio - class weight',
        '022 - diminuita class weight',
        '024 - BIG corretto - validation singolo, loss 15, no class_w - 100 reti',
        '025 - BIG rilanciato - validation singolo, loss 15, no class_w - 90 reti'

    ]
'''
experiments = [
        #'019 - Class weight 8-1-1 - loss bilanciata soft',
        '021 - ripetizione 020 corretto',
        '022 - diminuita class weight',
        '024 - BIG corretto - validation singolo, loss 15, no class_w - 100 reti',
        '025 - BIG rilanciato - validation singolo, loss 15, no class_w - 90 reti'
    ]
'''

for experiment_name in experiments:
    rh = ResultsHandler(experiment_name=experiment_name)

    #rh.run_ensemble(thrs=[0.35, 0.40], remove_nets=False)

    rh.get_final_decision_from_ensemble(type='ensemble_magg', validation_thr=validation_thr, validation_metric='romad', ensemble_thr=0.35)
    rh.get_final_decision_from_ensemble(type='ensemble_magg', validation_thr=validation_thr, validation_metric='romad', ensemble_thr=0.40)

    rh.get_final_decision_from_ensemble(type='ensemble_el_exclusive', validation_thr=validation_thr, validation_metric='romad', num_agreement=12)
    rh.get_final_decision_from_ensemble(type='ensemble_el_exclusive', validation_thr=validation_thr, validation_metric='romad', num_agreement=15)
    rh.get_final_decision_from_ensemble(type='ensemble_el_exclusive', validation_thr=validation_thr, validation_metric='romad', num_agreement=18)

    rh.get_report_excel(type='ensemble_magg', ensemble_thr=0.35, report_name= experiment_name + ' - Report ensemble_magg penalty - thr 0.35')
    rh.get_report_excel(type='ensemble_magg', ensemble_thr=0.40, report_name= experiment_name + ' - Report ensemble_magg penalty - thr 0.40')

    rh.get_report_excel(type='ensemble_el_exclusive', num_agreement=12, report_name= experiment_name + ' - Report ensemble_el_exclusive penalty - num agreement 12')
    rh.get_report_excel(type='ensemble_el_exclusive', num_agreement=15, report_name= experiment_name + ' - Report ensemble_el_exclusive penalty - num agreement 15')
    rh.get_report_excel(type='ensemble_el_exclusive', num_agreement=18, report_name= experiment_name + ' - Report ensemble_el_exclusive penalty - num agreement 18')