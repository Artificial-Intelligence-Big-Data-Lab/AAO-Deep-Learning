import pandas as pd
from classes.Measures import Measures
from classes.Market import Market
from classes.ResultsHandler import ResultsHandler
from classes.Utils import df_date_merger

'''
experiment_name = '087 - Test univariate SP500 unico blocco 1h'
hdd = 0
thrs_ensemble_exclusive = [0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.80, 0.85, 0.90, 0.95, 1]
stop_loss = 1000
penalty = 25

#try:
rh = ResultsHandler(experiment_name=experiment_name)

    #result = rh.get_result_swipe(type=swipe_type, thrs_ensemble_magg=thrs_ensemble_magg, thrs_ensemble_exclusive=thrs_ensemble_exclusive, 
    #                            thrs_ensemble_elimination=thrs_ensemble_elimination, 
    #                            epoch_selection_policy=epoch_selection_policy, decision_folder=decision_folder, stop_loss=stop_loss, penalty=penalty)

result = rh.get_result_swipe(type='ensemble_exclusive', thrs_ensemble_exclusive=thrs_ensemble_exclusive,
                            epoch_selection_policy='long_only', decision_folder='test', stop_loss=int(stop_loss), penalty=int(penalty))
print(result)
'''

experiment_name = '087 - Test univariate SP500 unico blocco 1h'
hdd = 0
thr = 1
epoch_selection_policy = 'long_only'
ensemble_type = 'ensemble_exclusive'
stop_loss = 1000
penalty = 25
decision_folder = 'test_agl4'
    
rh = ResultsHandler(experiment_name=experiment_name)

if ensemble_type == 'ensemble_magg':
    ls_results, l_results, s_results, bh_results, bh_intraday_results, general_info = rh.get_results(ensemble_type=ensemble_type, epoch_selection_policy=epoch_selection_policy,
            thr_ensemble_magg=thr, stop_loss=stop_loss, penalty=penalty, decision_folder=decision_folder)

if ensemble_type == 'ensemble_elimination':
    ls_results, l_results, s_results, bh_results, bh_intraday_results, general_info = rh.get_results(ensemble_type=ensemble_type, epoch_selection_policy=epoch_selection_policy,
        thr_ensemble_elimination=thr, stop_loss=stop_loss, penalty=penalty, decision_folder=decision_folder)

if ensemble_type == 'ensemble_exclusive':
    ls_results, l_results, s_results, bh_results, bh_intraday_results, general_info = rh.get_results(ensemble_type=ensemble_type, epoch_selection_policy=epoch_selection_policy,
        thr_ensemble_exclusive=thr, stop_loss=stop_loss, penalty=penalty, decision_folder=decision_folder)

if ensemble_type == 'ensemble_exclusive_short':
    ls_results, l_results, s_results, bh_results, bh_intraday_results, general_info = rh.get_results(ensemble_type=ensemble_type, epoch_selection_policy=epoch_selection_policy,
        thr_ensemble_exclusive=thr, stop_loss=stop_loss, penalty=penalty, decision_folder=decision_folder)

result = {
    
    'ls_results': {
            'return': ls_results['return'],
            'mdd': ls_results['mdd'],
            'romad': ls_results['romad'], 
            'equity_line': ls_results['equity_line'].tolist(),
    },
    'lh_results': {
            'return': l_results['return'],
            'mdd': l_results['mdd'],
            'romad': l_results['romad'], 
            'equity_line': l_results['equity_line'].tolist(),
    },
    'sh_results': {
            'return': s_results['return'],
            'mdd': s_results['mdd'],
            'romad': s_results['romad'], 
            'equity_line': s_results['equity_line'].tolist(),
    },
    'bh_results': {
            'return': bh_results['return'],
            'mdd': bh_results['mdd'],
            'romad': bh_results['romad'],
            'equity_line' : [x for x in bh_results['equity_line']], 
    },
    'bh_2_results': {
            'return': bh_intraday_results['return'],
            'mdd': bh_intraday_results['mdd'],
            'romad': bh_intraday_results['romad'],
            'equity_line' : [x for x in bh_intraday_results['equity_line']],
    },
    #'general_info': general_info
}

result = {
    
    'ls_results': {
            'return': ls_results['return'],
            'mdd': ls_results['mdd'],
            'romad': ls_results['romad'], 
            'equity_line': ls_results['equity_line'].tolist(),
    }
    #'general_info': general_info
}
print(result)