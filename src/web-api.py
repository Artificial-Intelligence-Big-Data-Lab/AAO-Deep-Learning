from flask import Flask
from flask import request
import pandas as pd
from classes.Measures import Measures
from classes.Market import Market
from classes.ResultsHandler import ResultsHandler
from classes.Utils import df_date_merger
from flask_cors import CORS, cross_origin
from flask import jsonify 
import json 

app = Flask(__name__)
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'

@app.route("/get_market_bh", methods=['POST'])
@cross_origin()
def get_market_bh():
    
    r = request.form

    dataset = r['dataset']
    freq = r['freq']
    start_date = r['start_date']
    end_date = r['end_date']
    multiplier = int(r['multiplier'])
    penalty = int(r['penalty'])
    stop_loss = int(r['stop_loss'])

    try:
        m = Market(dataset=dataset)

        df = m.group(freq=freq)

        m = Market.get_df_by_data_range(df=df.copy(), start_date=start_date, end_date=end_date)
        m['date_time'] = m['date_time'].dt.strftime('%Y-%m-%d')

        bh_quity_line, bh_global_return, bh_mdd, bh_romad, bh_i, bh_j = Measures.get_return_mdd_romad_bh(close=m['close'].tolist(), multiplier=multiplier)

        bh_2_equity_line, bh_2_global_return, bh_2_mdd, bh_2_romad, bh_2_i, bh_2_j = Measures.get_equity_return_mdd_romad(df=m.copy(), multiplier=multiplier, type='bh_long', penalty=penalty, stop_loss=stop_loss, delta_to_use='delta_current_day')


        return {
                'bh': {
                    'date_time': m['date_time'].tolist(),
                    'equity_line': m['close'].tolist(),
                    'return': bh_global_return, 
                    'mdd': bh_mdd, 
                    'romad': bh_romad
                },
                'bh_2': {
                    'date_time': m['date_time'].tolist(),
                    'equity_line': bh_2_equity_line.tolist(),
                    'return': bh_2_global_return, 
                    'mdd': bh_2_mdd, 
                    'romad': bh_2_romad
                }
        }
    except:
            return {'error': 'Qualcosa è andato storto'}


@app.route("/get_period_subset", methods=['POST'])
@cross_origin()
def get_period_subset():
    
    r = request.form

    experiment_name = r['experiment_name']
    hdd = r['id_hdd']
    sp_net = r['net']
    sp_walk = r['walk']
    sp_epoch = r['epoch']
    start_date = r['start_date']
    end_date = r['end_date']
    set_type = r['set']
    penalty = r['penalty']
    stop_loss = r['stop_loss']

    path_hdd = ''
    if "path_hdd" in r:
        path_hdd = r['path_hdd']
    else:
        path_hdd = ''

    try:
        rh = ''
        if path_hdd != '':
            rh = ResultsHandler(experiment_name=experiment_name, experiment_path=path_hdd)
        else: 
            rh = ResultsHandler(experiment_name=experiment_name)

        net_json = rh.get_result_for_walk_net(start_date=start_date, end_date=end_date, index_walk=int(sp_walk), net=int(sp_net), epoch=int(sp_epoch), penalty=int(penalty), stop_loss=int(stop_loss), set_type=set_type)
        return net_json 
    except:
        return {'error': 'Qualcosa è andato storto'}
    

@app.route("/get_swipe_result", methods=['POST'])
@cross_origin()
def get_swipe_result():
    
    r = request.form

    experiment_name = r['experiment_name']
    hdd = r['id_hdd']

    ensemble_type = r['ensemble_type']
    thrs_ensemble_magg = json.loads(r['thrs_ensemble_magg'])
    thrs_ensemble_exclusive = json.loads(r['thrs_ensemble_exclusive'])
    thrs_ensemble_elimination = json.loads(r['thrs_ensemble_elimination'])
    epoch_selection_policy = r['epoch_selection_policy']
    decision_folder = r['decision_folder']
    stop_loss = r['stop_loss']
    penalty = r['penalty']

    path_hdd = ''
    if "path_hdd" in r:
        path_hdd = r['path_hdd']
    else:
        path_hdd = ''

    try:
        rh = ''
        if path_hdd != '':
            rh = ResultsHandler(experiment_name=experiment_name, experiment_path=path_hdd)
        else: 
            rh = ResultsHandler(experiment_name=experiment_name)

        result = rh.get_result_swipe(type=ensemble_type, thrs_ensemble_exclusive=thrs_ensemble_exclusive, thrs_ensemble_magg=thrs_ensemble_magg,
                                thrs_ensemble_elimination=thrs_ensemble_elimination, 
                                epoch_selection_policy=epoch_selection_policy, decision_folder=decision_folder, stop_loss=int(stop_loss), penalty=int(penalty))
        return result
    except:
        return {'error': 'Qualcosa è andato storto in get_swipe_result()'}


@app.route("/get_report_detail", methods=['POST'])
@cross_origin()
def get_report_detail():
    
    r = request.form

    experiment_name = r['experiment_name']
    hdd = r['id_hdd']
    ensemble_type = r['ensemble_type']
    thr = r['thr']
    epoch_selection_policy = r['epoch_selection_policy']
    decision_folder = r['decision_folder']
    stop_loss = int(r['stop_loss'])
    penalty = int(r['penalty'])

    rh = ''
    if "path_hdd" in r:
        rh = ResultsHandler(experiment_name=experiment_name, experiment_path=r['path_hdd'])
    else:
        rh = ResultsHandler(experiment_name=experiment_name)

    try:
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
            'general_info': general_info
        }
        return result
        
    except:
        return {'error': 'Qualcosa è andato storto in get_report_detail()'}


'''
' VIX
'''
@app.route("/get_report_swipe_vix", methods=['POST'])
@cross_origin()
def get_report_swipe_vix():
    r = request.form

    experiment_name = r['experiment_name']
    stop_loss = int(r['stop_loss'])
    penalty = int(r['penalty'])
    thrs = json.loads(r['thrs'])

    rh = ''
    if "path_hdd" in r:
        rh = ResultsHandler(experiment_name=experiment_name, experiment_path=r['path_hdd'])
    else:
        rh = ResultsHandler(experiment_name=experiment_name)
    
    result = rh.get_result_swipe_vix(thrs=thrs, stop_loss=stop_loss, penalty=penalty)

    return result


@app.route("/get_report_detail_vix", methods=['POST'])
@cross_origin()
def get_report_detail_vix():
    
    r = request.form

    experiment_name = r['experiment_name']
    thr = r['thr']
    stop_loss = int(r['stop_loss'])
    penalty = int(r['penalty'])

    rh = ''
    if "path_hdd" in r:
        rh = ResultsHandler(experiment_name=experiment_name, experiment_path=r['path_hdd'])
    else:
        rh = ResultsHandler(experiment_name=experiment_name)

    try:
        ls_results, l_results, s_results, bh_results, bh_intraday_results, general_info = rh.get_results_vix(thr=thr, stop_loss=stop_loss, penalty=penalty)

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
            'general_info': general_info
        }
        return result
        
    except:
        return {'error': 'Qualcosa è andato storto in get_report_detail()'}



if __name__ == '__main__':
    app.run(host="192.167.149.145", port='8080')
    #app.run(host="127.0.0.1", port='8080', debug=True)
