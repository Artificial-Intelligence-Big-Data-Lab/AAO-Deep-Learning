from flask import Flask
from flask import request
import pandas as pd
from classes.Measures import Measures
from classes.Utils import df_date_merger

app = Flask(__name__)

@app.route("/")
def hello():
    walk = 0#request.args.get('walk')
    net = 0#request.args.get('net')
    experiment_name = '009 - 18gennaio - replica exp 004 fullbatch' #request.args.get('experiment_name')
    set_type = 'validation' #request.args.get('type')

    df = pd.read_csv('/media/unica/HDD 9TB Raid0 - 1/experiments/' + experiment_name + '/predictions/predictions_during_training/' + str(set_type) + '/walk_' + str(walk) + '/' + 'net_' + str(net) + '.csv')

    # mergio con le label, così ho un subset del df con le date che mi servono e la predizione 
    df_merge_with_label = df_date_merger(df=df, columns=['date_time', 'delta_next_day', 'close', 'open'], dataset='sp500_cet')

    ls_returns = []
    lh_returns = []
    sh_returns = []

    ls_romads = []
    lh_romads = []
    sh_romads = []

    ls_mdds = []
    lh_mdds = []
    sh_mdds = []

    longs_precisions = []
    shorts_precisions = []
    longs_coverage = []
    shorts_coverage = []
    longs_poc = []
    shorts_poc = []


    long_operations = []
    short_operations = []
    hold_operations = []

    label_coverage = Measures.get_delta_coverage(delta=df_merge_with_label['delta_next_day'].tolist())

    # calcolo il return per un epoca
    for epoch in range(1, 30): 
        y_pred = df_merge_with_label['epoch_' + str(epoch)].tolist()
        delta = df_merge_with_label['delta_next_day'].tolist()

        ls_equity_line, ls_global_return, ls_mdd, ls_romad, ls_i, ls_j  = Measures.get_equity_return_mdd_romad(y_pred=y_pred, delta=delta, multiplier=50, type='long_short')
        lh_equity_line, lh_global_return, lh_mdd, lh_romad, lh_i, lh_j  = Measures.get_equity_return_mdd_romad(y_pred=y_pred, delta=delta, multiplier=50, type='long_only')
        sh_equity_line, sh_global_return, sh_mdd, sh_romad, sh_i, sh_j  = Measures.get_equity_return_mdd_romad(y_pred=y_pred, delta=delta, multiplier=50, type='short_only')

        long, short, hold, general = Measures.get_precision_count_coverage(y_pred=y_pred, delta=delta)
        
        #long_poc, short_poc = Measures.get_precision_over_coverage(y_pred=y_pred, delta=delta)

        ls_returns.append(ls_global_return)
        lh_returns.append(lh_global_return)
        sh_returns.append(sh_global_return)

        ls_romads.append(ls_romad)
        lh_romads.append(lh_romad)
        sh_romads.append(sh_romad)

        ls_mdds.append(ls_mdd)
        lh_mdds.append(lh_mdd)
        sh_mdds.append(sh_mdd)

        longs_precisions.append(long['precision'])
        shorts_precisions.append(short['precision'])

        longs_coverage.append(long['coverage'])
        shorts_coverage.append(short['coverage'])

        #longs_poc.append(long_poc)
        #shorts_poc.append(short_poc)

        long_operations.append(long['count'])
        short_operations.append(short['count'])
        hold_operations.append(hold['count'])

    return {'ls_returns': ls_returns}
if __name__ == '__main__':
     app.run(host="192.167.149.145", port='8080')