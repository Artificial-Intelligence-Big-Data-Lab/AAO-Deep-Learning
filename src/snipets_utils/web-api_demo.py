from flask import Flask
from flask import request
import pandas as pd
from classes.Measures import Measures
from classes.Market import Market
from classes.ResultsHandler import ResultsHandler
from classes.Utils import df_date_merger
from flask_cors import CORS, cross_origin
#from flask_sslify import SSLify
from OpenSSL import SSL

app = Flask(__name__)
#sslify = SSLify(app) #ssl

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

    try:
        m = Market(dataset=dataset)

        df = m.group(freq=freq)

        m = Market.get_df_by_data_range(df=df, start_date=start_date, end_date=end_date)
        m['date_time'] = m['date_time'].dt.strftime('%Y-%m-%d')
        return {
                'date_time': m['date_time'].tolist(),
                'close': m['close'].tolist()
                }
    except:
        return {'error': 'Qualcosa e\' andato storto'}


@app.route("/get_period_subset", methods=['POST'])
@cross_origin()
def get_period_subset():
    hdd_path = ['sp500', 'gold'] # hawkeye patch

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

    #try:
    rh = ResultsHandler(experiment_name=experiment_name, additional_path=hdd_path[int(hdd)])

    net_json = rh.get_result_for_walk_net(start_date=start_date, end_date=end_date, index_walk=int(sp_walk), net=int(sp_net), epoch=int(sp_epoch), penalty=int(penalty), stop_loss=int(stop_loss), set_type=set_type)
    return net_json 
    #except:
    #    return {'error': 'Qualcosa e\' andato storto'}
    
    
'''    
context = SSL.Context(SSL.TLSv1_2_METHOD)
context.use_privatekey_file('/etc/letsencrypt/live/hawkeye.unica.it/privkey.pem')
context.use_certificate_chain_file('/etc/letsencrypt/live/hawkeye.unica.it/fullchain.pem')
context.use_certificate_file('/etc/letsencrypt/live/hawkeye.unica.it/cert.pem')
context = ('cert.crt', 'key.key')
'''

if __name__ == '__main__':
    #app.run(host="172.31.33.193", port='8080')
    #app.run(host="172.31.33.193", port=8080, threaded=True, ssl_context=context)
    app.run(host="172.31.33.193", port=8080, ssl_context=('/etc/letsencrypt/live/hawkeye.unica.it/fullchain.pem', '/etc/letsencrypt/live/hawkeye.unica.it/privkey.pem'))
    #app.run(host="3.130.0.231", port='8080')
    #app.run(host="127.0.0.1", port='8080', debug=True)