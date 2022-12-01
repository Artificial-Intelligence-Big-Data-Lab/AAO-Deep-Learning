from classes.ClassifierHandler import ClassifierHandler

'''
# Walk 2 anni
'''
training_set = [
        #['2000-01-01', '2009-12-31'], 
        #['2000-01-01', '2011-12-31'], 
        #['2000-01-01', '2013-12-31'],  
        #['2000-01-01', '2015-12-31'],  
        #['2000-01-01', '2017-12-31']

        ['2003-10-01', '2009-12-31'], 
        ['2003-10-01', '2011-12-31'], 
        ['2003-10-01', '2013-12-31'],  
        ['2003-10-01', '2015-12-31'],  
        ['2003-10-01', '2017-12-31']
 ]

validation_set = [
        ['2010-01-01', '2010-12-31'], 
        ['2012-01-01', '2012-12-31'], 
        ['2014-01-01', '2014-12-31'], 
        ['2016-01-01', '2016-12-31'], 
        ['2018-01-01', '2018-12-31'] 

 ]

test_set = [
        ['2011-01-01', '2012-12-31'],  
        ['2013-01-01', '2014-12-31'], 
        ['2015-01-01', '2016-12-31'], 
        ['2017-01-01', '2018-12-31'],
        ['2019-01-01', '2020-07-31']
]


'''
# Walk 1 anno
training_set_1_anno = [
        ['2000-01-01', '2009-12-31'], 
        ['2000-01-01', '2010-12-31'], 
        ['2000-01-01', '2011-12-31'],  
        ['2000-01-01', '2012-12-31'],  
        ['2000-01-01', '2013-12-31'],
        ['2000-01-01', '2014-12-31'], 
        ['2000-01-01', '2015-12-31'], 
        ['2000-01-01', '2016-12-31'],  
        ['2000-01-01', '2017-12-31'],  
        ['2000-01-01', '2018-12-31']
 ]

validation_set_1_anno = [
        ['2010-01-01', '2010-12-31'],  
        ['2011-01-01', '2011-12-31'],  
        ['2012-01-01', '2012-12-31'], 
        ['2013-01-01', '2013-12-31'], 
        ['2014-01-01', '2014-12-31'],
        ['2015-01-01', '2015-12-31'],
        ['2016-01-01', '2016-12-31'],  
        ['2017-01-01', '2017-12-31'], 
        ['2018-01-01', '2018-12-31'], 
        ['2019-01-01', '2019-12-31'],

 ]

test_set_1_anno = [
        ['2011-01-01', '2011-12-31'],  
        ['2012-01-01', '2012-12-31'], 
        ['2013-01-01', '2013-12-31'], 
        ['2014-01-01', '2014-12-31'],
        ['2015-01-01', '2015-12-31'],
        ['2016-01-01', '2016-12-31'],  
        ['2017-01-01', '2017-12-31'], 
        ['2018-01-01', '2018-12-31'], 
        ['2019-01-01', '2019-12-31'],
        ['2020-01-01', '2020-07-31']
]
'''

#thrs_binary_labeling = [-1.1, -0.9, -0.7, -0.5, -0.4, -0.3, -0.2, -0.1, 0.1, 0.2, 0.3, 0.4]
thrs_binary_labeling = [0.4]

for thr_binary_labeling in thrs_binary_labeling:
        iperparameters = { 
                        'training_set': training_set,
                        'validation_set': validation_set,
                        'test_set': test_set,
                        'predictions_dataset': 'sp500_cet',
                        'balance_binary': True,
                        'thr_binary_labeling': thr_binary_labeling, #binary
                        'dataset_path': 'Json Dataset Labeling ' + str(thr_binary_labeling)
        }

        ch = ClassifierHandler(iperparameters=iperparameters)
        #percs = [0, 2, 10, 50, 100, 200, 800]
        percs = [0]
        #for i, e in enumerate(training_set):

        #for i in [0, 1, 2]:
        #        ch.create_json(dataset='sp500_cet', index_walk=i, percs=percs, classifier='xgboost')

        # RICORDATI DI CAMBIARE LE PATH DENTRO IL METODO
        #ch.create_csv(dataset='sp500_cet', index_walk=i, percs=percs, classifier='xgboost', sample="20 sample 1h")
        #ch.create_csv(dataset='sp500_cet', index_walk=i, percs=percs, classifier='xgboost', sample="96 sample 5min")

        print("thr labeling:", thr_binary_labeling)
        ch.create_plot(scale_pos_weight_list=[3, 2, 1, 0.8, 0.5, 0.4, 0.3, 0.2, 0.1, 0.05, 0.02], perc=0, sample="20 sample 1h", plot_name='avg_labeling'+ str(thr_binary_labeling))

#ch.get_results_bh()

#ch.create_plot(scale_pos_weight_list=[0.5, 0.4], perc=0, sample="20 sample 1h")

# RICORDATI DI CAMBIARE LE PATH DENTRO IL METODO
#ch.calculate_results(dataset='sp500_cet', index_walk=0, percs=percs, sample="20 sample 1h")
#ch.calculate_results(dataset='sp500_cet', index_walk=1, percs=percs, sample="20 sample 1h")
#ch.calculate_results(dataset='sp500_cet', index_walk=2, percs=percs, sample="20 sample 1h")


#ch.calculate_results(dataset='sp500_cet', index_walk=0, percs=percs, sample="96 sample 5min")
#ch.calculate_results(dataset='sp500_cet', index_walk=1, percs=percs, sample="96 sample 5min")
#ch.calculate_results(dataset='sp500_cet', index_walk=2, percs=percs, sample="96 sample 5min")

#ch.calculate_results(dataset='sp500_cet', index_walk=2, percs=percs, sample="227 sample 5min")
#ch.run(dataset='sp500_cet', index_walk=0, percs=[0], classifier='xgboost')


#ch.run_walks(dataset='sp500_cet', classifier='xgboost')

#ch.logic_or()