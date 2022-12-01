from classes.VggHandler import VggHandler
from classes.ResultsHandler import ResultsHandler
from classes.Market import Market

# import os
# os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID";
# os.environ["CUDA_VISIBLE_DEVICES"]="1";

training_set = [
                ['2000-02-01', '2009-01-31'],
                ['2000-05-01', '2009-04-30'],

                ['2000-08-01', '2009-07-31'],
                ['2000-11-01', '2009-10-31'],

                ['2001-02-01', '2010-01-31'],
                ['2001-05-01', '2010-04-30'],

                ['2001-08-01', '2010-07-31'],
                ['2001-11-01', '2010-10-31'],

                ['2002-02-01', '2011-01-31'],
                ['2002-05-01', '2011-04-30'],

                ['2002-08-01', '2011-07-31'],
                ['2002-11-01', '2011-10-31'],

                ['2003-02-01', '2012-01-31'],
                ['2003-05-01', '2012-04-30'],


                ['2003-08-01', '2012-07-31'],
                ['2003-11-01', '2012-10-31'],


                ['2004-02-01', '2013-01-31'],
                ['2004-05-01', '2013-04-30'],

                ['2004-08-01', '2013-07-31'], 
                ['2004-11-01', '2013-10-31'],

                ['2005-02-01', '2014-01-31'], 
                ['2005-05-01', '2014-04-30'], # Fine esperimenti silvio
                ##
                ['2005-08-01', '2014-07-31'],
                ['2005-11-01', '2014-10-31'],

                ['2006-02-01', '2015-01-31'],
                ['2006-05-01', '2015-04-30'],

                ['2006-08-01', '2015-07-31'],
                ['2006-11-01', '2015-10-31'],
                
                ['2008-02-01', '2016-01-31'],
                ['2008-05-01', '2016-04-30'],

                ['2007-08-01', '2016-07-31'],
                ['2007-11-01', '2016-10-31'],

                ['2008-02-01', '2017-01-31'],
                ['2008-05-01', '2017-04-30'],
                ]

validation_set = [
                ['2009-02-01', '2009-04-30'],
                ['2009-05-01', '2009-07-31'],

                ['2009-08-01', '2009-10-31'],
                ['2009-11-01', '2010-01-31'],

                ['2010-02-01', '2010-04-30'],
                ['2010-05-01', '2010-07-31'],

                ['2010-08-01', '2010-10-31'],
                ['2010-11-01', '2011-01-31'],

                ['2011-02-01', '2011-04-30'],
                ['2011-05-01', '2011-07-31'],

                ['2011-08-01', '2011-10-31'],
                ['2011-11-01', '2012-01-31'],

                ['2012-02-01', '2012-04-30'],
                ['2012-05-01', '2012-07-31'],

                ['2012-08-01', '2012-10-31'],
                ['2012-11-01', '2013-01-31'],

                ['2013-02-01', '2013-04-30'],
                ['2013-05-01', '2013-07-31'],

                ['2013-08-01', '2013-10-31'],
                ['2013-11-01', '2014-01-31'],

                ['2014-02-01', '2014-04-30'],  
                ['2014-05-01', '2014-07-31'],# Fine esperimenti silvio
                ##
                ['2014-08-01', '2014-10-31'],
                ['2014-11-01', '2015-01-31'],

                ['2015-02-01', '2015-04-30'],
                ['2015-05-01', '2015-07-31'],

                ['2015-08-01', '2015-10-31'],
                ['2015-11-01', '2016-01-31'],

                ['2016-02-01', '2016-04-30'],
                ['2016-05-01', '2016-07-31'],

                ['2016-08-01', '2016-10-31'],
                ['2016-11-01', '2017-01-31'],

                ['2017-02-01', '2017-04-30'], 
                ['2017-05-01', '2017-07-31']  
        ]

test_set = [
                ['2009-05-01', '2009-07-31'],
                ['2009-08-01', '2009-10-31'],

                ['2009-11-01', '2010-01-31'],
                ['2010-02-01', '2010-04-30'],

                ['2010-05-01', '2010-07-31'],
                ['2010-08-01', '2010-10-31'],

                ['2010-11-01', '2011-01-31'],
                ['2011-02-01', '2011-04-30'],

                ['2011-05-01', '2011-07-31'],
                ['2011-08-01', '2011-10-31'],

                ['2011-11-01', '2012-01-31'],
                ['2012-02-01', '2012-04-30'],

                ['2012-05-01', '2012-07-31'],
                ['2012-08-01', '2012-10-31'],

                ['2012-11-01', '2013-01-31'],
                ['2013-02-01', '2013-04-30'],

                ['2013-05-01', '2013-07-31'],
                ['2013-08-01', '2013-10-31'],

                ['2013-11-01', '2014-01-31'],
                ['2014-02-01', '2014-04-30'],

                ['2014-05-01', '2014-07-31'],
                ['2014-08-01', '2014-10-31'], # Fine esperimenti silvio

                ['2014-11-01', '2015-01-31'],
                ['2015-02-01', '2015-04-30'],

                ##
                ['2015-05-01', '2015-07-31'],
                ['2015-08-01', '2015-10-31'],

                ['2015-11-01', '2016-01-31'],
                ['2016-02-01', '2016-04-30'],

                ['2016-05-01', '2016-07-31'],
                ['2016-08-01', '2016-10-31'],

                ['2016-11-01', '2017-01-31'],
                ['2017-02-01', '2017-04-30'],

                ['2017-05-01', '2017-07-31'],
                ['2017-08-01', '2017-10-31'],

                #['2017-11-01', '2018-01-31']
]



for i in range(1,3):
        vgg = VggHandler()

        vgg.net_config(epochs=200, number_of_nets=20)

        vgg.run_initialize(dataset='sp500',
                input_folder='delta_experiment/gadf/delta/',
                training_set=training_set,
                validation_set=validation_set,
                test_set=test_set,
                input_shape=(40,40,3),
                output_folder='3_months_delta_experiment_' + str(i))

        #vgg.run()

        results_handler = ResultsHandler(experiment_name='3_months_delta_experiment_' + str(i), dataset='sp500')

        #print("Calculatin training predictions and results...")
        #vgg.get_predictions(set_type='training')
        #results_handler.generate_ensemble_and_plots(set_type='training')
        #results_handler.calculate_return(set_type='training')

        #print("Calculatin validation predictions and results...")
        #vgg.get_predictions(set_type='validation')
        #results_handler.generate_ensemble_and_plots(set_type='validation')
        #results_handler.calculate_return(set_type='validation')

        print("Calculatin test predictions and results...")
        vgg.get_predictions(set_type='test')
        results_handler.generate_ensemble_and_plots(set_type='test')
        results_handler.calculate_return(set_type='test')