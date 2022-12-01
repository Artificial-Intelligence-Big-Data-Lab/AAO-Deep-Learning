import json
import numpy as np 

# CONFIGURATION ZONE 
loss_weight = w_array = np.ones((3,3))

loss_weight[2, 2] = 0 # 
loss_weight[1, 2] = 2 # 
loss_weight[0, 2] =6 # 

loss_weight[2, 1] = 2 # 
loss_weight[1, 1] = 0 # 
loss_weight[0, 1] = 2 # 

loss_weight[2, 0] = 10 # 
loss_weight[1, 0] = 4 # 
loss_weight[0, 0] = 0 # 

'''
' 17 walk
''' 
training_set = [['2000-02-01', '2009-01-30'],
                ['2000-08-01', '2009-07-31'],
                ['2001-02-01', '2010-01-31'],
                ['2001-08-01', '2010-07-30'],
                ['2002-02-01', '2011-01-31'],
                ['2002-08-01', '2011-07-31'],
                ['2003-01-02', '2012-01-31'],
                ['2003-08-01', '2012-07-31'],
                ['2004-02-01', '2013-01-31'],
                ['2004-08-01', '2013-07-31'], 
                ['2005-02-01', '2014-01-31'], # Fine esperimenti silvio
                ['2005-08-01', '2014-07-31'],
                ['2006-02-01', '2015-01-30'],
                ['2006-08-01', '2015-07-31'],
                ['2008-02-01', '2016-01-31'],
                ['2007-08-01', '2016-07-31'],
                ['2008-02-01', '2017-01-31']
                ]

validation_set = [['2009-02-01', '2009-07-31'],
                ['2009-08-02', '2010-01-31'],
                ['2010-02-01', '2010-07-30'],
                ['2010-08-01', '2011-01-31'],
                ['2011-02-01', '2011-07-31'],
                ['2011-08-01', '2012-01-31'],
                ['2012-02-01', '2012-07-31'],
                ['2012-08-01', '2013-01-31'],
                ['2013-02-01', '2013-07-31'],
                ['2013-08-01', '2014-01-31'],
                ['2014-02-02', '2014-07-31'],  # Fine esperimenti silvio
                ['2014-08-01', '2015-01-30'],
                ['2015-02-01', '2015-07-31'],
                ['2015-08-02', '2016-01-31'],
                ['2016-02-01', '2016-07-31'],
                ['2016-08-01', '2017-01-31'],
                ['2017-02-01', '2017-07-31'],   
        ]

test_set = [['2009-08-02', '2010-01-31'],
                ['2010-02-01', '2010-07-30'],
                ['2010-08-01', '2011-01-31'],
                ['2011-02-01', '2011-07-31'],
                ['2011-08-01', '2012-01-31'],
                ['2012-02-01', '2012-07-31'],
                ['2012-08-01', '2013-01-31'],
                ['2013-02-01', '2013-07-31'],
                ['2013-08-01', '2014-01-31'],
                ['2014-02-02', '2014-07-31'],
                ['2014-08-01', '2015-01-30'], # Fine esperimenti silvio
                ['2015-02-01', '2015-07-31'],
                ['2015-08-01', '2016-01-31'],
                ['2016-02-01', '2016-07-31'],
                ['2016-08-01', '2017-01-31'],
                ['2017-02-01', '2017-07-31'],
                ['2017-08-01', '2018-01-31'],
        ]


''' 
1 walk
training_set = [['2000-02-01', '2010-01-31']]
validation_set = [['2010-02-01', '2012-01-31']]
test_set = [['2012-02-01', '2019-08-31']]
'''

'''
' 4 walk
'''
training_set = [['2000-02-01', '2009-01-30'],
                ['2000-08-01', '2009-07-31'],
                ['2001-02-01', '2010-01-31'],
                ['2001-08-01', '2010-07-30']
            ]

validation_set = [['2009-02-01', '2009-07-31'],
                ['2009-08-02', '2010-01-31'],
                ['2010-02-01', '2010-07-30'],
                ['2010-08-01', '2011-01-31'] 
        ]

test_set = [['2009-08-02', '2010-01-31'],
            ['2010-02-01', '2010-07-30'],
            ['2010-08-01', '2011-01-31'],
            ['2011-02-01', '2011-07-31']
            ]


iperparameters = { 
                'experiment_name': '015 - 28gennaio - ripetizione 012', 
                'epochs' : 1200, # n° epoche per il training
                'number_of_nets': 30, # n° di reti da utilizzare per il training l'ensemble
                'save_pkl': False, # salva un file di history del training 
                'save_model_history': False, # booleano per salvare il modello ogni tot epoche
                'bs': 500, # dimensione batch size
                'init_lr': 0.001, # dimensione learning rate
                'return_multiplier': 50, #25 dax, moltiplicatore per convertire i punti di mercato in $
                'loss_function': 'w_categorical_crossentropy', # sparse_categorical_crossentropy | w_categorical_crossentropy
                'loss_weight': loss_weight.tolist(), # pesi della custom loss,
                'validation_thr': 15, # l'intorno da utilizzare per selezionare l'epoca migliore in validation
                'training_set': training_set,
                'validation_set': validation_set,
                'test_set': test_set,
                'input_images_folders': ['merge/merge_sp500_cet/gadf/delta/'],
                'input_datasets': ['sp500_cet'],
                'predictions_dataset': 'sp500_cet',
                'predictions_images_folder': 'merge/merge_sp500_cet/gadf/delta/',
                'input_shape': (40,40,3)
                }

with open('../experiments/' + iperparameters['experiment_name'] + '/log.json', 'w') as json_file:
        json.dump(iperparameters, json_file, indent=4)