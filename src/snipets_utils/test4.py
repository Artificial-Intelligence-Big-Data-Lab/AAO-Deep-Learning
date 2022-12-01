import pandas as pd 
import numpy as np 
from classes.Market import Market
from classes.ResultsHandler import ResultsHandler
from classes.Downloader import Downloader

#from classes.VggHandler import VggHandler

import time
import pandas as pd

d = Downloader()
d.run(dataset='msft')
d.run(dataset='aapl')
input("Done")


#experiments = ['Example 1']
#experiments = ['Example 2']
experiments = ['Example 3']

for experiment in experiments: 
    rh = ResultsHandler(experiment_name=experiment)

    #for walk in range(0, 9): # example 1 2
    for walk in range(0, 1): # example 3
        #example 1
        # example 2 (0, 100)
        #example 3 (0, 300) 
        #for net in range(0, 30): # example 1
        #for net in range(0, 100): # example 2
        for net in range(0, 300): # example 3
            print("Generating single net for walk", walk, "net:", net)
            rh.generate_single_net_json(index_walk=walk, index_net=net, type='validation', penalty=0, stop_loss=0)
            rh.generate_single_net_json(index_walk=walk, index_net=net,  type='test', penalty=0, stop_loss=0)
    
    print("Generating AVG for", experiment)
    rh.generate_avg_json(type='validation')
    rh.generate_avg_json(type='test')