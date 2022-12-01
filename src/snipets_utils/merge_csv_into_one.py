import os
import pandas as pd 

names = [
        'walk_0.csv', 
        'walk_1.csv', 
        'walk_2.csv', 
        'walk_3.csv', 
        'walk_4.csv', 
        'walk_5.csv', 
        'walk_6.csv', 
        'walk_7.csv', 
        'walk_8.csv', 
        'walk_9.csv', 
        'walk_10.csv', 
        'walk_11.csv', 
        'walk_12.csv', 
        'walk_13.csv', 
        'walk_14.csv', 
        'walk_15.csv',
        'walk_16.csv'
        ]

main_path = 'C:/Users/Utente/Documents/GitHub/PhD-Market-Nets/experiments/Esperimenti Vacanze/Esperimento 16 Walks 6mesi , SP500_CET, SGD BS 300, Labeling hold 0.3, Salvatore Capodanno (seba)/predictions_ensemble/test/0.4/ensemble_offset/' 
#main_path = '~/PhD-Market-Nets/experiments/delta_experiment_third_run/results/ensemble/' + set + '/' 

df = pd.DataFrame()

for name in names:
    f = pd.read_csv(main_path + name)

    df = pd.concat([df, f], axis=0)

df.to_csv(main_path + 'walk_global.csv', header=True, index=False)
