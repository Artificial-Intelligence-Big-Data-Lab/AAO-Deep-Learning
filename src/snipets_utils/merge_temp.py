import os
import pandas as pd 

names = [
        'GADF_walk_0.csv', 
        'GADF_walk_1.csv', 
        'GADF_walk_2.csv', 
        'GADF_walk_3.csv', 
        #'GADF_walk_4.csv', 
        'GADF_walk_5.csv', 
        'GADF_walk_6.csv', 
        'GADF_walk_7.csv', 
        'GADF_walk_8.csv', 
        'GADF_walk_9.csv', 
        'GADF_walk_10.csv' 
        ]


main_path = 'C:\\Users\\andre\\Desktop\\merge_differents_thr\\run 5\\' 

df = pd.DataFrame()

for name in names:
    f = pd.read_csv(main_path + name)

    df = pd.concat([df, f], axis=0)

df.to_csv(main_path + 'walk_merge.csv', header=True, index=False)
