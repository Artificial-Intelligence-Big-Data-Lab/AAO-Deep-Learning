import pandas as pd 
import numpy as np 

filename_list = []
for i in range (0, 34): 
    filename_list.append('GADF_walk_' + str(i) + '.csv')

experiments =  ['3_months_delta_experiment_1', '3_months_delta_experiment_2', '3_months_delta_experiment_3', '3_months_delta_experiment_4', '3_months_delta_experiment_5']

f=open('lista_reti_utilizzate.txt','a')
for index_experiment, experiment in enumerate(experiments): 
    for index_walk, file in enumerate(filename_list):
        path = '../experiments/' + experiment + '/results/datasets/validation/columns/'

        columns = pd.read_csv(path + file).columns.values
        columns = np.delete(columns, 0)

        

        f.write("Run n° " + str(index_experiment+1) + " Walk " + str(index_walk+1) + " - Used nets: " + str(len(columns)) + ' - ')
        for column in columns: 
            f.write(column + ',')
        f.write("\n")
        

        #print("Run n° " + str(index_experiment+1) + " Walk " + str(index_walk+1) + " - Used columns: " + str(len(columns)) + ' - ' + str(columns))
    #print("\n\n\n")
    #df[df.columns[df.columns.isin(columns)]]
f.close()