import json 

dataset_new = {}
dataset_old = {}

with open('C:/Users/Utente/Desktop/Swipe classificatori -0.5 96sample/json/is_1/walk_2_anni/96_sample/test_walk_0.json') as json_file:
    dataset_new = json.load(json_file)
    
with open('C:/Users/Utente/Desktop/json-csv risultati classificatori/walk 2 anni/classifier_json_dataset_96_sample_5min/sp500_walk_0_perc_0_test.json') as json_file:
    dataset_old = json.load(json_file)

datelist_new = dataset_new['date_time']
datelist_old = dataset_old['date_list']

x_new = dataset_new['x']
x_old = dataset_old['x']


for i, d in enumerate(dataset_new): 
    if datelist_new[i] != datelist_old[i]:
        print("[DATE] Differenza il", datelist_new[i], datelist_old[i])


for i, d in enumerate(x_new): 
    if x_new[i] != x_old[i]:
        print("[X] Differenza il", datelist_new[i], datelist_old[i])
