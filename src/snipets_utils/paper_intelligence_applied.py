import numpy as np 
import pandas as pd 
from  classes.Utils import df_date_merger
from  classes.Market import Market



dfff = pd.read_csv('C:/Users/Utente/Desktop/CSV Anselmo Applied Intelligence/CSV  CNN/mrk/training.csv')
for i in range(0, 1000): 
    dfff = dfff.rename(columns={dfff.columns[i]: 'prediction_' + str(i)})
print(dfff)
input()

dfff.to_csv('C:/Users/Utente/Desktop/CSV Anselmo Applied Intelligence/CSV  CNN/mrk/training.csv', header=True, index=False)
input("Salvato")


def get_range(list, bottom, top):
    for i in range(bottom, top):
        list.append('epoch_' + str(i+1))


DATASET = 'mrk'

'''
# SP500 training
'''
#start_date = '2004-01-01'
#end_date = '2008-12-31'
# sp500 test
#start_date = '2009-01-01'
#end_date = '2015-12-31'


'''
# AAPL - MSFT - JPM - MRK
'''

# valid
#start_date = '2003-01-01'
#end_date = '2006-12-31'

#test
#start_date = '2007-01-01'
#end_date = '2012-12-31'




#base_path = "D:PhD-Market-Nets/experiments/Applied Intelligence - Sinica 2/predictions/predictions_during_training/test/walk_0/"

# SP500
#base_path = "D:PhD-Market-Nets/experiments/Applied Intelligence - Sinica 3/predictions/predictions_during_training/validation/walk_0/" # sinica 3 training
#base_path = "D:PhD-Market-Nets/experiments/Applied Intelligence - Sinica 3/predictions/predictions_during_training/test/walk_0/" # sinica 3 test

# AAPL
#base_path = "D:PhD-Market-Nets/experiments/Applied Intelligence - AAPL 2/predictions/predictions_during_training/test/walk_0/"

# MSFT
#base_path = "D:PhD-Market-Nets/experiments/Applied Intelligence - MSFT 2/predictions/predictions_during_training/test/walk_0/"

# JPM
#base_path = "D:PhD-Market-Nets/experiments/Applied Intelligence - JPM/predictions/predictions_during_training/test/walk_0/"


# MRK
base_path = "D:PhD-Market-Nets/experiments/Applied Intelligence - MRK/predictions/predictions_during_training/validation/walk_0/"
base_path = "D:PhD-Market-Nets/experiments/Applied Intelligence - MRK/predictions/predictions_during_training/test/walk_0/"

'''
# SINICA 2
''
net_0 = pd.read_csv(base_path + 'net_0.csv') # 300 - 450 
net_1 = pd.read_csv(base_path + 'net_3.csv') # 150 - 400
net_2 = pd.read_csv(base_path + 'net_7.csv') # 350 - 500
net_3 = pd.read_csv(base_path + 'net_12.csv') # 250 - 500
net_4 = pd.read_csv(base_path + 'net_14.csv') # 300 - 500
'''

'''
# SINICA 3
''
net_0 = pd.read_csv(base_path + 'net_4.csv') # 0 - 500 
net_1 = pd.read_csv(base_path + 'net_14.csv') # 30 - 130
net_2 = pd.read_csv(base_path + 'net_16.csv') # 350 - 450
net_3 = pd.read_csv(base_path + 'net_24.csv') # 250 - 450
'''

'''
' MRK
'''
net_0 = pd.read_csv(base_path + 'net_0.csv') # 250 - 450 
net_1 = pd.read_csv(base_path + 'net_3.csv') # 100 - 200
net_2 = pd.read_csv(base_path + 'net_5.csv') # 50 - 150
net_3 = pd.read_csv(base_path + 'net_13.csv') # 50 - 200
net_4 = pd.read_csv(base_path + 'net_15.csv') # 150 a 200 
net_5 = pd.read_csv(base_path + 'net_16.csv') # 250 a 450
net_6 = pd.read_csv(base_path + 'net_19.csv') # 100 a 200
net_7 = pd.read_csv(base_path + 'net_26.csv') # 50 a 150


'''
# MSFT
''
net_0 = pd.read_csv(base_path + 'net_2.csv') #  150 - 450
net_1 = pd.read_csv(base_path + 'net_8.csv') #  100 - 250
net_2 = pd.read_csv(base_path + 'net_11.csv') # 50 - 400
net_3 = pd.read_csv(base_path + 'net_20.csv') # 100 - 200
net_4 = pd.read_csv(base_path + 'net_27.csv') # 400 - 500
'''

'''
# AAPL 
''
net_0 = pd.read_csv(base_path + 'net_0.csv') #  
net_1 = pd.read_csv(base_path + 'net_2.csv') # 
net_2 = pd.read_csv(base_path + 'net_3.csv') # 
net_3 = pd.read_csv(base_path + 'net_12.csv') # 
net_4 = pd.read_csv(base_path + 'net_16.csv') # 
net_5 = pd.read_csv(base_path + 'net_28.csv') # 
'''

'''
# JPM 
''
net_0 = pd.read_csv(base_path + 'net_0.csv') #  300 - 400
net_1 = pd.read_csv(base_path + 'net_1.csv') # 250 - 500 
net_2 = pd.read_csv(base_path + 'net_7.csv') # 200 - 400
net_3 = pd.read_csv(base_path + 'net_19.csv') # 50 - 500
#net_4 = pd.read_csv(base_path + 'net_16.csv') # 
#net_5 = pd.read_csv(base_path + 'net_28.csv') # 
'''

date_list = net_0['date_time'].tolist()

#filtro a monte
net_0 = net_0.drop(columns=['Unnamed: 0', 'date_time'])
net_1 = net_1.drop(columns=['Unnamed: 0', 'date_time'])
net_2 = net_2.drop(columns=['Unnamed: 0', 'date_time'])
net_3 = net_3.drop(columns=['Unnamed: 0', 'date_time'])
net_4 = net_4.drop(columns=['Unnamed: 0', 'date_time'])
net_5 = net_5.drop(columns=['Unnamed: 0', 'date_time'])
net_6 = net_6.drop(columns=['Unnamed: 0', 'date_time'])
net_7 = net_7.drop(columns=['Unnamed: 0', 'date_time'])

range_0 = []
range_1 = []
range_2 = []
range_3 = []
range_4 = []
range_5 = []
range_6 = []
range_7 = []

'''
# sinica 2 
''
get_range(range_0, 300, 450)
get_range(range_1, 150, 400)
get_range(range_2, 350, 500)
get_range(range_3, 250, 500)
get_range(range_4, 300, 500)
#get_range(range_5, 300, 400)
'''

'''
# sinica 3
''
get_range(range_0, 0, 500)
get_range(range_1, 30, 130)
get_range(range_2, 350, 450)
get_range(range_3, 250, 450)
#get_range(range_5, 300, 400)
'''

'''
' MRK
'''
get_range(range_0, 250, 450) # 200
get_range(range_1, 100, 200) # 100
get_range(range_2, 50, 150) # 100
get_range(range_3, 50, 200) # 150
get_range(range_4, 150, 200) # 50
get_range(range_5, 250, 450) # 200
get_range(range_6, 100, 200) # 100
get_range(range_7, 50, 150) # 100


'''
' msft
''
get_range(range_0, 150, 450)
get_range(range_1, 100, 250)
get_range(range_2, 50, 400)
get_range(range_3, 100, 200)
get_range(range_4, 400, 500)
'''

'''
# aapl
''
get_range(range_0, 300, 450)
get_range(range_1, 150, 400)
get_range(range_2, 350, 500)
get_range(range_3, 250, 500)
get_range(range_4, 300, 500)
#get_range(range_5, 300, 400)
'''


'''
# JPM
''
get_range(range_0, 300, 400)
get_range(range_1, 250, 500)
get_range(range_2, 200, 400)
get_range(range_3, 50, 500)
'''


# filtro range seba
net_0 = net_0[range_0]
net_1 = net_1[range_1]
net_2 = net_2[range_2]
net_3 = net_3[range_3]
net_4 = net_4[range_4]
net_5 = net_5[range_5]
net_6 = net_6[range_6]
net_7 = net_7[range_7]


df_final = pd.DataFrame()

df_final = pd.concat([df_final, net_0], axis=1)
df_final = pd.concat([df_final, net_1], axis=1)
df_final = pd.concat([df_final, net_2], axis=1)
df_final = pd.concat([df_final, net_3], axis=1)
df_final = pd.concat([df_final, net_4], axis=1)
df_final = pd.concat([df_final, net_5], axis=1)
df_final = pd.concat([df_final, net_6], axis=1)
df_final = pd.concat([df_final, net_7], axis=1)

df_final['date_time'] = date_list

df_final = df_date_merger(df=df_final,thr_hold=0.3, columns=['open', 'close', 'delta_next_day'], dataset=DATASET)
df_final = df_final.drop(columns=['label_next_day', 'label_current_day'])

#df_final = Market.get_df_by_data_range(df=df_final, start_date=start_date, end_date=end_date)
#df_final = df_final.drop(columns=['index'])


#df_final.to_csv('C:/Users/Utente/Desktop/CSV Anselmo Applied Intelligence/CSV  CNN/mrk/training.csv', header=True, index=False)
df_final.to_csv('C:/Users/Utente/Desktop/CSV Anselmo Applied Intelligence/CSV  CNN/mrk/test.csv', header=True, index=False)
