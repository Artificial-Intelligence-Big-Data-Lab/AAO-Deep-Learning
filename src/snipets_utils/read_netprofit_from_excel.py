'''
from xlrd import open_workbook

book = open_workbook("C:/Users/Utente/Desktop/0.6-SP500-STOPLOSS1000.xlsx")
sheet = book.sheet_by_index(1) #If your data is on sheet 1

cum_profit = []
#...

for row in range(4, 2463): #start from 1, to leave out row 0
    cum_profit.append(sheet.cell(row, 10)) #extract from first col


print(cum_profit[0])
print(cum_profit[1])
print(cum_profit[2])
print(cum_profit[3])
print(cum_profit[4])
print(cum_profit[5])
'''

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score
from cycler import cycler
from classes.Market import Market

df = pd.read_excel('C:/Users/Utente/Desktop/0.6-SP500-STOPLOSS1000.xlsx', sheet_name=1) # can also index sheet by name or fetch all sheets
my_list = df['Unnamed: 10'].tolist()

my_list = [x for x in my_list if str(x) != 'nan']

my_list = my_list[1:]

my_list = [(i / 50) + 983.50 for i in my_list]

df_sp = Market(dataset='sp500_cet')
sp500 = df_sp.group(freq='1d')

sp500 = Market.get_df_by_data_range(df=sp500, start_date='2009-08-02', end_date='2015-01-29')
close = sp500['close'].tolist()
date = sp500['date_time'].tolist()

x = np.arange(0, len(my_list))
close = close[0:len(my_list)]
date = date[0:len(my_list)]

plt.figure(figsize=(10,8))
#plt.style.use('seaborn')
plt.xlabel("Dates")
plt.ylabel("Cumulative Profit")
plt.plot(date, my_list)
plt.plot(date, close)
plt.ylim(top=3000, bottom=500) 
plt.show()