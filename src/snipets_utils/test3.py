import pandas as pd 
import numpy as np 
from classes.Market import Market

def print_figo(label, n, anno):
    short = label.count(0)
    hold = label.count(1)
    long = label.count(2)
    tot = len(label)

    long_perc = "{:.4f}".format(long/tot)
    short_perc = "{:.4f}".format(short/tot)
    hold_perc = "{:.4f}".format(hold/tot)

    print("Walk:", n, "(test:", anno, ") | Long:", long_perc, "Hold:", hold_perc, "Short:", short_perc)

sp500 = Market(dataset='sp500')
sp500 = sp500.get_label(freq='1d', thr=0)


walk_0 = Market.get_df_by_data_range(df=sp500.copy(), start_date='2000-01-01', end_date='2009-12-31')
walk_1 = Market.get_df_by_data_range(df=sp500.copy(), start_date='2000-01-01', end_date='2010-12-31')
walk_2 = Market.get_df_by_data_range(df=sp500.copy(), start_date='2000-01-01', end_date='2011-12-31')
walk_3 = Market.get_df_by_data_range(df=sp500.copy(), start_date='2000-01-01', end_date='2012-12-31')
walk_4 = Market.get_df_by_data_range(df=sp500.copy(), start_date='2000-01-01', end_date='2013-12-31')
walk_5 = Market.get_df_by_data_range(df=sp500.copy(), start_date='2000-01-01', end_date='2014-12-31')
walk_6 = Market.get_df_by_data_range(df=sp500.copy(), start_date='2000-01-01', end_date='2015-12-31')
walk_7 = Market.get_df_by_data_range(df=sp500.copy(), start_date='2000-01-01', end_date='2016-12-31')
walk_8 = Market.get_df_by_data_range(df=sp500.copy(), start_date='2000-01-01', end_date='2017-12-31')

label_0 = walk_0['label_next_day'].tolist()
label_1 = walk_1['label_next_day'].tolist()
label_2 = walk_2['label_next_day'].tolist()
label_3 = walk_3['label_next_day'].tolist()
label_4 = walk_4['label_next_day'].tolist()
label_5 = walk_5['label_next_day'].tolist()
label_6 = walk_6['label_next_day'].tolist()
label_7 = walk_7['label_next_day'].tolist()
label_8 = walk_8['label_next_day'].tolist()

print_figo(label=label_0, n=0, anno='2011')
print_figo(label=label_1, n=1, anno='2012')
print_figo(label=label_2, n=2, anno='2013')
print_figo(label=label_3, n=3, anno='2014')
print_figo(label=label_4, n=4, anno='2015')
print_figo(label=label_5, n=5, anno='2016')
print_figo(label=label_6, n=6, anno='2017')
print_figo(label=label_7, n=7, anno='2018')
print_figo(label=label_8, n=8, anno='2019')