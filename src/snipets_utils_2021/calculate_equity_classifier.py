import numpy as np 
import pandas as pd 
from classes.Measures import Measures
from classes.Market import Market
from classes.Utils import df_date_merger_binary 
import matplotlib.pyplot as plt
from cycler import cycler
import xlsxwriter
# provo a ricalcolare le equity partendo dai file di decisioni dei classificatori

#df_intra = pd.read_csv('C:/Users/Utente/Documents/GitHub/PhD-Market-Nets/classifier_bhintra/predictions_walk_all_perc_bhintra.csv')
df_20 = pd.read_csv('C:/Users/Utente/Documents/GitHub/PhD-Market-Nets/classifier_json_dataset_20_sample_1h_walk_1anno/predictions_walk_all_perc_0.csv')
df_96 = pd.read_csv('C:/Users/Utente/Documents/GitHub/PhD-Market-Nets/classifier_json_dataset_96_sample_5min_walk_1anno/predictions_walk_all_perc_0.csv')
df_logic_or = pd.read_csv('C:/Users/Utente/Documents/GitHub/PhD-Market-Nets/classifier_json_dataset_OR_20_96_sample_1anno/predictions_walk_all_perc_LOGIC_OR.csv')


#df_intra = df_intra.rename(columns={'predictions': "decision"})
df_20 = df_20.rename(columns={'predictions': "decision"})
df_96 = df_96.rename(columns={'predictions': "decision"})

#df_intra = df_date_merger_binary(df=df_intra, thr=-0.5, columns=['open', 'close', 'high', 'low', 'delta_current_day'], dataset='sp500_cet')
df_20 = df_date_merger_binary(df=df_20, thr=-0.5, columns=['open', 'close', 'high', 'low', 'delta_current_day'], dataset='sp500_cet')
df_96 = df_date_merger_binary(df=df_96, thr=-0.5, columns=['open', 'close', 'high', 'low', 'delta_current_day'], dataset='sp500_cet')
df_logic_or = df_date_merger_binary(df=df_logic_or, thr=-0.5, columns=['open', 'close', 'high', 'low', 'delta_current_day'], dataset='sp500_cet')

#df_intra = df_intra.sort_values(by=['date_time'])
df_20 = df_20.sort_values(by=['date_time'])
df_96 = df_96.sort_values(by=['date_time'])
df_logic_or = df_logic_or.sort_values(by=['date_time'])

#equity_line_intra, global_return_intra, mdd_intra, romad_intra, i_intra, j_intra = Measures.get_equity_return_mdd_romad(df=df_intra, multiplier=50, type='long_short', penalty=25, stop_loss=1000, delta_to_use='delta_current_day')
equity_line_20, global_return_20, mdd_20, romad_20, i_20, j_20 = Measures.get_equity_return_mdd_romad(df=df_20, multiplier=50, type='long_short', penalty=25, stop_loss=1000, delta_to_use='delta_current_day')
equity_line_96, global_return_96, mdd_96, romad_96, i_96, j_96 = Measures.get_equity_return_mdd_romad(df=df_96, multiplier=50, type='long_short', penalty=25, stop_loss=1000, delta_to_use='delta_current_day')
equity_line_logic_and, global_return_logic_and, mdd_logic_and, romad_logic_and, i_logic_and, j_logic_and = Measures.get_equity_return_mdd_romad(df=df_logic_or, multiplier=50, type='long_short', penalty=25, stop_loss=1000, delta_to_use='delta_current_day')

#bh_quity_line, bh_global_return, bh_mdd, bh_romad, bh_i, bh_j = Measures.get_return_mdd_romad_bh(close=df_intra['close'].tolist(), multiplier=50)

#first_close = bh_quity_line[0]
#bh_quity_line = [i - first_close for i in bh_quity_line]

print("Strategy\t Romad\t Return\t\t MDD")
#print("BH Intra\t", "{:.2f}".format(romad_intra), "\t", global_return_intra, "\t", mdd_intra)
print("20 Sample\t", "{:.2f}".format(romad_20), "\t", global_return_20, "\t", mdd_20)
print("96 Sample\t", "{:.2f}".format(romad_96), "\t", global_return_96, "\t", mdd_96)
print("OR 20-96\t", "{:.2f}".format(romad_logic_and), "\t", global_return_logic_and, "\t", mdd_logic_and)

#dates = df_intra['date_time'].tolist()

'''
plt.figure(figsize=(18, 12))
plt.style.use("ggplot")
plt.rcParams['axes.prop_cycle'] = cycler(color='bgrcmyk')
plt.xlabel("Dates")
plt.ylabel("Equity $")
plt.plot(range(0, len(dates)), bh_quity_line, label="B&H Classico")
plt.plot(range(0, len(dates)), equity_line_intra, label="B&H Intraday")
plt.plot(range(0, len(dates)), equity_line_20, label="XBG 20 Sample")
plt.plot(range(0, len(dates)), equity_line_96, label="XGB 96 Sample")
plt.legend(loc="upper left")
plt.show()
'''
workbook = xlsxwriter.Workbook('forplot.xlsx')
worksheet = workbook.add_worksheet('Report')
#worksheet.write_column('AA1', dates)
#worksheet.write_column('AB1', bh_quity_line)
#worksheet.write_column('AC1', equity_line_intra)
worksheet.write_column('AD1', equity_line_20)
worksheet.write_column('AE1', equity_line_96)
worksheet.write_column('AF1', equity_line_logic_and)
workbook.close()