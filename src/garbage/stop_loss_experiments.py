import math
import numpy as np 
import pandas as pd
from sklearn import metrics
import matplotlib.pyplot as plt
from classes.Market import Market




'''
                             _     _                       
  / _|  _   _   _      ___  | |_  (_)   ___    _     ___ 
 | |_  | | | | | '_ \   / | | | | |  / _ \  | '_ \  / |
 |  _| | |_| | | | | | | (  | |_  | | | (_) | | | | | \ \
 |_|    \__,_| |_| |_|  \___|  \__| |_|  \___/  |_| |_| |___/
                                                             
'''

# if the slope is a +ve value --> increasing trend
# if the slope is a -ve value --> decreasing trend
# if the slope is a zero value --> No trend'
def get_trend_value(indexes, values, order=1):
  coeffs = np.polyfit(indexes, list(values), order)

  p = np.poly1d(coeffs)

  slope = coeffs[-2]

  return float(slope)

# Return the trend line of a time series
def get_trend_line(indexes, values, order=1):
  coeffs = np.polyfit(indexes, list(values), order)

  p = np.poly1d(coeffs)

  return p(indexes)

# p(trand) = 1/(2*pi) * arctg(trand) + 1
def get_p(trend_value):
  return ( 1 / (2  * math.pi)) * math.atan(trend_value) + 1

# Var_0, definiamo v(Var) = 1/(2*pi) * arctg(Var - Var_0) + 1
def get_v(trend_value, variance):
  x = 50 # Soglia personale di tolleranza, ho provato con 80 ma la stopp loss rimane troppo bassa
  return ( 1 / (2 * math.pi)) * math.atan( np.sign(trend_value) * (variance - x)) + 1

# sl = sl_0 * p(t) * v(Var)
def get_stop_loss(trend_value, variance, base_stop_loss=1000): 
    return base_stop_loss * get_p(trend_value) * get_v(trend_value, variance)

# Generate custom Y vector with a selected accuracy
def generate_accuracy_vector(y_truth, percentage):
  y = list(y_truth)

  for i in range(len(y)):

    if metrics.accuracy_score(y_truth, y)<= percentage:
      break

    if(y[i] ==1):
      y[i]= 0
    else:
      y[i] =1

  print("[DEBUG generate_accuracy_vector] Accuracy: ", metrics.accuracy_score(y, y_truth))
  return y
  
'''
  ____            _                                           _                                   _     _                 
 |  _ \     _  | |_     _     _  ___      _   _    (_)  _     _   _   _       _  | |_  (_)   ___    _   
 | | | |  / _` | | |  / _` |   | '_  _ \   / _ | | '_ \  | | | '_ \  | | | | | '_ \   / _` | | | | |  / _ \  | '_ \ 
 | |_| | | (_| | | |_  | (_| |   | | | | | | | (_| | | | | | | | | |_) | | |_| | | |_) | | (_| | | |_  | | | (_) | | | | |
 |____/   \,_|  \__|  \__,_|   |_| |_| |_|  \__,_| |_| |_| |_| | ./   \,_| | ./   \,_|  \__| |_|  \___/  |_| |_|
                                                                 |_|             |_|                                      
'''

dataset = 'sp500'

start_date = pd.to_datetime('2009-08-02') # start test set 
end_date = pd.to_datetime('2015-01-30') # end test set

accuracy = 0.5 # custom accuracy used for backtesting

stop_loss_base = 1000 # base stop loss

market = Market(dataset=dataset)

market_df_1d = market.group(freq='1d', nan=False)

market_df = Market.get_df_until_data(df=market_df_1d, end_date=end_date)

market_df['date_time'] = pd.to_datetime(market_df['date_time'])

df_stop_loss = pd.DataFrame(columns=['date_time', 'stop_loss'])

print("Start calculating stop loss for each day...")
for index, row in market_df[::-1].iterrows():

  if row['date_time'] < start_date: 
    break
  
  market_small = market_df.iloc[index-30:index] # get the previous 30 days
  
  indexes = market_small.index.to_list() # get the index of the close values
  values = market_small.close.to_list() # get the close values as list

  # get the trend values of previous 30 days
  trend_value = get_trend_value(indexes=indexes, values=values, order=1)

  # get the variance 
  variance = market_small['close'].var()

  # calculate the custom stop loss for this period
  stop_loss = get_stop_loss(trend_value=trend_value, variance=variance)

  # append the values on new DF
  data = {"date_time": row['date_time'], 'stop_loss': stop_loss}
  df_stop_loss = df_stop_loss.append(data, ignore_index=True)

# Get the DF with the label of the next day
label_current_day = market.get_label_current_day(freq='1d', columns=['open', 'close'])
# Get the DF with data range used on test set backtesting
df_backtest = Market.get_df_by_data_range(df=label_current_day, start_date=start_date, end_date=end_date)

# Get the final DF with date_time, Label, stop_loss, open and close value foreach day
df_label_stop_loss = df_backtest.merge(df_stop_loss, on='date_time')

print(df_label_stop_loss)

temp = df_label_stop_loss[['date_time', 'label', 'stop_loss']]
temp['stop_loss'] = temp['stop_loss'].astype(int)
temp['label'] = temp['label'].astype(int)
temp.to_csv('label_custom_sl.csv', header=True, index=False)

# Get the trend line for all dataset
trend_line = get_trend_line(df_label_stop_loss.index.to_list(), values=df_label_stop_loss.close.to_list(), order=1)

'''
# Plot close 
plt.figure(figsize=(15, 12))

plt.subplot(2, 1, 1)
plt.plot(df_label_stop_loss['date_time'], df_label_stop_loss['close'], color="b")
plt.plot(df_label_stop_loss['date_time'], trend_line, color="orange")
plt.xlabel('Date')
plt.ylabel('Close')

plt.subplot(2, 1, 2)
plt.plot(df_label_stop_loss['date_time'], df_label_stop_loss['stop_loss'])
plt.xlabel('Date')
plt.ylabel('Stop loss')

plt.show()
'''

'''
' Syntatic data test
' Genero una stop loss sintetica in caso di mercato crescente o decrescente
''
indexes = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10] 
values = [1, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000] 
#values.reverse() # inverto la tendenza di mercato, da forte crescita a forte decrescita
trend_value, trend_line = trend_line(indexes=indexes, values=values, order=1)
fig, ax = plt.subplots()
ax.plot(indexes, values)
ax.plot(indexes, trend_line,"r--" )
ax.set(xlabel='Date', ylabel='close', title= dataset + ' Market Value')
ax.grid()
plt.show()
'''


'''
' Blocco per generare vettori con accuracy prestabilite
''
label_df = market.get_label_current_day(freq='1d')
label_df = Market.get_df_by_data_range(label_df, start_date, end_date).set_index('date_time')

label_df['label'] = generate_accuracy_vector(label_df['label'].to_list(), accuracy)
label_df.to_csv('C:\\Users\\andre\\Desktop\\predizioni_test.csv', index=True, header=True)
'''