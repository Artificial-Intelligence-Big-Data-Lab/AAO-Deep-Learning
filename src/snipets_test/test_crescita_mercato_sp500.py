from classes.Market import Market
import numpy as np 

def get_df_by_data_range(df, start_date, end_date):
    # Search mask
    mask = (df['date_time'] >= start_date) & (df['date_time'] <= end_date)
    # Get the subset of sp500
    return df.loc[mask]

sp500 = Market(dataset='sp500')

df = sp500.get_label_current_day(freq='1d', columns=['open', 'close', 'delta', 'delta_percentage']).reset_index()

df = get_df_by_data_range(df=df, start_date='2010-01-01', end_date='2018-12-31')

positive = np.where(df['delta'] > 1)
negative = np.where(df['delta'] < 0)

count_positive = len(np.where(df['delta'] > 1)[0])
count_negative = len(np.where(df['delta'] < 1)[0])

sum_positive = df.loc[df['delta'] > 1, 'delta'].sum()
sum_negative = abs(df.loc[df['delta'] < 1, 'delta'].sum())

print("Crescita media in punti di mercato: " + str(sum_positive/count_positive))
print("Decrescita media in punti di mercato: " + str(sum_negative/count_negative))
