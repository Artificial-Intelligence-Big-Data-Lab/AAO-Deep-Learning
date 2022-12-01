import pandas as pd 
import numpy as np 
from classes.Market import Market

sp500 = Market(dataset='sp500')

sp500_daily = sp500.get_label_current_day(freq='1d', columns=['open', 'close', 'delta'])

PERIOD_SIZE = 3 # MESI

start_quarter = pd.to_datetime('2000-02-01').date()

latest_date = pd.to_datetime('2017-11-30').date()

training_set = []
validation_set = []
test_set = []

i = 1 

df = pd.DataFrame(columns=['index', 'trimestre', 'label trimestre', 'numero label positivi giornalieri', 'numero label negativi giornalieri', 'varianza close', 'valore medio close', 'close primo giorno', 'close fine trimestre', 'max close', 'min close', '(close[0]-low)/(high-low)', '(close[-1]-low)/(high-low)', 'val_test'])



print("Generating dates for training...")
while pd.to_datetime(start_quarter) <= pd.to_datetime(latest_date):

    end_quarter = (start_quarter + pd.DateOffset(months=PERIOD_SIZE)).date()

    print("[", i, "] Trimestre: ", start_quarter, " - ", end_quarter)

    subset_df = Market.get_df_by_data_range(df=sp500_daily, start_date=start_quarter, end_date=end_quarter)

    #label trimestre
    #print("primo: ", subset_df['close'].iloc[-1], " - secondo", subset_df['open'].iloc[0])
    if (subset_df['close'].iloc[-1] - subset_df['open'].iloc[0]).astype(int) > 0:
        label_quarter = 1
    else:
        label_quarter = -1

    #varianza close
    variance = subset_df['close'].var()
    #media close
    mean = subset_df['close'].mean()

    # close primo giorno
    close_first_day = subset_df['close'].iloc[0]

    #close ultimo giorno
    close_last_day = subset_df['close'].iloc[-1]

    # valore massimo close
    close_max = subset_df['close'].max()
    # valore min close
    close_min = subset_df['close'].min()

    # (close[1]-low)/(high-low) 
    rapp_1 = (close_first_day-close_min)/(close_max-close_min)
    # close-low/(high-low)
    rapp_2 = (close_last_day-close_min)/(close_max-close_min) 

    # numero label positivi giornalieri
    positive_delta = len(np.where(subset_df['label'] == 1)[0])
    # numero label negativi giornalieri
    negative_delta = len(np.where(subset_df['label'] == -1)[0])

    df = df.append({
                    'index': i,
                    'trimestre': str(start_quarter) + " - " + str(end_quarter), 
                    'label trimestre': label_quarter,
                    'numero label positivi giornalieri': positive_delta,
                    'numero label negativi giornalieri': negative_delta,
                    'varianza close': variance,
                    'valore medio close': mean,
                    'close primo giorno': close_first_day, 
                    'close fine trimestre': close_last_day, 
                    'max close': close_max, 
                    'min close': close_min, 
                    '(close[0]-low)/(high-low)': rapp_1, 
                    '(close[-1]-low)/(high-low)': rapp_2,
                    'val_test': variance / close_last_day
                    }, ignore_index=True)
    i = i+1
    start_quarter = (start_quarter + pd.DateOffset(months=PERIOD_SIZE)).date()

df = df.set_index('index')
df.to_csv('C:\\Users\\andre\\Documents\\GitHub\\PhD-Market-Nets\\quarter_details.csv', index=True, header=True)