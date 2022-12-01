import pandas as pd

def group(df, freq, nan=False):
    # Get first Open, Last Close, Highest value of "High" and Lower value of "Low", and sum(Volume)
    df['DateTime'] = pd.to_datetime(df['DateTime'])
    grouped = df.drop(['Date', 'Time'], axis=1).groupby(pd.Grouper(key='DateTime', freq=freq), sort=True).agg({
        'Open': 'first',
        'Close': 'last',
        'High': 'max',
        'Low': 'min',
        'Up': 'sum',
        'Down': 'sum'
    })

    grouped['Delta'] = (grouped['Close'] - grouped['Open']) / grouped['Open']
    
    if nan == True:
        return grouped
    else:
        return grouped.dropna()

# daily
#df = pd.read_csv('C:/Users/andre/Documents/GitHub/PhD-Market-Nets/datasets/single_company/daily_adjusted/MSFT.csv')
#result = df.loc[df['Close'] == df['Adj Close']]

#print(result.shape)

# 5 min
df = pd.read_csv('C:/Users/andre/Documents/GitHub/PhD-Market-Nets/datasets/single_company/MSFT_5MIN_With_DT.csv')

#df['DateTime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'])
#df.to_csv('C:/Users/andre/Documents/GitHub/PhD-Market-Nets/datasets/single_company/MSFT_5MIN_With_DT.csv', header=True, index=True)
grouped = group(df=df, freq='1d')
#df['delta'] = (df['Close'] - df['Open'] ) / df['Open']
#print(df)
#print(df.loc[df['Open'] == df['Close']])
print(grouped)
#print(df.loc[df['delta'] > 0.01])