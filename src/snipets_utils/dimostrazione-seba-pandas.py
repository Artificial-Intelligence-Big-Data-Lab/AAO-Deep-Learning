import pandas as pd
import time

list = range(1, 1 000 000)

df = pd.DataFrame()

df['colonna'] = list

print(df)

start = time.time()
df['colonna'] = df['colonna'] + 1
end = time.time()
print("ETA: " + "{:.3f}".format(end-start))



start = time.time()
#for index, row in df.iterrows():
for index in list: 
    #df.iloc[index]['colonna'] = row + 1
    df.at[index-1, 'colonna'] = df.at[index-1, 'colonna'] +2 
end = time.time()
print("ETA: " + "{:.3f}".format(end-start))


