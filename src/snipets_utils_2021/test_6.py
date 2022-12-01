import pandas as pd
df1 = pd.read_csv('D:/PhD-Market-Nets/experiments/130 - SP500 walk 1 anno sino al 2020 con nuovi dati pesi bilanciati/predictions/final_decisions/ensemble_exclusive/senza-rimozione-reti/long_only/decisions_ensemble_exclusive_0.1.csv')
df2 = pd.read_csv('D:/PhD-Market-Nets/experiments/130 - SP500 walk 1 anno sino al 2020 con nuovi dati pesi bilanciati/predictions/final_decision_alg4/ensemble_exclusive/senza-rimozione-reti/long_only/decisions_ensemble_exclusive_0.1.csv')


print("ALG3:", df1.shape)
print("ALG4:", df2.shape)


df1 = df1['date_time'].tolist()
df2 = df2['date_time'].tolist()

diff1 = list(set(df2) - set(df1))
diff1.sort()

diff2 = list(set(df1) - set(df2))
diff2.sort()

print("Diff:", len(diff1), "\nDays:", diff1)
print("Diff2:", len(diff2), "\nDays:", diff2)