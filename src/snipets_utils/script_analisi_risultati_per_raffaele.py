import pandas as pd 
import numpy as np 

df_walk = pd.read_csv("C:/Users/andre/Desktop/CSV RANDOM GUESSING/post ensemble/1.0/random_guess_all_walks.csv")
df_label = pd.read_csv("C:/Users/andre/Documents/GitHub/PhD-Market-Nets/datasets/debug_diego/sp500_label_curent_day.csv")

df = pd.merge(df_walk, df_label, how='inner', on='date_time')


df_corretti= df[(df.ensemble == df.label)]


correct = len(np.where(df['ensemble'] == df['label'])[0])
count_coverage = len(np.where(df['ensemble'] != 0)[0])

if count_coverage > 0:
    accuracy =  (correct / count_coverage) * 100

totale_operazioni = df.shape[0]

coverage = (count_coverage * 100) / totale_operazioni
#long_totali = len(df[(df.ensemble == 1)])
#short_totali = len(df[(df.ensemble == -1)])

#long_corretti = len(df_corretti[(df_corretti.ensemble == 1)])
#short_corretti = len(df_corretti[(df_corretti.ensemble == -1)])

#accuracy_long = (long_corretti / correct) * 100
#accuracy_short = (short_corretti / correct) * 100



print("Accuracy globale: " + str(accuracy))
print("Coverage globale: " + str(coverage))

#print("Totale operazioni long: " + str(long_totali) + " - Totale operazioni short: " + str(short_totali) + " -  Numero totale di operazioni: " + str(totale_operazioni))
#print("Operzioni corrette: " + str(correct) + " di cui " + str(long_corretti) + " e " + str(short_corretti) + " short.")

#print("Sul totale delle operazioni corrette (" + str(correct) + ") abbiamo: ")
#print("Percentuale di long corrette: " + str(accuracy_long))
#print("Percentuale di short corrette: " + str(accuracy_short))

#print("/nReturn totale: " + str(return_totale))
#print("Return derivato dalle long: " + str(return_by_long))
#print("Return derivato dalle short: " + str(return_by_short))
