import numpy as np 
import pandas as pd 

from classes.Market import Market

'''
' Conto quante volte entro a comprare/vendere
'''
def get_count_coverage(df):
    return len(np.where(df[LABEL_TO_CHECk] != 0)[0])

'''
' Calcolo la coverage in percentuale, ovvero 
' quante volte sono entrato nel mercato. Se sono entrato
' 10 volte su 100, avrò lo 0.1 di coverage
'''
def get_coverage_perc(df):
    # Conto il numero di righe del DF (numero di giorni in cui faccio predizioni)
    size = df.shape[0]
    coverage = get_count_coverage(df)
    return (coverage * 100) / size


'''
' Conto quante volte ho fatto una long/short correttamente (-1, 1)
'''
def get_count_correct(df):
    return len(np.where(df[LABEL_TO_CHECk] == df['label'])[0])


'''
' Calcolo la % di accuracy sulle volte che sono entrato nel mercato
' quindi solamente se il valore è 1 o -1. Le hold (0) non le considero proprio
'''
def get_accuracy(df):
    correct = get_count_correct(df)
    count_coverage = get_count_coverage(df)

    if count_coverage > 0:
        return (correct / count_coverage) * 100
    else:
            return 0

'''
df = pd.read_csv('C:/Users/andre/Documents/GitHub/PhD-Market-Nets/experiments/multi_company_exp_multi_walk_6w/predictions/test/GADF_walk_0.csv')


for i in range(0, 20): 
    LABEL_TO_CHECk = 'net_' + str(i)
    accuracy = get_accuracy(df)

    print("Accuracy " + LABEL_TO_CHECk + ": " + str(accuracy))
'''
dataset_name = 'aapl'
sp500 = Market(dataset=dataset_name)

df = sp500.get_label_current_day(freq='1d', columns=['open', 'close'])

print(df)
true_label = len(np.where(df['label'] == 1)[0])
false_label = len(np.where(df['label'] == -1)[0])
all_samples = true_label + false_label

print("DF " + dataset_name + " with " + str(all_samples) + " samples.")
print("Giorni chiusi in positivo: " + str(true_label) + " [" + str((true_label*100)/all_samples) + " %]")
print("Giorni chiusi in negativo: " + str(false_label) + " [" + str((false_label*100)/all_samples) + " %]")