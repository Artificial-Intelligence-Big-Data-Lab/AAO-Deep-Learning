import pandas as pd 
from classes.Utils import create_folder, df_date_merger_binary
from sklearn.metrics import confusion_matrix
from classes.Measures import Measures
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, balanced_accuracy_score

index_walk = 2

'''
df_20 = pd.read_csv('C:/Users/Utente/Desktop/CVS per And/20 sample 0.2/walk_' + str(index_walk) + '_perc_0_test.csv')
df_96 = pd.read_csv('C:/Users/Utente/Desktop/CVS per And/96 sample 0.05/walk_' + str(index_walk) + '_perc_0_test.csv')

df_final = pd.DataFrame()
list_20 = df_20['decision'].tolist()
list_96 = df_96['decision'].tolist()

list_final = []

for i, e in enumerate(list_20):
    if list_20[i] == 0 and list_96[i] == 0:
        list_final.append(0)
    else: 
        list_final.append(1)


df_final['date_time'] = df_20['date_time'].tolist()
df_final['date_time'] = pd.to_datetime(df_final['date_time'])
df_final['decision'] = list_final
df_final = df_final.sort_values(by=['date_time'])

df_final.to_csv('C:/Users/Utente/Desktop/CVS per And/OR_walk_' + str(index_walk) + '_perc_0.csv', header=True, index=False)


'''


path = 'C:/Users/Utente/Desktop/CVS per And/OR_walk_' + str(index_walk) + '_perc_0.csv'
perc = 0
print("\nWalk:", index_walk)
#print("Perc\t Delta Down %\t\t Label Down %\t\t Balanced Accuracy %\t\t Delta Por Down\t\t Label Por Down\t\t Delta AVG Precision\t\t Label Avg Precision\t\tOperazioni % / N° operazioni")
print("Perc\t Delta Down %\t\t Label Down %\t\t Balanced Accuracy %\t\t Delta AVG Precision\t\t Label Avg Precision\t\tOperazioni % / N° operazioni")

df = pd.read_csv(path)

df = df_date_merger_binary(df=df.copy(), thr=-0.5, columns=['delta_current_day', 'delta_next_day', 'open', 'close', 'high', 'low'], dataset='sp500_cet')

results = Measures.get_equity_return_mdd_romad(df=df.copy(), multiplier=50, type='long_only', penalty=25, stop_loss=1000, delta_to_use='delta_current_day', compact_results=True)

label_precisions = confusion_matrix(df['label_current_day'].tolist(), df['decision'].tolist(), normalize='pred', labels = [0, 1])
label_random_precision = Measures.get_binary_coverage(y=df['label_current_day'].tolist())

coverage = Measures.get_binary_coverage(y=df['decision'].tolist())

# Precision delta + random delta
delta_precision = Measures.get_binary_delta_precision(y=df['decision'].tolist(), delta=df['delta_current_day'].tolist(), delta_val=-25)

balanced_accuracy = balanced_accuracy_score(df['label_current_day'].tolist(), df['decision'].tolist())

print(perc, "\t", \
        round(delta_precision['down'], 2), "/", round(delta_precision['random_down'], 2), "\t\t",
        round(label_precisions[0][0] * 100, 2), "/", round(label_random_precision['down_perc'], 2), "\t\t",
        round(balanced_accuracy * 100, 2), "\t\t\t\t", 
        #round(((delta_precision['down'] / delta_precision['random_down']) - 1 ) * 100, 2), "\t\t",
        #round((((label_precisions[0][0] * 100) / label_random_precision['down_perc']) - 1 ) * 100, 2), "\t\t",
        round((delta_precision['down'] + delta_precision['up']) / 2, 2), "\t\t\t\t",
        round(((label_precisions[0][0] + label_precisions[1][1]) / 2) * 100, 2), "\t\t\t\t",
        coverage['down_perc'], "/", coverage['down_count'],
)
