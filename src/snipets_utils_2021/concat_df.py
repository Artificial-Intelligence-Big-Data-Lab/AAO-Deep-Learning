import pandas as pd 

df0 = pd.read_csv('C:/Users/Utente/Desktop/Json Dataset/walk 2 anni/csv/20 sample 1h/walk_0_perc_0_test.csv')
df1 = pd.read_csv('C:/Users/Utente/Desktop/Json Dataset/walk 2 anni/csv/20 sample 1h/walk_1_perc_0_test.csv')
df2 = pd.read_csv('C:/Users/Utente/Desktop/Json Dataset/walk 2 anni/csv/20 sample 1h/walk_2_perc_0_test.csv')
df3 = pd.read_csv('C:/Users/Utente/Desktop/Json Dataset/walk 2 anni/csv/20 sample 1h/walk_3_perc_0_test.csv')
df4 = pd.read_csv('C:/Users/Utente/Desktop/Json Dataset/walk 2 anni/csv/20 sample 1h/walk_4_perc_0_test.csv')


df_final = pd.DataFrame()

df_final = pd.concat([df0, df1])
df_final = pd.concat([df_final, df2])
df_final = pd.concat([df_final, df3])
df_final = pd.concat([df_final, df4])

df_final.to_csv('C:/Users/Utente/Desktop/Json Dataset/walk 2 anni/csv/20 sample 1h/pwalk_all_perc_0.csv', index=False, header=True)