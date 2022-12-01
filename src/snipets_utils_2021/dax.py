from classes.Market import Market

dax = Market(dataset='dax_cet')

dax = dax.group(freq='1d')


dax.to_csv('C:/Users/Utente/Desktop/Test Script/dax_daily.csv', header=True, index=True)
print(dax)