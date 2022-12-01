from classes.Market import Market 
from classes.Utils import create_folder
market = Market(dataset='sp500_cet')

print('Grouping Datasets...')


# Raggruppo il dataset per 1h, 1d, 1w, 1m
one_h = market.group(freq='1h', nan=False)
four_h = market.group(freq='4h', nan=False)
eight_h = market.group(freq='8h', nan=False)
one_d = market.group(freq='1d', nan=False)


training_set = [
                ['2000-01-01', '2011-12-31'],
                ['2000-01-01', '2012-12-31'],
                ['2000-01-01', '2013-12-31'],
                ['2000-01-01', '2014-12-31']
            ]

for idx, date in enumerate(training_set):
    print(idx, date[0], date[1])
    create_folder('C:/Users/Utente/Desktop/Diego/walk_' + str(idx))

    one_h_ = Market.get_df_by_data_range(df=one_d.copy(), start_date=date[0], end_date=date[1])
    four_h_ = Market.get_df_by_data_range(df=four_h.copy(), start_date=date[0], end_date=date[1])
    eight_h_ = Market.get_df_by_data_range(df=eight_h.copy(), start_date=date[0], end_date=date[1])
    one_d_ = Market.get_df_by_data_range(df=one_d.copy(), start_date=date[0], end_date=date[1])
    

    one_h_.to_csv('C:/Users/Utente/Desktop/Diego/walk_' + str(idx) + '/1h.csv', header=True, index=False)
    four_h_.to_csv('C:/Users/Utente/Desktop/Diego/walk_' + str(idx) + '/4h.csv', header=True, index=False)
    eight_h_.to_csv('C:/Users/Utente/Desktop/Diego/walk_' + str(idx) + '/8h.csv', header=True, index=False)
    one_d_.to_csv('C:/Users/Utente/Desktop/Diego/walk_' + str(idx) + '/1d.csv', header=True, index=False)

'''    
get_df_by_data_range
one_h.to_csv('C:/Users/Utente/Desktop/Diego/1h.csv', header=True, index=False)
four_h.to_csv('C:/Users/Utente/Desktop/Diego/4h.csv', header=True, index=False)
eight_h.to_csv('C:/Users/Utente/Desktop/Diego/8h.csv', header=True, index=False)
one_d.to_csv('C:/Users/Utente/Desktop/Diego/1d.csv', header=True, index=False)
'''