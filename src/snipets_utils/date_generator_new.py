import pandas as pd 

# USO COME PUNTO DI PARTENZA IL TRAINING

TRAINING_SET_SIZE = 8 # YEARS
VALIDATION_SET_SIZE = 10 # MONTHS
TEST_SET_SIZE = 10 # MONTHS

NUMERO_WALK = 22
ANCORATO = True

# DATA DI PARTENZA 
START_DATE_CURRENT = pd.to_datetime('2000-01-01').date()

# IN CASO DI TRAINING ANCORATO 
START_TRAINING = pd.to_datetime('2000-01-01').date()


# genero la coppia data inizio - data fine per il training set usando l'offset  specificato
def get_training_set(start_date):
    last_date = (start_date + pd.DateOffset(years=TRAINING_SET_SIZE) - pd.DateOffset(days=1)).date()

    if ANCORATO == True: 
        return [str(START_TRAINING), str(last_date)]
    else: 
        return [str(START_DATE), str(last_date)]
# genero la coppia data inizio - data fine per il validation set usando l'offset  specificato
def get_validation_set(date):

    start_date = (date + pd.DateOffset(years=TRAINING_SET_SIZE)).date()
    last_date = (start_date + pd.DateOffset(months=VALIDATION_SET_SIZE)  - pd.DateOffset(days=1)).date()
    
    return [str(start_date), str(last_date)]

# genero la coppia data inizio - data fine per il test set usando l'offset  specificato
def get_test_set(date):
    start_date = (date + pd.DateOffset(years=TRAINING_SET_SIZE) + pd.DateOffset(months=VALIDATION_SET_SIZE)).date()
    last_date = (start_date + pd.DateOffset(months=TEST_SET_SIZE)  - pd.DateOffset(days=1)).date()

    return [str(start_date), str(last_date)]

def start_log(training_set, validation_set, test_set):
        f=open('C:/Users/Utente/Desktop/test/dates.py','a')
        
        f.write("training_set = [")
        for index_walk in range(len(training_set)):    
            f.write("['" + training_set[index_walk][0] + "', '" + training_set[index_walk][1] + "'], \n ")
        f.write("]\n\n")


        f.write("validation_set = [")
        for index_walk in range(len(validation_set)):    
            f.write("['" + validation_set[index_walk][0] + "', '" + validation_set[index_walk][1] + "'], \n ")
        f.write("]\n\n")

        f.write("test_set = [")
        for index_walk in range(len(test_set)):
            f.write("['" + test_set[index_walk][0] + "', '" + test_set[index_walk][1] + "'], \n")
        f.write("]\n\n")

        f.write("\n")
        f.close()



training_set_list = []
validation_set_list = []
test_set_list = []

for index_walk in range(0, NUMERO_WALK + 1):
    training_set_list.append(get_training_set(START_DATE_CURRENT))
    validation_set_list.append(get_validation_set(START_DATE_CURRENT))
    test_set_list.append(get_test_set(START_DATE_CURRENT))

    # aggiorno la data di fine per la nuova coppia di test set togliendo l'offset
    # si potrebbe togliere anche un giorno per evitare di avere un giorno in comune per ogni test set
    START_DATE_CURRENT = (START_DATE_CURRENT + pd.DateOffset(months=VALIDATION_SET_SIZE)).date()

training_set_list
validation_set_list
test_set_list


print("training_set = ", training_set_list)
print("validation_set = ", validation_set_list)
print("test_set = ", test_set_list)

start_log(training_set_list, validation_set_list, test_set_list)

#print(len(training_set_list))
#print(len(validation_set_list))
#print(len(test_set_list))