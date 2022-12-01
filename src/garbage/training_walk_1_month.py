import pandas as pd
from classes.VggHandler import VggHandler
from classes.ResultsHandler import ResultsHandler
from classes.Market import Market

# import os
# os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID";
# os.environ["CUDA_VISIBLE_DEVICES"]="1";

TRAINING_SET_SIZE = 10 # YEARS
VALIDATION_SET_SIZE = 3 # MONTHS
TEST_SET_SIZE = 1 # MONTHS

# genero la coppia data inizio - data fine per il training set usando l'offset  specificato
def get_training_set(last_date):
    start_date = (last_date - pd.DateOffset(years=TRAINING_SET_SIZE)).date()
    return [str(start_date), str(last_date)]

# genero la coppia data inizio - data fine per il validation set usando l'offset  specificato
def get_validation_set(last_date):
    start_date = (last_date - pd.DateOffset(months=VALIDATION_SET_SIZE)).date()

    return [str(start_date), str(last_date)]

# genero la coppia data inizio - data fine per il test set usando l'offset  specificato
def get_test_set(last_date):
    start_date = (last_date - pd.DateOffset(months=TEST_SET_SIZE)).date()

    return [str(start_date), str(last_date)]

def start_log(training_set, validation_set, test_set):
        f=open('log.txt','a')

        f.write("Number of walks: " + str(len(training_set)) + "\n")
        
        f.write("Walk \t\t Training \t\t\t Validation \t\t\t Test\n")
        
        for index_walk in range(len(training_set)):
            f.write("Walk " + str(index_walk) + ": \t ")
            f.write("[" + training_set[index_walk][0] + " - " + training_set[index_walk][1] + "] \t ")
            f.write("[" + validation_set[index_walk][0] + " - " + validation_set[index_walk][1] + "] \t ")
            f.write("[" + test_set[index_walk][0] + " - " + test_set[index_walk][1] + "] \n")
        
        f.write("\n")
        f.close()

latest_date_test = pd.to_datetime('2017-10-31').date()

first_day_test = pd.to_datetime('2010-05-31').date()


training_set_list = []
validation_set_list = []
test_set_list = []

while pd.to_datetime(first_day_test) <= pd.to_datetime(latest_date_test):
    # Per il training tolgo la dimensione del test + dimensione del validation + 2 giorni (1giorno di validation + 1 giorno di training)
    latest_training_day = latest_date_test - pd.DateOffset(months=TEST_SET_SIZE) - pd.DateOffset(months=VALIDATION_SET_SIZE) - pd.DateOffset(days=2)
    latest_date_training = latest_training_day.date()

     # Per il validation tolgo la dimensione del test + 1 giorno (dovrebbe finire il giorno prima del test)
    latest_validation_day = latest_date_test - pd.DateOffset(months=TEST_SET_SIZE) - pd.DateOffset(days=1)
    latest_date_validation = latest_validation_day.date()

    # creo le liste contenente le coppie per ogni test
    training_set_list.append(get_training_set(latest_date_training))
    validation_set_list.append(get_validation_set(latest_date_validation))
    test_set_list.append(get_test_set(latest_date_test))

    # aggiorno la data di fine per la nuova coppia di test set togliendo l'offset
    # si potrebbe togliere anche un giorno per evitare di avere un giorno in comune per ogni test set
    latest_date_test = (latest_date_test - pd.DateOffset(months=TEST_SET_SIZE)).date()

training_set_list.reverse()
validation_set_list.reverse()
test_set_list.reverse()


for i in range(1,2):
        vgg = VggHandler()

        vgg.net_config(epochs=200, number_of_nets=20)

        vgg.run_initialize(dataset='sp500',
                input_folder='delta_experiment/gadf/delta/',
                training_set=training_set_list,
                validation_set=validation_set_list,
                test_set=test_set_list,
                input_shape=(40,40,3),
                output_folder='1_month_delta_experiment_' + str(i))

        vgg.run()

        results_handler = ResultsHandler(experiment_name='1_month_delta_experiment_' + str(i), dataset='sp500')

        print("Calculatin training predictions and results...")
        vgg.get_predictions(set_type='training')
        results_handler.generate_ensemble_and_plots(set_type='training')
        results_handler.calculate_return(set_type='training')

        print("Calculatin validation predictions and results...")
        vgg.get_predictions(set_type='validation')
        results_handler.generate_ensemble_and_plots(set_type='validation')
        results_handler.calculate_return(set_type='validation')

        print("Calculatin test predictions and results...")
        vgg.get_predictions(set_type='test')
        results_handler.generate_ensemble_and_plots(set_type='test')
        results_handler.calculate_return(set_type='test')
