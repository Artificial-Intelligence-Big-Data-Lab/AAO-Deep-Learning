import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from classes.Market import Market

'''
' @author Andrea Corriga
'
'''
class ResultsHandlerTemperature:

    df = pd.DataFrame()

    # path dove verranno salvati gli esperimenti
    __input_base_path = '../experiments/'

    # da dove legge le predizioni
    __input_folder = ''
    
    # dove verranno salvati i risultati
    __output_folder = ''

    # output di vggHandler, usato in input qui nel costrutore
    __experiment_name = ''

    # soglie utilizzate per l'ensemble
    __thresholds = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

    # qui inseriro il nome del mercato che voglio utilizzare per analizzare i dati
    __dataset = ''

    # Tutte le cartelle di output in cui salverò i risultati
    __output_folder_datasets = ''
    # il path del file dove verranno salvate la lista delle colonne da usare in test_set
    __validation_column_folder = ''
    # il path dove verranno salvati gli walk una volta rimosse le colonne inutili calcolate in validation
    __test_df_folder_after_column_remover = ''
    # il path dove salverò i risultati degli ensemble
    __output_folder_ensemble = ''
    # il path dove salverò i plot post ensemble
    __output_folder_plots = ''
    # il path dove salverò i risultati aggregati post walk
    __output_folder_csv_aggregate_by_walk = ''
    # il path dove salverò i risultati aggregati in un unico walk
    __output_folder_csv_aggregate_unique_walk = ''
    # il path dove salverò i risultati aggregati post walk
    __output_folder_csv_aggregate_by_walk_bh = ''
    # il path dove salverò i risultati aggregati in un unico walk
    __output_folder_csv_aggregate_unique_walk_bh = ''
    '''
    ' Inizializzo la classe.
    ' Il parametro in input è il nome dell'esperimento, ovvero l'output
    ' dato a VggHandler. In questo modo la classe sa dove leggere i modelli e dove stampare
    ' gli output
    '''
    def __init__(self, experiment_name):
        self.__experiment_name = experiment_name
        # Setto la cartella dove leggerò le predizioni
        self.__input_folder = self.__input_base_path + experiment_name + '/predictions/'

        # Setto la cartella dove salverò i risultati
        self.__output_folder = self.__input_base_path + experiment_name + '/results/'

        # Setto le cartelle dove salvare le colonne del validation e poi dove salvare i df risultati in test set
        self.__validation_column_folder = self.__output_folder + "datasets/validation/columns/"
        self.__test_df_folder_after_column_remover = self.__output_folder + "datasets/test/walk_after_column_remove/"

        self.__output_folder_ensemble = self.__output_folder + "ensemble/"
        self.__output_folder_plots = self.__output_folder + "plots_after_ensemble/"

        self.__output_folder_csv_aggregate_by_walk = self.__output_folder + "csv_aggregate_by_walk/"
        self.__output_folder_csv_aggregate_unique_walk = self.__output_folder + "csv_aggregate_unique_walk/"

        self.__output_folder_csv_aggregate_by_walk_bh = self.__output_folder + "csv_aggregate_by_walk_bh/"
        self.__output_folder_csv_aggregate_unique_walk_bh = self.__output_folder + "csv_aggregate_unique_walk_bh/"

        # Inserisco il dataset specificato. In questo modo posso utilizzare il metodo 
        # anche per altri mercati come il dax
        #self.__dataset = dataset

        # Preparo l'output folder
        if not os.path.isdir(self.__output_folder):
            os.makedirs(self.__output_folder)

        if not os.path.isdir(self.__validation_column_folder):
            os.makedirs(self.__validation_column_folder)
        
        if not os.path.isdir(self.__test_df_folder_after_column_remover):
            os.makedirs(self.__test_df_folder_after_column_remover)

    '''
    ' Calcola l'ensemable sulle colonne (reti) con il 100% di agreement
    '''
    def __full_ensemble(self, df):
        # Controllo quali righe hanno tutto 1
        m1 = df.eq(1).all(axis=1)

        # Controllo quali righe hanno tutto 0
        m2 = df.eq(-1).all(axis=1)

        # Prevengo sovrascritture di memoria
        local_df = df.copy()
        # Creo una nuova colonna ensemble, mettendo ad 1 se tutte le colonne sono a 1, -1 se sono tutte a 0, 0 altrimenti
        local_df['ensemble'] = np.select([m1, m2], [1, -1], 0)

        # rimuovo tutte le colonne e lascio una sola colonna ensemble che contiene solamente l'operazione da fare (1, -1, 0)
        local_df = local_df.drop(local_df.columns.difference(['ensemble']), axis=1)

        return local_df

    '''
    ' Calcola l'ensemable sulle colonne (reti) con una % di agreement
    '''
    def __perc_ensemble(self, df, thr=0.7):
        c1 = (df.eq(1).sum(1) / df.shape[1]).gt(thr)
        c2 = (df.eq(-1).sum(1) / df.shape[1]).gt(thr)
        c2.astype(int).mul(-1).add(c1)
        m = pd.DataFrame(np.select([c1, c2], [1, -1], 0), index=df.index, columns=['ensemble'])

        return m

    '''
    ' Questo metodo elimina tutte le colonne che hanno tutto 0 o tutto 1
    ' L'assunzione di base è che quella specifica rete non abbia imparato nulla. 
    ' Restituisce il df risultante, senza le colonne delle reti inutili
    '''
    def __remove_useless_net(self, df):
        nunique = df.apply(pd.Series.nunique)
        cols_to_drop = nunique[nunique == 1].index
        df = df.drop(cols_to_drop, axis=1)

        return df

    '''
    ' legge la lista delle colonne ritenute UTILI, calcolate da 
    ' __remove_useless_net(self, df), in modo da ottenere un df
    ' utilizzando le stesse reti. Questo metodo viene utilizzato per il test set
    '''
    def __remove_useless_net_from_validation(self, df, filename):
        # Leggo le colonne considerate inutili in validation
        columns = pd.read_csv(self.__validation_column_folder + filename + '.csv')
        return df[df.columns[df.columns.isin(columns)]]

    '''
    ' Funzione di appoggio, leggo tutti csv
    ' dentro la cartella delle predizioni (passata come input folder). 
    ' Uso questa funzione per 
    ' prendere i nomi di tutti i csv relativi ai walk
    '''
    def __csv_filename_reader(self, set_type):
        filename_list = []

        lenght = len(os.listdir(self.__input_folder + set_type + '/'))

        for index in range(lenght):
            filename_list.append("GADF_walk_" + str(index))
        
        return filename_list
    
    '''
    ' Stampo i plots di accuracy e coverage per diverse soglie di agreement
    ' I plot nell'asse delle y di accuracy, hanno uno zoom sui valori 
    ' in range da 40-70, per avere maggiore dettaglio nelle zone che più ci interessano
    ' tanto acc maggiori di 70% sono totalmente irrealistiche
    '''
    def __do_plot(self, thresholds, coverage, accuracy, set_type, filename):
        output_folder = self.__output_folder_plots + set_type + '/'

        if not os.path.isdir(output_folder):
            os.makedirs(output_folder)

        plt.figure(figsize=(15, 12))

        plt.subplot(2, 1, 1)
        plt.plot(thresholds, coverage)
        plt.title('Coverage / Accuracy plot per ' + filename)
        plt.xlabel('Threshold')
        plt.ylabel('Coverage')
        plt.grid(color='black', linestyle='-', linewidth=0.5)
        #plt.yticks(np.arange(0, 70, step=10))

        plt.subplot(2, 1, 2)
        plt.plot(thresholds, accuracy)
        plt.xlabel('Threshold')
        plt.ylabel('Accuracy')

        # Plotto di 0.5 da 40 a 60 nell'asse y, tanto le altre acc sono poco realistiche
        #plt.yticks(np.arange(40, 70, step=1))
        axes = plt.gca()
        # Setto i limiti da 40 a 65
        #axes.set_ylim([40, 70])

        plt.grid(color='black', linestyle='-', linewidth=0.5)

        #plt.show()

        plt.savefig(output_folder + filename + '.png')
        plt.close()

    '''
    '
    '''
    def __save_ensemble(self, df, set_type, thr, filename):
        path = self.__output_folder_ensemble + set_type + '/' + str(thr) + "/"

        # Se non esiste la cartella, la creo
        if not os.path.isdir(path):
            os.makedirs(path)

        df.to_csv(path + filename + '.csv', mode="w", header=True, index=True)

    '''
    ' Conto quante volte entro a comprare/vendere
    '''
    def __get_trade_count_for_df(self, df):
        return len(np.where(df['ensemble'] != 0)[0])

    '''
    ' Calcolo la coverage in percentuale, ovvero 
    ' quante volte sono entrato nel mercato. Se sono entrato
    ' 10 volte su 100, avrò lo 0.1 di coverage
    '''
    def __get_coverage_perc_for_df(self, df):
        # Conto il numero di righe del DF (numero di giorni in cui faccio predizioni)
        size = df.shape[0]
        coverage = self.__get_trade_count_for_df(df)
        return (coverage * 100) / size


    '''
    ' Conto quante volte ho fatto una long/short correttamente (-1, 1)
    '''
    def __get_trade_count_correct_for_df(self, df):
        return len(np.where(df['ensemble'] == df['label'])[0])


    '''
    ' Calcolo la % di accuracy sulle volte che sono entrato nel mercato
    ' quindi solamente se il valore è 1 o -1. Le hold (0) non le considero proprio
    '''
    def __get_accuracy_for_df(self, df):
        correct = self.__get_trade_count_correct_for_df(df)
        count_coverage = self.__get_trade_count_for_df(df)

        if count_coverage > 0:
            return (correct / count_coverage) * 100
        else:
            return 0

    '''
    '
    '''
    def __get_return_for_df(self, df): 
        walk_return = 0 

        df = df.reset_index()
        # per ogni elemento di join calcolo il return
        #for index, row in df.iterrows():
        for index, row in enumerate(df.itertuples(index=False)): 
            
            # siccome il return lo calcolo sul giorno successivo, se l'indice +1 arriva alla dimensione skippo
            # così non vado out-of-bound sull'array
            if index + 1 == df.shape[0]: 
                break

            # se entro nel mercato, != hold, sommo o sottraggo a seconda della correttezza dell'operazione
            if row.ensemble != 0:
                # nel caso in cui abbia predetto correttamente cosa fare 
                if row.ensemble == row.label:
                    walk_return = walk_return + abs(df.iloc[index + 1].delta) # sommo il return per questo walk

                    #print("DELTA GIORNO " + str(row.date_time) + " - " + str(row.delta))
                    #print("IL GIORNO " + str(df.iloc[index].date_time) + " ESEGUO UNA OPERAZIONE IL GIORNO " + str(df.iloc[index+1].date_time))
                    #print("DELTA GIORNO " + str(df.iloc[index+1].date_time) + " - " + str(df.iloc[index].delta))
                    #print(" ---- ")
                # nel caso in cui NON abbia predetto correttamente cosa fare 
                else:
                    walk_return = walk_return - abs(df.iloc[index + 1].delta) # sottraggo il return per questo walk
        
        return walk_return

    '''
    ' Funzione che data una serie storica
    ' calcola l'mm, romad, posizione i e j-esima dell'mdd, 
    ' il valore effettivo di della posizione i e j, ed il return
    ' di bh per quel periodo
    '
    ' Il parametro series dovrebbe essere l'array della colona close
    '''
    def __get_mdd_romad(self, series):
        cumulative = np.maximum.accumulate(series) - series
        
        if all(cumulative == 0): 
            if len(cumulative) <= 1:
                bh_global_return = 999
            else:
                bh_global_return = series[-1] - series[0]
            return 999, 999, 999, 999, 999, 999, bh_global_return

        # calcolo la posizione i-esima
        i = np.argmax(cumulative) # end of the period
        # calcolo la posizione j-esima
        j = np.argmax(series[:i]) # start of period

        bh_global_return = series[-1] - series[0]
        # mdd
        mdd = series[j] - series[i] 
        # romad
        romad = bh_global_return / mdd 
        # return totale
        #bh_global_return = sum(series) 

        return mdd, romad, i, j, series[i], series[j], bh_global_return

    '''
    ' Genero le colonne che userò nel DF per i risultati
    ' raggrupati riassuntivi per tutti gli walk
    ' il formato sarà [thr, [lista_walk]]
    '''
    def __gen_columns_for_results_csv(self, filenames):
        # colonna per la soglia
        columns = ['thr']
        # per ogni walk credo una colonna walk_indice
        for index, filename in enumerate(filenames):
            columns.append('walk_' + str(index))

        return columns

    '''
    ' Questo metodo genera i file aggregati per ogni soglia
    ' unendo tutti gli walk in un unico file.
    ' Legge i file degli ensemble, walk per walk e li unisce in un unico file. 
    ' Il file generato viene utilizzato per generare i report su multicharts
    '''
    def __merge_ensemble(self, set_type):

        # Prendo i nome dei file da mergiare a seconda del tipo di set specificato
        filenames = self.__csv_filename_reader(set_type=set_type)

        # path generale per leggere i file
        main_path = self.__output_folder + "ensemble/" + set_type + '/' 

        # path in cui salverò gli output. Se non esiste creo la cartella
        merge_path = main_path + 'merge/'
        if not os.path.isdir(merge_path):
            os.makedirs(merge_path)
        
        # per ogni soglia, leggo i file e li unisco
        for thr in self.__thresholds: 
            df = pd.DataFrame()

            file_path = main_path + str(thr) + '/'

            for name in filenames:
                f = pd.read_csv(file_path + name + '.csv')
                # concateno i file appena letti
                df = pd.concat([df, f], axis=0)

            # salvo il file mergiato per questa soglia. nel nome del file sarà indicata la soglia
            df.to_csv(merge_path + 'walk_global_' + str(thr) + '.csv', header=True, index=False)
    

    '''
    ' Metodi specifici per l'1D
    '''
    def __group(self, df, freq):

        # Get first Open, Last Close, Highest value of "High" and Lower value of "Low", and sum(Volume)
        grouped = df.groupby(pd.Grouper(key='date_time', freq=freq), sort=True).agg({
            'temperature': 'mean'
        })
        
        return grouped.dropna()

    '''
    ' Metodi specifici per l'1D experiment
    '''
    def __get_label_next_day_using_close(self, df, columns=[]):
        
        df['label'] = 0

        # Calcolo per ogni riga la Label, ovvero se il giorno dopo il mercato
        # salira' o scendera'
        # quindi se oggi la temperatura è 10 e domani 12, la label sarà 1 altrimenti 0
        df['label'] = (df['temperature'].diff().shift(-1) > 0).astype(int)
        
        # Invece di mettere la label sul giorno precedente, la mette sul giorno corrente. 
        #df['label'] = (df['close'].diff().bfill() > 0).astype(int)
        
        df.loc[df['label'] == 0, ['label']] = -1

        if len(columns) > 0:
            columns.append('label')
            df = df.filter(columns, axis=1)
        else:
            df = df.filter(['label'], axis=1) 

        return df


    '''
    ' Questo metodo calcola in automatico l'ensemble e genera i plot 
    ' per analizzare i risultati delle reti in training, validation e test. 
    ' Per correttezza questo metodo va eseguito prima con il parametro set_type = training, poi validation
    ' ed infine test
    '''
    def generate_ensemble(self, set_type):
        # Lista dei file che uso per prendere le predizioni per walk
        filenames = self.__csv_filename_reader(set_type=set_type)

        ##### Parametri dello script
        dataset_path = '../datasets/temperature/temperature.csv'
        choosen_city = 'Los Angeles'
        #####
        # leggo il dataset
        df = pd.read_csv(dataset_path)

        # Rimuovo le righe contenenti NaN
        df = df[['datetime', choosen_city]].dropna()

        # converto la temperatura in Celsius
        df[choosen_city] =  df[choosen_city]  - 273.15

        # converto in datetime la colonna, per essere usata in futuro
        df['datetime'] = pd.to_datetime(df['datetime'])
        # converto il nome della città in temperature, così lo script può essere generico
        df = df.rename(columns={'datetime': 'date_time', choosen_city: 'temperature'})

        df_1d = self.__group(df=df, freq='1d')
       
        
        # Aggiungo la label a fianco alla temperatura
        labels = self.__get_label_next_day_using_close(df=df_1d, columns=['datetime', 'temperature'])
        #labels = labels.drop(['temperature'], 1)
        
        # Per ogni file calcolo coverage, acc per diverse soglie
        for filename in filenames:  
            # Leggo il dataframe delle predizioni per ogni walk
            df = pd.read_csv(self.__input_folder + set_type + '/' + filename + '.csv')
            # Setto la data come indice
            df = df.set_index('date_time')

            # Qui salverò il nuovo df, in cui avrò rimosso le colonne inutili a seconda
            # del caso
            new_net = pd.DataFrame()

            # Se sto valutando il training set, rimuovo le colonne inutili (tutto 0 o 1) senza salvare niente.
            # Faccio tutto a runtime, poiché non mi interessa ai fini pratici, salvarmi quali colonne rimangono dal training
            # deprecato, ormai in training usiamo diverse compagnie, non ha senso utilizzarlo
            #if set_type == 'training':
                #new_net = self.__remove_useless_net(df=df)

            # Se sto valutando il validation set, rimuovo le colonne inutili (tutto 0 o 1) e salvo un csv con le colonne
            # che ho salvato dentro /results/datasets/validation/columns/nome_walk.csv
            if set_type == 'validation':
                # Rimuovo le reti inutili, quelle che danno tutte [0 | 1]
                new_net = self.__remove_useless_net(df=df)
                # salvo le colonne valide come un csv semplice dentro /results/datasets/validation/columns/nome_walk.csv
                pd.DataFrame(new_net.columns.tolist()).T.to_csv(self.__validation_column_folder + filename + '.csv', mode="w", header=False, index=False)
            
            # Se sto valutando il test set, rimuovo le colonne inutili (tutto 0 o 1) che ho rimosso nel validation
            # per fare ciò leggo le colonne ottenute nel blocco di if precedente (salvato su file) e rimuovo le colonne
            # dal file delle predizioni del test set. Salvo anche il csv ottenuto dentro
            # results/datasets/test/walk_after_column_remove/
            if set_type == 'test':
                new_net = self.__remove_useless_net_from_validation(df=df,  filename=filename)
                new_net.to_csv(self.__test_df_folder_after_column_remover + filename + '.csv', mode="w", header=True, index=True)

            
            # Rimuovo la colonna di label, altrimenti la uso per l'ensemble 
            new_net = new_net.drop(['label', 'index'], 1)
           
            # Per ogni soglia adesso calcolo il full ensemble e l'ensemble per una % di agreement
            for threshold in self.__thresholds:

                if threshold < 1:
                    # Calcolo l'eseamble con il % agreement
                    ensemble = self.__perc_ensemble(df=new_net, thr=threshold)
                else:
                    # Calcolo l'eseamble con il full agreement
                    ensemble = self.__full_ensemble(df=new_net)

                # Joinno con le label originali in modo da avere label, predizione affiancati
                df_join = ensemble.join(labels)

                # Salvo il file risultante dall'ensemble
                self.__save_ensemble(df=df_join, set_type=set_type, thr=threshold, filename=filename)

        # genero i file mergiati per i vari walk, soglia per soglia. li userò per multicharts
        self.__merge_ensemble(set_type=set_type)

    '''
    ' Genera i plot con andamento dell'accuracy e coverage 
    ' con varie soglie di ensemble
    '''
    def generate_plots(self, set_type):
        filenames = self.__csv_filename_reader(set_type=set_type)
        
        # Per ogni soglia adesso calcolo il full ensemble e l'ensemble per una % di agreement
        for filename in filenames: 
            coverage_array = []
            accuracy_array = []
            # Per ogni file calcolo coverage, acc per diverse soglie
            for threshold in self.__thresholds: 
                path = self.__output_folder + "/ensemble/" + set_type + '/' + str(threshold) + "/"
                 # Leggo il dataframe delle predizioni per ogni walk
                df = pd.read_csv(path + filename + '.csv')

                # Conto quante predizioni ho fatto corrette
                accuracy = self.__get_accuracy_for_df(df=df)

                # % di coverage
                percentage_coverage = self.__get_coverage_perc_for_df(df=df)

                # Aggiungo i risultati in un array per plottare
                coverage_array.append(percentage_coverage)
                accuracy_array.append(accuracy)

            # Per ogni filename stampo i plot per le varie soglie
            self.__do_plot(thresholds=self.__thresholds, coverage=coverage_array, accuracy=accuracy_array, set_type=set_type, filename=filename)

    '''
    ' A seconda del set_type specificato leggo il csv delle predizioni
    ' Dopodichè leggo il dataset specificato nel costruttore e lo raggruppo per 1 giorno 
    '''
    def generate_csv_aggregate_by_walk(self, set_type):
        # Lista dei file che uso per prendere le predizioni per walk
        filenames = self.__csv_filename_reader(set_type=set_type)
        
        # Leggo il dataset e lo raggruppo per un giorno
        # di base mi serve il valore delta dal dataframe market_1d per calcolarmi i return
        market = Market(dataset=self.__dataset)
        market_1d = market.group(freq='1d')
        
        # genero le colonne per il file riassuntivo in base al numero di walk specificati
        # le colonne saranno ['thr', [lista di walk]]
        columns = self.__gen_columns_for_results_csv(filenames)

        # credo il dataframe in cui salverò le informazioni
        df_return = pd.DataFrame(columns=columns)
        df_accuracy = pd.DataFrame(columns=columns)
        df_coverage = pd.DataFrame(columns=columns)
        df_trade_count = pd.DataFrame(columns=columns)
        df_mdd = pd.DataFrame(columns=columns)
        df_romad = pd.DataFrame(columns=columns)

        # Da qui in poi calcolo il guadagno per ogni soglia e per ogni walk
        for threshold in self.__thresholds:

            # in questa lista inserirò come primo elemento la soglia, 
            # poi per ogni walk un elemento con il guadagno relativo a quel walk
            # e infine il guadagno totale per un walk. 
            # questa variabile verrà convertita poi in DF per poter essere convertito in un csv
            # che sarà una matrice con soglie e walk
            walk_return_arr = [threshold]
            accuracy_array = [threshold]
            coverage_array = [threshold]
            trade_count_array = [threshold]
            mdd_array = [threshold]
            romad_array = [threshold]

            # Inizio il ciclo per ogni specifico filename, che semanticamente indica un walk
            for filename in filenames:
                

                # leggo il csv ottenuto calcolando l'ensemble per una soglia x
                result = pd.read_csv(self.__output_folder + "ensemble/" + set_type + '/' + str(threshold) + "/" + filename + '.csv')
                # imposto la chiave date_tme e unisco assieme al mercato raggruppato per un giorno
                result['date_time'] = pd.to_datetime(result['date_time'])
                result = result.set_index('date_time')
                # in questo modo dentro join ho label, ensemble (che rappresenta la label calcolata con l'ensemble) e il valore delta
                df_join = market_1d.join(result, on='date_time', how='right')

                # Torno nel for dei filename e aggiungo il walk return all'array dei return divisi per walk
                walk_return_arr.append(self.__get_return_for_df(df=df_join))

                accuracy_array.append(self.__get_accuracy_for_df(df=df_join))
                coverage_array.append(self.__get_coverage_perc_for_df(df=df_join))
                trade_count_array.append(self.__get_trade_count_for_df(df=df_join))

                # Calcolo MDD e Romad sul walk eliminando gli ensemble dal datasets
                df_without_hold = df_join[df_join['ensemble'] != 0]
                xs = df_without_hold['close'].values.flatten().tolist()
                mdd, romad, BH_return, i, j, i_val, j_val = self.__get_mdd_romad(xs)
                mdd_array.append(mdd)
                romad_array.append(romad)

            # Converto la lista walk_return_arr in un dataframe e lo aggiungo a df.
            # df diventerà  all_thr_results.csv 
            df_return = df_return.append( pd.DataFrame([walk_return_arr], columns=columns), 
                            ignore_index=True)
            df_accuracy = df_accuracy.append( pd.DataFrame([accuracy_array], columns=columns), 
                            ignore_index=True)
            df_coverage = df_coverage.append( pd.DataFrame([coverage_array], columns=columns), 
                            ignore_index=True)
            df_trade_count = df_trade_count.append( pd.DataFrame([trade_count_array], columns=columns), 
                            ignore_index=True)
            df_mdd = df_mdd.append( pd.DataFrame([mdd_array], columns=columns), 
                            ignore_index=True)
            df_romad = df_romad.append( pd.DataFrame([romad_array], columns=columns), 
                            ignore_index=True)

        # Finito tutto salvo il csv
        df_return = df_return.set_index('thr')
        df_accuracy = df_accuracy.set_index('thr')
        df_coverage = df_coverage.set_index('thr')
        df_trade_count = df_trade_count.set_index('thr')
        df_mdd = df_mdd.set_index('thr')
        df_romad = df_romad.set_index('thr')
        
        output_path = self.__output_folder_csv_aggregate_by_walk + set_type + '/'
        
        if not os.path.isdir(output_path):
            os.makedirs(output_path)
        
        # Il file del nome sarà: tipologia_set_all_thr_results.csv
        df_return.to_csv(output_path + 'return_foreach_walk.csv', mode="w", header=True, index=True)
        df_accuracy.to_csv(output_path + 'accuracy_foreach_walk.csv', mode="w", header=True, index=True)
        df_coverage.to_csv(output_path + 'coverage_foreach_walk.csv', mode="w", header=True, index=True)
        df_trade_count.to_csv(output_path + 'trade_count_foreach_walk.csv', mode="w", header=True, index=True)
        df_mdd.to_csv(output_path + 'mdd_foreach_walk.csv', mode="w", header=True, index=True)
        df_romad.to_csv(output_path + 'romad_foreach_walk.csv', mode="w", header=True, index=True) 
    
    '''
    ' A seconda del set_type specificato leggo il csv delle predizioni
    ' Dopodichè leggo il dataset specificato nel costruttore e lo raggruppo per 1 giorno 
    '''
    def generate_csv_aggregate_unique_walk(self, set_type):
        # Lista dei file che uso per prendere le predizioni per walk
        filenames = self.__csv_filename_reader(set_type=set_type)
        
        # Leggo il dataset e lo raggruppo per un giorno
        # di base mi serve il valore delta dal dataframe market_1d per calcolarmi i return
        market = Market(dataset=self.__dataset)
        market_1d = market.group(freq='1d')
        
        # genero le colonne per il file riassuntivo in base al numero di walk specificati
        # le colonne saranno ['thr', [lista di walk]]
        columns = ['thr', 'walk_global']

        # credo il dataframe in cui salverò le informazioni
        df_return = pd.DataFrame(columns=columns)
        df_accuracy = pd.DataFrame(columns=columns)
        df_coverage = pd.DataFrame(columns=columns)
        df_trade_count = pd.DataFrame(columns=columns)
        df_mdd = pd.DataFrame(columns=columns)
        df_romad = pd.DataFrame(columns=columns)

        # Da qui in poi calcolo il guadagno per ogni soglia e per ogni walk
        for threshold in self.__thresholds:
            df_unique = pd.DataFrame()
            # Inizio il ciclo per ogni specifico filename, che semanticamente indica un walk
            for filename in filenames:
                # leggo il csv ottenuto calcolando l'ensemble per una soglia x
                result = pd.read_csv(self.__output_folder + "ensemble/" + set_type + '/' + str(threshold) + "/" + filename + '.csv')
                # imposto la chiave date_tme e unisco assieme al mercato raggruppato per un giorno
                result['date_time'] = pd.to_datetime(result['date_time'])
                result = result.set_index('date_time')
                # in questo modo dentro join ho label, ensemble (che rappresenta la label calcolata con l'ensemble) e il valore delta
                df_join = market_1d.join(result, on='date_time', how='right')
                df_unique = pd.concat([df_unique, df_join])

            # in questa lista inserirò come primo elemento la soglia, 
            # poi per ogni walk un elemento con il guadagno relativo a quel walk
            # e infine il guadagno totale per un walk. 
            # questa variabile verrà convertita poi in DF per poter essere convertito in un csv
            # che sarà una matrice con soglie e walk
            walk_return_arr = [threshold]
            accuracy_array = [threshold]
            coverage_array = [threshold]
            trade_count_array = [threshold]
            mdd_array = [threshold]
            romad_array = [threshold]

            # Torno nel for dei filename e aggiungo il walk return all'array dei return divisi per walk
            walk_return_arr.append(self.__get_return_for_df(df=df_unique))

            accuracy_array.append(self.__get_accuracy_for_df(df=df_unique))
            coverage_array.append(self.__get_coverage_perc_for_df(df=df_unique))
            trade_count_array.append(self.__get_trade_count_for_df(df=df_unique))
            
            # Calcolo MDD e Romad sul walk eliminando gli ensemble dal datasets
            df_without_hold = df_unique[df_unique['ensemble'] != 0]
            xs = df_without_hold['close'].values.flatten().tolist()
            mdd, romad, BH_return, i, j, i_val, j_val = self.__get_mdd_romad(xs)
            mdd_array.append(mdd)
            romad_array.append(romad)

            # Converto la lista walk_return_arr in un dataframe e lo aggiungo a df.
            # df diventerà  all_thr_results.csv 
            df_return = df_return.append( pd.DataFrame([walk_return_arr], columns=columns), 
                            ignore_index=True)
            df_accuracy = df_accuracy.append( pd.DataFrame([accuracy_array], columns=columns), 
                            ignore_index=True)
            df_coverage = df_coverage.append( pd.DataFrame([coverage_array], columns=columns), 
                            ignore_index=True)
            df_trade_count = df_trade_count.append( pd.DataFrame([trade_count_array], columns=columns), 
                            ignore_index=True)
            df_mdd = df_mdd.append( pd.DataFrame([mdd_array], columns=columns), 
                            ignore_index=True)
            df_romad = df_romad.append( pd.DataFrame([romad_array], columns=columns), 
                            ignore_index=True)

        # Finito tutto salvo il csv
        df_return = df_return.set_index('thr')
        df_accuracy = df_accuracy.set_index('thr')
        df_coverage = df_coverage.set_index('thr')
        df_trade_count = df_trade_count.set_index('thr')
        df_mdd = df_mdd.set_index('thr')
        df_romad = df_romad.set_index('thr')

        output_path = self.__output_folder_csv_aggregate_unique_walk + set_type + '/'
        
        if not os.path.isdir(output_path):
            os.makedirs(output_path)
        
        # Il file del nome sarà: tipologia_set_all_thr_results.csv
        df_return.to_csv(output_path + 'return_foreach_walk.csv', mode="w", header=True, index=True)
        df_accuracy.to_csv(output_path + 'accuracy_foreach_walk.csv', mode="w", header=True, index=True)
        df_coverage.to_csv(output_path + 'coverage_foreach_walk.csv', mode="w", header=True, index=True)
        df_trade_count.to_csv(output_path + 'trade_count_foreach_walk.csv', mode="w", header=True, index=True)
        df_mdd.to_csv(output_path + 'mdd_foreach_walk.csv', mode="w", header=True, index=True)
        df_romad.to_csv(output_path + 'romad_foreach_walk.csv', mode="w", header=True, index=True)
    
    '''
    ' BUY HOLD
    '''
    def generate_csv_aggregate_by_walk_BH(self, set_type):
        # Lista dei file che uso per prendere le predizioni per walk
        filenames = self.__csv_filename_reader(set_type=set_type)
        
        # Leggo il dataset e lo raggruppo per un giorno
        # di base mi serve il valore delta dal dataframe market_1d per calcolarmi i return
        market = Market(dataset=self.__dataset)
        market_1d = market.group(freq='1d')
        
        # genero le colonne per il file riassuntivo in base al numero di walk specificati
        # le colonne saranno ['thr', [lista di walk]]
        columns = self.__gen_columns_for_results_csv(filenames)

        # credo il dataframe in cui salverò le informazioni
        df_return = pd.DataFrame(columns=columns)
        df_mdd = pd.DataFrame(columns=columns)
        df_romad = pd.DataFrame(columns=columns)

        # Da qui in poi calcolo il guadagno per ogni soglia e per ogni walk
        for threshold in self.__thresholds:

            # in questa lista inserirò come primo elemento la soglia, 
            # poi per ogni walk un elemento con il guadagno relativo a quel walk
            # e infine il guadagno totale per un walk. 
            # questa variabile verrà convertita poi in DF per poter essere convertito in un csv
            # che sarà una matrice con soglie e walk
            walk_return_arr = [threshold]
            mdd_array = [threshold]
            romad_array = [threshold]

            # Inizio il ciclo per ogni specifico filename, che semanticamente indica un walk
            for filename in filenames:
                # leggo il csv ottenuto calcolando l'ensemble per una soglia x
                result = pd.read_csv(self.__output_folder + "ensemble/" + set_type + '/' + str(threshold) + "/" + filename + '.csv')
                # imposto la chiave date_tme e unisco assieme al mercato raggruppato per un giorno
                result['date_time'] = pd.to_datetime(result['date_time'])
                result = result.set_index('date_time')
                # in questo modo dentro join ho label, ensemble (che rappresenta la label calcolata con l'ensemble) e il valore delta
                df_join = market_1d.join(result, on='date_time', how='right')

                # Calcolo MDD e Romad sul walk eliminando gli ensemble dal datasets
                # per BH non tolgo i valori hold, tanto il return lo calcolo come la differenza tra l'ultimo ed il primo
                xs = df_join['close'].values.flatten().tolist()
                mdd, romad, BH_return, i, j, i_val, j_val = self.__get_mdd_romad(xs)
                walk_return_arr.append(BH_return)
                mdd_array.append(mdd)
                romad_array.append(romad)

            # Converto la lista walk_return_arr in un dataframe e lo aggiungo a df.
            # df diventerà  all_thr_results.csv 
            df_return = df_return.append( pd.DataFrame([walk_return_arr], columns=columns), 
                            ignore_index=True)          
            df_mdd = df_mdd.append( pd.DataFrame([mdd_array], columns=columns), 
                            ignore_index=True)
            df_romad = df_romad.append( pd.DataFrame([romad_array], columns=columns), 
                            ignore_index=True)

        # Finito tutto salvo il csv
        df_return = df_return.set_index('thr')
        df_mdd = df_mdd.set_index('thr')
        df_romad = df_romad.set_index('thr')
        
        output_path = self.__output_folder_csv_aggregate_by_walk_bh + set_type + '/'
        
        if not os.path.isdir(output_path):
            os.makedirs(output_path)
        
        # Il file del nome sarà: tipologia_set_all_thr_results.csv
        df_return.to_csv(output_path + 'return_foreach_walk.csv', mode="w", header=True, index=True)
        df_mdd.to_csv(output_path + 'mdd_foreach_walk.csv', mode="w", header=True, index=True)
        df_romad.to_csv(output_path + 'romad_foreach_walk.csv', mode="w", header=True, index=True)

    def generate_csv_aggregate_unique_walk_BH(self, set_type):
        # Lista dei file che uso per prendere le predizioni per walk
        filenames = self.__csv_filename_reader(set_type=set_type)
        
        # Leggo il dataset e lo raggruppo per un giorno
        # di base mi serve il valore delta dal dataframe market_1d per calcolarmi i return
        market = Market(dataset=self.__dataset)
        market_1d = market.group(freq='1d')
        
        # genero le colonne per il file riassuntivo in base al numero di walk specificati
        # le colonne saranno ['thr', [lista di walk]]
        columns = ['thr', 'walk_global']

        # credo il dataframe in cui salverò le informazioni
        df_return = pd.DataFrame(columns=columns)
        df_mdd = pd.DataFrame(columns=columns)
        df_romad = pd.DataFrame(columns=columns)

        # Da qui in poi calcolo il guadagno per ogni soglia e per ogni walk
        for threshold in self.__thresholds:
            df_unique = pd.DataFrame()
            # Inizio il ciclo per ogni specifico filename, che semanticamente indica un walk
            for filename in filenames:
                # leggo il csv ottenuto calcolando l'ensemble per una soglia x
                result = pd.read_csv(self.__output_folder + "ensemble/" + set_type + '/' + str(threshold) + "/" + filename + '.csv')
                # imposto la chiave date_tme e unisco assieme al mercato raggruppato per un giorno
                result['date_time'] = pd.to_datetime(result['date_time'])
                result = result.set_index('date_time')
                # in questo modo dentro join ho label, ensemble (che rappresenta la label calcolata con l'ensemble) e il valore delta
                df_join = market_1d.join(result, on='date_time', how='right')
                df_unique = pd.concat([df_unique, df_join])

            # in questa lista inserirò come primo elemento la soglia, 
            # poi per ogni walk un elemento con il guadagno relativo a quel walk
            # e infine il guadagno totale per un walk. 
            # questa variabile verrà convertita poi in DF per poter essere convertito in un csv
            # che sarà una matrice con soglie e walk
            walk_return_arr = [threshold]
            mdd_array = [threshold]
            romad_array = [threshold]
            
            # Calcolo MDD e Romad sul walk eliminando gli ensemble dal datasets
            # non tolgo le hold perché in Bh calcolo il return come ultimo giorno - primo
            xs = df_unique['close'].values.flatten().tolist()
            mdd, romad, BH_return, i, j, i_val, j_val = self.__get_mdd_romad(xs)
            walk_return_arr.append(BH_return)
            mdd_array.append(mdd)
            romad_array.append(romad)

            # Converto la lista walk_return_arr in un dataframe e lo aggiungo a df.
            # df diventerà  all_thr_results.csv 
            df_return = df_return.append( pd.DataFrame([walk_return_arr], columns=columns), 
                            ignore_index=True)
            df_mdd = df_mdd.append( pd.DataFrame([mdd_array], columns=columns), 
                            ignore_index=True)
            df_romad = df_romad.append( pd.DataFrame([romad_array], columns=columns), 
                            ignore_index=True)

        # Finito tutto salvo il csv
        df_return = df_return.set_index('thr')
        df_mdd = df_mdd.set_index('thr')
        df_romad = df_romad.set_index('thr')
        
        output_path = self.__output_folder_csv_aggregate_unique_walk_bh + set_type + '/'
        
        if not os.path.isdir(output_path):
            os.makedirs(output_path)
        
        # Il file del nome sarà: tipologia_set_all_thr_results.csv
        df_return.to_csv(output_path + 'return_foreach_walk.csv', mode="w", header=True, index=True)
        df_mdd.to_csv(output_path + 'mdd_foreach_walk.csv', mode="w", header=True, index=True)
        df_romad.to_csv(output_path + 'romad_foreach_walk.csv', mode="w", header=True, index=True)

    

    