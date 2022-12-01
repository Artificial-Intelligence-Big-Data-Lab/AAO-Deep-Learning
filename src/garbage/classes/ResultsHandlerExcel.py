import os
import xlsxwriter
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from classes.Market import Market

'''
' @author Andrea Corriga
'
'''
class ResultsHandlerExcel:

    df = pd.DataFrame()

    # path dove verranno salvati gli esperimenti
    __input_base_path = '../experiments/'

    # da dove legge le predizioni
    __input_folder = ''
    
    # dove verranno salvati i risultati
    __output_folder = ''

    # output di vggHandler, usato in input qui nel costrutore
    __experiment_name = ''

    # il path del file dove verranno salvate la lista delle colonne da usare in test_set
    __validation_column_folder = ''

    # il path dove verranno salvati gli walk una volta rimosse le colonne inutili calcolate in validation
    __test_df_folder_after_column_remover = ''

    # soglie utilizzate per l'ensemble
    __thresholds = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

    # qui inseriro il nome del mercato che voglio utilizzare per analizzare i dati
    __dataset = '',

    '''
    ' Inizializzo la classe.
    ' Il parametro in input è il nome dell'esperimento, ovvero l'output
    ' dato a VggHandler. In questo modo la classe sa dove leggere i modelli e dove stampare
    ' gli output
    '''
    def __init__(self, experiment_name, dataset):
        self.__experiment_name = experiment_name
        # Setto la cartella dove leggerò le predizioni
        self.__input_folder = self.__input_base_path + experiment_name + '/predictions/'

        # Setto la cartella dove salverò i risultati
        self.__output_folder = self.__input_base_path + experiment_name + '/results/'

        # Setto le cartelle dove salvare le colonne del validation e poi dove salvare i df risultati in test set
        self.__validation_column_folder = self.__output_folder + "datasets/validation/columns/"
        self.__test_df_folder_after_column_remover = self.__output_folder + "datasets/test/walk_after_column_remove/"

        # Inserisco il dataset specificato. In questo modo posso utilizzare il metodo 
        # anche per altri mercati come il dax
        self.__dataset = dataset

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
        output_folder = self.__output_folder + 'plots/' + set_type + '/'

        if not os.path.isdir(output_folder):
            os.makedirs(output_folder)

        plt.figure(figsize=(15, 12))

        plt.subplot(2, 1, 1)
        plt.plot(thresholds, coverage)
        plt.title('Coverage / Accuracy plot per ' + filename)
        plt.xlabel('Threshold')
        plt.ylabel('Coverage')
        plt.grid(color='black', linestyle='-', linewidth=0.5)
        plt.yticks(np.arange(0, 70, step=10))

        plt.subplot(2, 1, 2)
        plt.plot(thresholds, accuracy)
        plt.xlabel('Threshold')
        plt.ylabel('Accuracy')

        # Plotto di 0.5 da 40 a 60 nell'asse y, tanto le altre acc sono poco realistiche
        plt.yticks(np.arange(40, 70, step=1))
        axes = plt.gca()
        # Setto i limiti da 40 a 65
        axes.set_ylim([40, 70])

        plt.grid(color='black', linestyle='-', linewidth=0.5)

        #plt.show()

        plt.savefig(output_folder + filename + '.png')
        plt.close()

    '''
    '
    '''
    def __save_ensemble(self, df, set_type, thr, filename):
        path = self.__output_folder + "/ensemble/" + set_type + '/' + str(thr) + "/"

        # Se non esiste la cartella, la creo
        if not os.path.isdir(path):
            os.makedirs(path)

        df.to_csv(path + filename + '.csv', mode="w", header=True, index=True)

    '''
    ' Conto quante volte entro a comprare/vendere
    '''
    def __get_count_coverage(self, df):
        return len(np.where(df['ensemble'] != 0)[0])

    '''
    ' Calcolo la coverage in percentuale, ovvero 
    ' quante volte sono entrato nel mercato. Se sono entrato
    ' 10 volte su 100, avrò lo 0.1 di coverage
    '''
    def __get_coverage_perc(self, df, size):
        coverage = self.__get_count_coverage(df)
        return (coverage * 100) / size


    '''
    ' Conto quante volte ho fatto una long/short correttamente (-1, 1)
    '''
    def __get_count_correct(self, df):
        return len(np.where(df['ensemble'] == df['label'])[0])


    '''
    ' Calcolo la % di accuracy sulle volte che sono entrato nel mercato
    ' quindi solamente se il valore è 1 o -1. Le hold (0) non le considero proprio
    '''
    def __get_accuracy(self, df):
        correct = self.__get_count_correct(df)
        count_coverage = self.__get_count_coverage(df)

        if count_coverage > 0:
            return (correct / count_coverage) * 100
        else:
            return 0

    '''
    ' Questo metodo calcola in automatico l'ensemble e genera i plot 
    ' per analizzare i risultati delle reti in training, validation e test. 
    ' Per correttezza questo metodo va eseguito prima con il parametro set_type = training, poi validation
    ' ed infine test
    '''
    def generate_ensemble_and_plots(self, set_type):
        # Lista dei file che uso per prendere le predizioni per walk
        filenames = self.__csv_filename_reader(set_type=set_type)

        # Prendo il mercato specificato nel costruttore. 
        market = Market(dataset=self.__dataset)   

        # ottengo la label per il giorno corrente per il mercato specificato
        labels = market.get_label_current_day(freq='1d', columns=['close'])
        
        # genero le colonne per il file riassuntivo in base al numero di walk specificati
        # le colonne saranno ['thr', [lista di walk], 'walk_global']
        columns = self.__gen_columns_for_calculate_return(filenames)

        # credo il dataframe in cui salverò le informazioni
        df_accuracy = pd.DataFrame(columns=columns)
        df_coverage = pd.DataFrame(columns=columns)
        # conto quante volte entro nel mercato
        df_trade_count = pd.DataFrame(columns=columns)

        df_mdd = pd.DataFrame(columns=columns)
        df_romad = pd.DataFrame(columns=columns)


        # Per ogni soglia adesso calcolo il full ensemble e l'ensemble per una % di agreement
        for threshold in self.__thresholds:
            coverage_array = []
            accuracy_array = []
            trade_count_array = []
            mdd_array = []
            romad_array = []

            coverage_array.append(threshold)
            accuracy_array.append(threshold)
            trade_count_array.append(threshold)
            mdd_array.append(threshold)
            romad_array.append(threshold)

            # Per ogni file calcolo coverage, acc per diverse soglie
            for filename in filenames:  

                # Leggo il dataframe delle predizioni per ogni walk
                df = pd.read_csv(self.__input_folder + set_type + '/' + filename + '.csv')
                # Setto la data come indice
                df = df.set_index('date_time')

                # Qui salverò il nuovo df, in cui avrò rimosso le colonne inutili a seconda
                # del caso
                new_net = pd.DataFrame()

                # Se sto valutando il validation set, rimuovo le colonne inutili (tutto 0 o 1) e salvo un csv con le colonne
                # che ho salvato dentro /results/datasets/validation/columns/nome_walk.csv
                if set_type == 'validation':
                    # Rimuovo le reti inutili, quelle che danno tutte [0 | 1]
                    new_net = self.__remove_useless_net(df=df)
                    # salvo le colonne valide 
                    pd.DataFrame(new_net.columns.tolist()).T.to_csv(self.__validation_column_folder + filename + '.csv', mode="w", header=False, index=False)
                
                # Se sto valutando il test set, rimuovo le colonne inutili (tutto 0 o 1) che ho rimosso nel validation
                # per fare ciò leggo le colonne ottenute nel blocco di if precedente (salvato su file) e rimuovo le colonne
                # dal file delle predizioni del test set. Salvo anche il csv ottenuto dentro
                # results/datasets/test/walk_after_column_remove/
                if set_type == 'test':
                    new_net = self.__remove_useless_net_from_validation(df=df,  filename=filename)
                    new_net.to_csv(self.__test_df_folder_after_column_remover + filename + '.csv', mode="w", header=True, index=True)
                

                if threshold < 1:
                    # Calcolo l'eseamble con il % agreement
                    ensemble = self.__perc_ensemble(df=new_net, thr=threshold)
                else:
                    # Calcolo l'eseamble con il full agreement
                    ensemble = self.__full_ensemble(df=new_net)

                # Joinno con le label originali in modo da avere label, predizione affiancati
                df_join = ensemble.join(labels)

                # Salvo il file risultante dall'ensemble
                #self.__save_ensemble(df=df_join, set_type=set_type, thr=threshold, filename=filename)

                # Conto il numero di righe del DF (numero di giorni in cui faccio predizioni)
                size = df_join.shape[0]

                # Conto quante predizioni ho fatto corrette
                accuracy = self.__get_accuracy(df=df_join)

                # % di coverage
                percentage_coverage = self.__get_coverage_perc(df=df_join, size=size)
                trade_count = self.__get_count_coverage(df=df_join)

                #mdd e romad
                df_without_hold = df_join[df_join['ensemble'] != 0]

                xs = df_without_hold['close'].values.flatten().tolist()
                
                mdd, romad, global_return, i, j, i_val, j_val = mdd_simple(xs)

                # Aggiungo i risultati in un array per plottare
                coverage_array.append(percentage_coverage)
                accuracy_array.append(accuracy)
                trade_count_array.append(trade_count)
                mdd_array.append(mdd)
                romad_array.append(romad)

            # Per ogni filename stampo i plot per le varie soglie
            df_accuracy = df_accuracy.append( pd.DataFrame([accuracy_array], columns=columns), 
                          ignore_index=True)

            df_coverage = df_coverage.append( pd.DataFrame([coverage_array], columns=columns), 
                          ignore_index=True)
            df_trade_count = df_trade_count.append( pd.DataFrame([trade_count_array], columns=columns), 
                          ignore_index=True)

            df_mdd =  df_mdd.append( pd.DataFrame([mdd_array], columns=columns), 
                          ignore_index=True)
            df_romad =  df_romad.append( pd.DataFrame([romad_array], columns=columns), 
                          ignore_index=True)

        return df_accuracy, df_coverage, df_trade_count, df_mdd, df_romad

    def generate_ensemble_and_plots_global(self, set_type):
        # Lista dei file che uso per prendere le predizioni per walk
        filenames = self.__csv_filename_reader(set_type=set_type)

        # Prendo il mercato specificato nel costruttore. 
        market = Market(dataset=self.__dataset)   

        # ottengo la label per il giorno corrente per il mercato specificato
        labels = market.get_label_current_day(freq='1d', columns=['close'])
        
        # genero le colonne per il file riassuntivo in base al numero di walk specificati
        # le colonne saranno ['thr', [lista di walk], 'walk_global']
        columns = ['thr', 'walk_all']

        # credo il dataframe in cui salverò le informazioni
        df_accuracy = pd.DataFrame(columns=columns)
        df_coverage = pd.DataFrame(columns=columns)
        # conto quante volte entro nel mercato
        df_trade_count = pd.DataFrame(columns=columns)

        df_mdd = pd.DataFrame(columns=columns)
        df_romad = pd.DataFrame(columns=columns)


        # Per ogni soglia adesso calcolo il full ensemble e l'ensemble per una % di agreement
        for threshold in self.__thresholds:
            coverage_array = []
            accuracy_array = []
            trade_count_array = []
            mdd_array = []
            romad_array = []

            coverage_array.append(threshold)
            accuracy_array.append(threshold)
            trade_count_array.append(threshold)
            mdd_array.append(threshold)
            romad_array.append(threshold)

            df = pd.DataFrame()
            # Per ogni file calcolo coverage, acc per diverse soglie
            for filename in filenames:  
                # Leggo il dataframe delle predizioni per ogni walk
                df2 = pd.read_csv(self.__input_folder + set_type + '/' + filename + '.csv')
                # Setto la data come indice
                df = pd.concat([df, df2])
            
            df = df.set_index('date_time')

            # Qui salverò il nuovo df, in cui avrò rimosso le colonne inutili a seconda
            # del caso
            new_net = pd.DataFrame()

            # Se sto valutando il validation set, rimuovo le colonne inutili (tutto 0 o 1) e salvo un csv con le colonne
            # che ho salvato dentro /results/datasets/validation/columns/nome_walk.csv
            if set_type == 'validation':
                # Rimuovo le reti inutili, quelle che danno tutte [0 | 1]
                new_net = self.__remove_useless_net(df=df)
                # salvo le colonne valide 
                pd.DataFrame(new_net.columns.tolist()).T.to_csv(self.__validation_column_folder + filename + '.csv', mode="w", header=False, index=False)
            
            # Se sto valutando il test set, rimuovo le colonne inutili (tutto 0 o 1) che ho rimosso nel validation
            # per fare ciò leggo le colonne ottenute nel blocco di if precedente (salvato su file) e rimuovo le colonne
            # dal file delle predizioni del test set. Salvo anche il csv ottenuto dentro
            # results/datasets/test/walk_after_column_remove/
            if set_type == 'test':
                new_net = self.__remove_useless_net_from_validation(df=df,  filename=filename)
                new_net.to_csv(self.__test_df_folder_after_column_remover + filename + '.csv', mode="w", header=True, index=True)
            

            if threshold < 1:
                # Calcolo l'eseamble con il % agreement
                ensemble = self.__perc_ensemble(df=new_net, thr=threshold)
            else:
                # Calcolo l'eseamble con il full agreement
                ensemble = self.__full_ensemble(df=new_net)

            # Joinno con le label originali in modo da avere label, predizione affiancati
            df_join = ensemble.join(labels)

            # Salvo il file risultante dall'ensemble
            #self.__save_ensemble(df=df_join, set_type=set_type, thr=threshold, filename=filename)

            # Conto il numero di righe del DF (numero di giorni in cui faccio predizioni)
            size = df_join.shape[0]

            # Conto quante predizioni ho fatto corrette
            accuracy = self.__get_accuracy(df=df_join)

            # % di coverage
            percentage_coverage = self.__get_coverage_perc(df=df_join, size=size)
            trade_count = self.__get_count_coverage(df=df_join)

            #mdd e romad
            df_without_hold = df_join[df_join['ensemble'] != 0]

            xs = df_without_hold['close'].values.flatten().tolist()
            
            mdd, romad, global_return, i, j, i_val, j_val = mdd_simple(xs)

            # Aggiungo i risultati in un array per plottare
            coverage_array.append(percentage_coverage)
            accuracy_array.append(accuracy)
            trade_count_array.append(trade_count)
            mdd_array.append(mdd)
            romad_array.append(romad)

            # Per ogni filename stampo i plot per le varie soglie
            df_accuracy = df_accuracy.append( pd.DataFrame([accuracy_array], columns=columns), 
                            ignore_index=True)

            df_coverage = df_coverage.append( pd.DataFrame([coverage_array], columns=columns), 
                            ignore_index=True)
            df_trade_count = df_trade_count.append( pd.DataFrame([trade_count_array], columns=columns), 
                            ignore_index=True)

            df_mdd =  df_mdd.append( pd.DataFrame([mdd_array], columns=columns), 
                            ignore_index=True)
            df_romad =  df_romad.append( pd.DataFrame([romad_array], columns=columns), 
                            ignore_index=True)

        return df_accuracy, df_coverage, df_trade_count, df_mdd, df_romad

    '''
    ' A seconda del set_type specificato leggo il csv delle predizioni
    ' Dopodichè leggo il dataset specificato nel costruttore e lo raggruppo per 1 giorno 
    '''
    def calculate_return(self, set_type):
        # Lista dei file che uso per prendere le predizioni per walk
        filenames = self.__csv_filename_reader(set_type=set_type)
        
        # Leggo il dataset e lo raggruppo per un giorno
        # di base mi serve il valore delta dal dataframe market_1d per calcolarmi i return
        market = Market(dataset=self.__dataset)
        market_1d = market.group(freq='1d')
        
        # genero le colonne per il file riassuntivo in base al numero di walk specificati
        # le colonne saranno ['thr', [lista di walk], 'walk_global']
        columns = self.__gen_columns_for_calculate_return(filenames)

        # credo il dataframe in cui salverò le informazioni
        df = pd.DataFrame(columns=columns)

        # Da qui in poi calcolo il guadagno per ogni soglia e per ogni walk
        for threshold in self.__thresholds:

            # in questa lista inserirò come primo elemento la soglia, 
            # poi per ogni walk un elemento con il guadagno relativo a quel walk
            # e infine il guadagno totale per un walk. 
            # questa variabile verrà convertita poi in DF per poter essere convertito in un csv
            # che sarà una matrice con soglie e walk
            walk_return_arr = [threshold]

            # qui salverà il guadagno totale per una specifica soglia
            global_return = 0

            # Inizio il ciclo per ogni specifico filename, che semanticamente indica un walk
            for filename in filenames:
                # Variabile in cui inserisco il return totale per un solo walk
                walk_return = 0

                # leggo il csv ottenuto calcolando l'ensemble per una soglia x
                result = pd.read_csv(self.__output_folder + "ensemble/" + set_type + '/' + str(threshold) + "/" + filename + '.csv')
                # imposto la chiave date_tme e unisco assieme al mercato raggruppato per un giorno
                result['date_time'] = pd.to_datetime(result['date_time'])
                result = result.set_index('date_time')
                # in questo modo dentro join ho label, ensemble (che rappresenta la label calcolata con l'ensemble) e il valore delta
                join = market_1d.join(result, on='date_time', how='right')
                
                # per ogni elemento di join calcolo il return
                for index, row in join.iterrows():

                    # nel caso in cui abbia predetto correttamente cosa fare 
                    if row['ensemble'] == row['label']:
                        walk_return = walk_return + abs(row['delta']) # sommo il return per questo walk
                        global_return = global_return + abs(row['delta']) # sommo il return per tutti gli walk

                    # nel caso in cui NON abbia predetto correttamente cosa fare 
                    else:
                        if row['ensemble'] != 0 and row['ensemble'] != row['label']:
                            walk_return = walk_return - abs(row['delta']) # sommo il return per questo walk
                            global_return = global_return - abs(row['delta']) # sommo il return per tutti gli walk

                # Torno nel for dei filename e aggiungo il walk return all'array dei return divisi per walk
                walk_return_arr.append(walk_return)
            # a questo punto manca solamente il return globale per la soglia specificata
            walk_return_arr.append(global_return)

            # Converto la lista walk_return_arr in un dataframe e lo aggiungo a df.
            # df diventerà  all_thr_results.csv 
            df = df.append( pd.DataFrame([walk_return_arr], columns=columns), 
                            ignore_index=True)

        # Finito tutto salvo il csv
        df = df.set_index('thr')
    
        # Il file del nome sarà: tipologia_set_all_thr_results.csv
        df.to_csv(self.__output_folder + "/ensemble/" + set_type + '/' + set_type + '_all_thr_results.csv', mode="w", header=True, index=True)
    

    '''
    ' Genero le colonne che userò nel DF per il file all_thr_result.csv
    ' usato in: calculate_return()
    '''
    def __gen_columns_for_calculate_return(self, filenames):
        # colonna per la soglia
        columns = ['thr']
        # per ogni walk credo una colonna walk_indice
        for index, filename in enumerate(filenames):
            columns.append('walk_' + str(index))

        return columns

    '''
    ' Questo metodo calcola in automatico l'ensemble e genera i plot 
    ' per analizzare i risultati delle reti in training, validation e test. 
    ' Per correttezza questo metodo va eseguito prima con il parametro set_type = training, poi validation
    ' ed infine test
    '''
    def generate_buy_hold(self, set_type):
        # Lista dei file che uso per prendere le predizioni per walk
        filenames = self.__csv_filename_reader(set_type=set_type)
        
        # Leggo il dataset e lo raggruppo per un giorno
        # di base mi serve il valore delta dal dataframe market_1d per calcolarmi i return
        market = Market(dataset=self.__dataset)
        market_1d = market.group(freq='1d')
        
        # genero le colonne per il file riassuntivo in base al numero di walk specificati
        # le colonne saranno ['thr', [lista di walk], 'walk_global']
        columns = self.__gen_columns_for_calculate_return(filenames)

        # credo il dataframe in cui salverò le informazioni
        df_return = pd.DataFrame(columns=columns)
        df_romad = pd.DataFrame(columns=columns)
        df_mdd = pd.DataFrame(columns=columns)

        walk_return_arr = [1]
        walk_romad_arr = [1]
        walk_mdd_arr = [1]

        return_totale = 0
        romad_totale = 0
        mdd_totale = 0

        df_totale = pd.DataFrame()
        # Inizio il ciclo per ogni specifico filename, che semanticamente indica un walk
        for filename in filenames:
            # Variabile in cui inserisco il return totale per un solo walk
            walk_return = 0

            # leggo il csv ottenuto calcolando l'ensemble per una soglia x
            result = pd.read_csv(self.__output_folder + "ensemble/" + set_type + '/0.5/' + filename + '.csv')
            # imposto la chiave date_tme e unisco assieme al mercato raggruppato per un giorno
            result['date_time'] = pd.to_datetime(result['date_time'])
            result = result.set_index('date_time')
            # in questo modo dentro join ho label, ensemble (che rappresenta la label calcolata con l'ensemble) e il valore delta
            join = market_1d.join(result, on='date_time', how='right')

            df_totale = pd.concat([df_totale, join])

            xs = join['close'].values.flatten().tolist()

            mdd, romad, global_return, i, j, i_val, j_val = mdd_simple(xs)

            # Torno nel for dei filename e aggiungo il walk return all'array dei return divisi per walk
            walk_return_arr.append(global_return)
            walk_romad_arr.append(romad)
            walk_mdd_arr.append(mdd)

        xs = df_totale['close'].values.flatten().tolist()

        mdd_totale, romad_totale, return_totale, i, j, i_val, j_val = mdd_simple(xs)

        # Converto la lista walk_return_arr in un dataframe e lo aggiungo a df.
        # df diventerà  all_thr_results.csv 
        df_return = df_return.append( pd.DataFrame([walk_return_arr], columns=columns), 
                        ignore_index=True)

        df_romad = df_romad.append( pd.DataFrame([walk_romad_arr], columns=columns), 
                        ignore_index=True)
        
        df_mdd = df_mdd.append( pd.DataFrame([walk_mdd_arr], columns=columns), 
                        ignore_index=True)

        return df_return, df_mdd, df_romad, return_totale

    def generate_buy_hold_global(self, set_type):
        # Lista dei file che uso per prendere le predizioni per walk
        filenames = self.__csv_filename_reader(set_type=set_type)
        
        # Leggo il dataset e lo raggruppo per un giorno
        # di base mi serve il valore delta dal dataframe market_1d per calcolarmi i return
        market = Market(dataset=self.__dataset)
        market_1d = market.group(freq='1d')
        
        # genero le colonne per il file riassuntivo in base al numero di walk specificati
        # le colonne saranno ['thr', [lista di walk], 'walk_global']
        columns = ['thr', 'walk_all']

        return_totale = 0
        romad_totale = 0
        mdd_totale = 0

        df_totale = pd.DataFrame()
        # Inizio il ciclo per ogni specifico filename, che semanticamente indica un walk
        df = pd.DataFrame()
        # Per ogni file calcolo coverage, acc per diverse soglie
        for filename in filenames:  
            # Leggo il dataframe delle predizioni per ogni walk
            df2 = pd.read_csv(self.__output_folder + "ensemble/" + set_type + '/0.5/' + filename + '.csv')
            # Setto la data come indice
            df = pd.concat([df, df2])
        df['date_time'] = pd.to_datetime(df['date_time'])
        df = df.set_index('date_time')

        # in questo modo dentro join ho label, ensemble (che rappresenta la label calcolata con l'ensemble) e il valore delta
        join = market_1d.join(df, on='date_time', how='right')

        xs = join['close'].values.flatten().tolist()

        mdd, romad, global_return, i, j, i_val, j_val = mdd_simple(xs)

        mdd_totale, romad_totale, return_totale, i, j, i_val, j_val = mdd_simple(xs)

        return return_totale, mdd_totale, romad_totale


def mdd_simple(series):
    
    cumulative = np.maximum.accumulate(series) - series
    
    if all(cumulative == 0): 
        if len(cumulative) <= 1:
            global_return = 999
        else:
            global_return = series[-1] - series[0]
        return 999, 999, global_return, 999, 999, 999, 999

    # calcolo la posizione i-esima
    i = np.argmax(cumulative) # end of the period
    # calcolo la posizione j-esima
    j = np.argmax(series[:i]) # start of period

    global_return = series[-1] - series[0]
    # mdd
    mdd = series[j] - series[i] 
    # romad
    romad = global_return / mdd 
    # return totale
    #global_return = sum(series) 

    return mdd, romad, global_return, i, j, series[i], series[j]