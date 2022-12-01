'''
' Vecchio metodo che genera i Json
' per ogni rete e poi calcola l'avg
'''
def generate_custom_metrics(self, type='validation', penalty=32, stop_loss=1000, start_by_walk=-1, end_at_walk=None):

    for index_walk in range(0, len(self.walks_list)):
        # per iniziare tot walk più avanti
        if(index_walk < start_by_walk):
            continue;
        # per finire al walk n° 
        if(end_at_walk != None):
            if(index_walk > end_at_walk):
                continue;

        walk_str = 'walk_' + str(index_walk)
        if type is 'validation':
            date_list, epochs_list = self.get_date_epochs_walk(path=self.original_predictions_validation_folder, walk=walk_str)
        if type is 'test': 
            date_list, epochs_list = self.get_date_epochs_walk(path=self.original_predictions_test_folder, walk=walk_str)

        
        avg_ls_returns = []
        avg_lh_returns = []
        avg_sh_returns = []
        avg_bh_returns = [] # BH

        # MDDS
        avg_ls_mdds = []
        avg_lh_mdds = []
        avg_sh_mdds = []
        avg_bh_mdds = [] # BH

        # ROMADS
        avg_ls_romads = []
        avg_lh_romads = []
        avg_sh_romads = []
        avg_bh_romads = []

        avg_longs_precisions = []
        avg_shorts_precisions = []
        
        avg_label_longs_coverage = []
        avg_label_shorts_coverage = []

        # NUOVO TEST AVG POC
        avg_longs_poc = []
        avg_shorts_poc = []

        # accuracy
        avg_accuracy = []

        # RETURNS
        all_ls_returns = np.zeros(shape=(len(self.nets_list), len(epochs_list)))
        all_lh_returns = np.zeros(shape=(len(self.nets_list), len(epochs_list)))
        all_sh_returns = np.zeros(shape=(len(self.nets_list), len(epochs_list)))
        all_bh_return = np.zeros(shape=(len(self.nets_list), len(epochs_list))) #BH

        # ROMADS
        all_ls_romads = np.zeros(shape=(len(self.nets_list), len(epochs_list)))
        all_lh_romads = np.zeros(shape=(len(self.nets_list), len(epochs_list)))
        all_sh_romads = np.zeros(shape=(len(self.nets_list), len(epochs_list)))
        all_bh_romads = np.zeros(shape=(len(self.nets_list), len(epochs_list))) #BH

        # MDDS
        all_ls_mdds = np.zeros(shape=(len(self.nets_list), len(epochs_list)))
        all_lh_mdds = np.zeros(shape=(len(self.nets_list), len(epochs_list)))
        all_sh_mdds = np.zeros(shape=(len(self.nets_list), len(epochs_list)))
        all_bh_mdds = np.zeros(shape=(len(self.nets_list), len(epochs_list))) # BH

        # PRECISIONI E LINEA RETTA DEL BILANCIAMENTO DELLE CLASSI
        all_longs_precisions = np.zeros(shape=(len(self.nets_list), len(epochs_list)))
        all_shorts_precisions = np.zeros(shape=(len(self.nets_list), len(epochs_list)))
        all_labels_longs_coverage = np.zeros(shape=(len(self.nets_list), len(epochs_list)))
        all_labels_shorts_coverage = np.zeros(shape=(len(self.nets_list), len(epochs_list)))

        # % di operazioni fatte 
        all_long_operations = np.zeros(shape=(len(self.nets_list), len(epochs_list)))
        all_short_operations = np.zeros(shape=(len(self.nets_list), len(epochs_list)))
        all_hold_operations = np.zeros(shape=(len(self.nets_list), len(epochs_list)))

        # Precision over coverage
        all_longs_poc = np.zeros(shape=(len(self.nets_list), len(epochs_list)))
        all_shorts_poc = np.zeros(shape=(len(self.nets_list), len(epochs_list)))
        
        #accuracy
        all_accuracy = np.zeros(shape=(len(self.nets_list), len(epochs_list)))

        for index_net in range(0, len(self.nets_list)):
            net = 'net_' + str(index_net) + '.csv'
            # leggo le predizioni fatte con l'esnemble
            df = pd.read_csv(self.experiment_original_path + self.iperparameters['experiment_name'] + '/predictions/predictions_during_training/' + type + '/walk_' + str(index_walk) + '/' + net)

            # mergio con le label, così ho un subset del df con le date che mi servono e la predizione 
            df_merge_with_label = df_date_merger(df=df, columns=['date_time', 'delta_next_day', 'close', 'open', 'high', 'low'], dataset=self.iperparameters['predictions_dataset'], thr_hold=self.iperparameters['hold_labeling'])

            # RETURNS 
            ls_returns = []
            lh_returns = []
            sh_returns = []
            bh_returns = [] # BH

            # ROMADS
            ls_romads = []
            lh_romads = []
            sh_romads = []
            bh_romads = [] # BH

            # MDDS
            ls_mdds = []
            lh_mdds = []
            sh_mdds = []
            bh_mdds = [] # BH

            # PRECISIONI E LINEA RETTA DEL BILANCIAMENTO DELLE CLASSI
            longs_precisions = []
            shorts_precisions = []
            longs_label_coverage = []
            shorts_label_coverage = []

            # % DI OPERAZIONI FATTE
            long_operations = []
            short_operations = []
            hold_operations = []

            # POC
            longs_poc = []
            shorts_poc = []
            
            #accuracy
            accuracy = []

            label_coverage = Measures.get_delta_coverage(delta=df_merge_with_label['delta_next_day'].tolist())

            bh_equity_line, bh_global_return, bh_mdd, bh_romad, bh_i, bh_j  = Measures.get_return_mdd_romad_bh(close=df_merge_with_label['close'].tolist(), multiplier=self.iperparameters['return_multiplier'])

            dates_debug = df_merge_with_label['date_time'].tolist()

            #print("Type set:", type, "| Return BH per le date:", dates_debug[0], "-", dates_debug[-1], "|", bh_global_return)
            #input()
            # calcolo il return per un epoca
            for epoch in range(1, len(epochs_list) + 1): 
                df_epoch_rename = df_merge_with_label.copy()
                df_epoch_rename = df_epoch_rename.rename(columns={'epoch_' + str(epoch): 'decision'})

                ls_equity_line, ls_global_return, ls_mdd, ls_romad, ls_i, ls_j  = Measures.get_equity_return_mdd_romad(df=df_epoch_rename.copy(), multiplier=self.iperparameters['return_multiplier'], type='long_short', penalty=penalty, stop_loss=stop_loss, delta_to_use='delta_next_day')
                lh_equity_line, lh_global_return, lh_mdd, lh_romad, lh_i, lh_j  = Measures.get_equity_return_mdd_romad(df=df_epoch_rename.copy(), multiplier=self.iperparameters['return_multiplier'], type='long_only', penalty=penalty, stop_loss=stop_loss, delta_to_use='delta_next_day')
                sh_equity_line, sh_global_return, sh_mdd, sh_romad, sh_i, sh_j  = Measures.get_equity_return_mdd_romad(df=df_epoch_rename.copy(), multiplier=self.iperparameters['return_multiplier'], type='short_only', penalty=penalty, stop_loss=stop_loss, delta_to_use='delta_next_day')
                
                long, short, hold, general = Measures.get_precision_count_coverage(df=df_epoch_rename.copy(), multiplier=self.iperparameters['return_multiplier'], stop_loss=0, penalty=0, delta_to_use='delta_next_day')
                long_poc, short_poc = Measures.get_precision_over_coverage(df=df_epoch_rename.copy(), multiplier=self.iperparameters['return_multiplier'], stop_loss=0, penalty=0, delta_to_use='delta_next_day')

                # RETURNS 
                ls_returns.append(ls_global_return)
                lh_returns.append(lh_global_return)
                sh_returns.append(sh_global_return)
                bh_returns.append(bh_global_return) # BH

                # ROMADS
                ls_romads.append(ls_romad)
                lh_romads.append(lh_romad)
                sh_romads.append(sh_romad)
                bh_romads.append(bh_romad) # BH

                # MDDS
                ls_mdds.append(ls_mdd)
                lh_mdds.append(lh_mdd)
                sh_mdds.append(sh_mdd)
                bh_mdds.append(bh_mdd) # BH

                # PRECISIONI E LINEA RETTA DEL BILANCIAMENTO DELLE CLASSI
                longs_precisions.append(long['precision'])
                shorts_precisions.append(short['precision'])
                longs_label_coverage.append(label_coverage['long'])
                shorts_label_coverage.append(label_coverage['short'])

                # % di operazioni fatte
                long_operations.append(long['coverage'])
                short_operations.append(short['coverage'])
                hold_operations.append(hold['coverage'])

                # POC
                longs_poc.append(long_poc)
                shorts_poc.append(short_poc)
                accuracy.append(general['accuracy'])

            net_json = {
                "ls_returns": ls_returns,
                "lh_returns": lh_returns,
                "sh_returns": sh_returns,
                "bh_returns": bh_returns,

                "ls_romads": ls_romads,
                "lh_romads": lh_romads,
                "sh_romads": sh_romads,
                "bh_romads": bh_romads,

                "ls_mdds": ls_mdds,
                "lh_mdds": lh_mdds,
                "sh_mdds": sh_mdds,
                "bh_mdds": bh_mdds,

                "longs_precisions": longs_precisions,
                "shorts_precisions": shorts_precisions,
                "longs_label_coverage": longs_label_coverage,
                "shorts_label_coverage": shorts_label_coverage,

                "long_operations": long_operations,
                "short_operations": short_operations,
                "hold_operations": hold_operations,

                "longs_poc": longs_poc,
                "shorts_poc": shorts_poc,
                "accuracy": accuracy           
            }

            output_path = self.experiment_original_path + self.iperparameters['experiment_name'] + '/calculated_metrics/' + type + '/walk_' + str(index_walk) + '/' 

            create_folder(output_path)

            with open(output_path + 'net_' + str(index_net) + '.json', 'w') as json_file:
            #    json.dump(net_json, json_file, indent=4)
                json.dump(net_json, json_file)
            
            # PLOT SINGOLA RETE
            do_plot(metrics=net_json, walk=index_walk, epochs=len(epochs_list), main_path=self.experiment_original_path, experiment_name=self.iperparameters['experiment_name'], net=index_net, type=type)
            
            print(self.iperparameters['experiment_name'] + ' | ' + type + " - Salvate le metriche per walk n° ", index_walk, " rete: ", net)

            # RETURNS
            all_ls_returns[index_net] = ls_returns
            all_lh_returns[index_net] = lh_returns
            all_sh_returns[index_net] = sh_returns
            all_bh_return[index_net] = bh_returns # BH

            # ROMADS
            all_ls_romads[index_net] = ls_romads
            all_lh_romads[index_net] = lh_romads
            all_sh_romads[index_net] = sh_romads
            all_bh_romads[index_net] = bh_romads # BH

            # MDDS
            all_ls_mdds[index_net] = ls_mdds
            all_lh_mdds[index_net] = lh_mdds
            all_sh_mdds[index_net] = sh_mdds
            all_bh_mdds[index_net] = bh_mdds #BH

            # PRECISIONI E LINEA RETTA DEL BILANCIAMENTO DELLE CLASSI
            all_longs_precisions[index_net] = longs_precisions
            all_shorts_precisions[index_net] = shorts_precisions
            all_labels_longs_coverage[index_net] = longs_label_coverage
            all_labels_shorts_coverage[index_net] = shorts_label_coverage

            # % di operazioni fatte
            all_long_operations[index_net] = long_operations
            all_short_operations[index_net] = short_operations
            all_hold_operations[index_net] = hold_operations

            all_longs_poc[index_net] = longs_poc
            all_shorts_poc[index_net] = shorts_poc

            #accuracy
            all_accuracy[index_net] = accuracy

        # RETURNS
        avg_ls_returns = np.around(np.average(all_ls_returns, axis=0), decimals=3)
        avg_lh_returns = np.around(np.average(all_lh_returns, axis=0), decimals=3)
        avg_sh_returns = np.around(np.average(all_sh_returns, axis=0), decimals=3)
        avg_bh_returns = np.average(all_bh_return, axis=0) # BH

        # MDDS
        avg_ls_mdds = np.around(np.average(all_ls_mdds, axis=0), decimals=3)
        avg_lh_mdds = np.around(np.average(all_lh_mdds, axis=0), decimals=3)
        avg_sh_mdds = np.around(np.average(all_sh_mdds, axis=0), decimals=3)
        avg_bh_mdds = np.average(all_bh_mdds, axis=0) # BH

        # ROMADS
        avg_ls_romads = np.divide(avg_ls_returns, avg_ls_mdds, out=np.zeros_like(avg_ls_returns), where=avg_ls_mdds!=0)
        avg_lh_romads = np.divide(avg_lh_returns, avg_lh_mdds, out=np.zeros_like(avg_ls_returns), where=avg_ls_mdds!=0)
        avg_sh_romads = np.divide(avg_sh_returns, avg_sh_mdds, out=np.zeros_like(avg_ls_returns), where=avg_ls_mdds!=0)
        avg_bh_romads = np.divide(avg_bh_returns, avg_bh_mdds)

        # rimuovo i nan dai romads
        avg_ls_romads = np.around(np.nan_to_num(avg_ls_romads), decimals=3)
        avg_lh_romads = np.around(np.nan_to_num(avg_lh_romads), decimals=3)
        avg_sh_romads = np.around(np.nan_to_num(avg_sh_romads), decimals=3)
        avg_sh_romads[~np.isfinite(avg_sh_romads)] = 0

        avg_longs_precisions = np.around(np.average(all_longs_precisions, axis=0), decimals=3)
        avg_shorts_precisions = np.around(np.average(all_shorts_precisions, axis=0), decimals=3)

        avg_label_longs_coverage = np.around(np.average(all_labels_longs_coverage, axis=0), decimals=3)
        avg_label_shorts_coverage = np.around(np.average(all_labels_shorts_coverage, axis=0), decimals=3)

        # NUOVO TEST AVG POC
        avg_longs_poc = np.around(np.divide(avg_longs_precisions, avg_label_longs_coverage), decimals=3)
        avg_shorts_poc = np.around(np.divide(avg_shorts_precisions, avg_label_shorts_coverage), decimals=3)

        avg_longs_poc = (avg_longs_poc - 1 ) * 100
        
        #accuracy 
        avg_accuracy = np.around(np.average(all_accuracy, axis=0), decimals=3)

        '''
        for avg_id, avg in enumerate(avg_longs_poc):
            if avg_longs_poc[avg_id] < -30:
                avg_longs_poc[avg_id] = -30
            if avg_longs_poc[avg_id] > 30:
                avg_longs_poc[avg_id] = 30
        '''
        avg_shorts_poc = (avg_shorts_poc - 1 ) * 100
        '''
        for avg_id, avg in enumerate(avg_shorts_poc):
            if avg_shorts_poc[avg_id] < -30:
                avg_shorts_poc[avg_id] = -30
            if avg_shorts_poc[avg_id] > 30:
                avg_shorts_poc[avg_id] = 30
        '''
        avg_long_operations = np.average(all_long_operations, axis=0)
        avg_short_operations= np.average(all_short_operations, axis=0)
        avg_hold_operations = np.average(all_hold_operations, axis=0)

        avg_json = {
            "ls_returns": avg_ls_returns.tolist(),
            "lh_returns": avg_lh_returns.tolist(),
            "sh_returns": avg_sh_returns.tolist(),
            "bh_returns": avg_bh_returns.tolist(), # BH

            "ls_romads": avg_ls_romads.tolist(),
            "lh_romads": avg_lh_romads.tolist(),
            "sh_romads": avg_sh_romads.tolist(),
            "bh_romads": avg_bh_romads.tolist(), # BH

            "ls_mdds": avg_ls_mdds.tolist(),
            "lh_mdds": avg_lh_mdds.tolist(),
            "sh_mdds": avg_sh_mdds.tolist(),
            "bh_mdds": avg_bh_mdds.tolist(), # BH

            "longs_precisions": avg_longs_precisions.tolist(),
            "shorts_precisions": avg_shorts_precisions.tolist(),
            "longs_label_coverage": avg_label_longs_coverage.tolist(),
            "shorts_label_coverage": avg_label_shorts_coverage.tolist(),

            "long_operations": avg_long_operations.tolist(),
            "short_operations": avg_short_operations.tolist(),
            "hold_operations": avg_hold_operations.tolist(),

            "longs_poc": avg_longs_poc.tolist(),
            "shorts_poc": avg_shorts_poc.tolist(),

            "accuracy": avg_accuracy.tolist()
        }

        avg_output_path = self.experiment_original_path + self.iperparameters['experiment_name'] + '/calculated_metrics/' + type + '/average/' 
        
        create_folder(avg_output_path)

        with open(avg_output_path + 'walk_' + str(index_walk) + '.json', 'w') as json_file:
            json.dump(avg_json, json_file, indent=4)
            #json.dump(net_json, json_file)
        
        print(self.iperparameters['experiment_name'] + ' | ' + type + " - Salvate le metriche AVG per walk n° ", index_walk)

        do_plot(metrics=avg_json, walk=index_walk, epochs=len(epochs_list), main_path=self.experiment_original_path, experiment_name=self.iperparameters['experiment_name'], net='average', type=type)