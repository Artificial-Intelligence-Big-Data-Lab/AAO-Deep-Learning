    '''
    '
    '''
    def get_final_decision_without_ensemble(self, type='without_ensemble', validation_thr=15, epoch_selection_policy='long_short', validation_metric='romad'):
        print("Generating final decision for: " + type + "...")

        # dataset wrapper
        dataset = Market(dataset=self.dataset)
        dataset_label = dataset.get_label(freq='1d', columns=['open', 'close', 'delta_current_day', 'delta_next_day'])
        dataset_label = dataset_label.reset_index()
        dataset_label['date_time'] = dataset_label['date_time'].astype(str)

        df_global = pd.DataFrame(columns=['date_time', 'open', 'close', 'high', 'low', 'delta_current_day', 'delta_next_day', 'label'])

        output_path = self.final_decision_folder + 'without_ensemble/'

        if not os.path.isdir(output_path):
            os.makedirs(output_path)

        for walk in self.walks_list:
            reti = []

            choose_net, choose_val_idx, choose_romad, choose_return, choose_mdd = -1000, -1000, -1000, -100000, 100000

            for net in self.nets_list:
                df_net = pd.read_csv(self.original_predictions_validation_folder + walk + '/' + net)
                df_merge_with_label = pd.merge(df_net, dataset_label, how="inner")

                if type is 'without_ensemble':
                    val_idx, romad, global_return, mdd = self.get_max_idx_from_validation(df_merge_with_label=df_merge_with_label, validation_thr=validation_thr, epoch_selection_policy=epoch_selection_policy, metric='romad')
                if type is 'without_ensemble_last_epoch':
                    y_pred = df_merge_with_label['epoch_' + str(self.iperparameters['epochs'])].to_list()
                    delta = df_merge_with_label['delta_next_day'].to_list()

                    equity_line, global_return, mdd, romad, i, j = Measures.get_equity_return_mdd_romad(y_pred, delta, self.iperparameters['return_multiplier'], type='long_short')

                if  (validation_metric == 'romad' and romad > choose_romad and romad != 10.0) \
                or (validation_metric == 'return' and global_return > choose_return) \
                or (validation_metric == 'mdd' and mdd < choose_mdd and mdd != 0.001):
                    choose_val_idx = val_idx
                    choose_romad = romad
                    choose_net = net
                    choose_return = global_return
                    choose_mdd = mdd
                
            # for net in nets_list

            print("Walk: ", walk, " - Rete finale scelta: ", choose_net, " - Indice epoca:", choose_val_idx, " - Valore romad: ", choose_romad)
            # input()
            test_net = pd.read_csv(self.original_predictions_test_folder + walk + '/' + choose_net)
            # print(test_net)
            # input()
            df_test = pd.merge(test_net, dataset_label, how="inner")

            # CODICE DI PROVA PER PRECISION -> CAMBIATO IL SUBSET DELLE COLONNE, AGGIUNTO LABEL CHE CONTERRA' LA LABEL ORIGINALE
            df_test = df_test[['epoch_' + str(choose_val_idx), 'open', 'close', 'high', 'low', 'delta_current_day', 'delta_next_day', 'date_time']]
            
            if type is 'without_ensemble':
                df_test = df_test.rename(columns={"epoch_" + str(choose_val_idx): "decision"})

            if type is 'without_ensemble_last_epoch':
                df_test = df_test.rename(columns={'epoch_' + str(self.iperparameters['epochs']): "decision"})

            df_global = pd.concat([df_global, df_test], sort=True)

            df_global = df_global.drop_duplicates(subset='date_time', keep="first")

        # AGGIUNGERE UN GIORNO AL DATETIME INVECE CHE FARE LO SHIFT
        df_global['date_time'] = pd.to_datetime(df_global['date_time']) + timedelta(days=1)
        df_global = df_global[['date_time', 'decision']]
        df_global = df_global.drop(df_global.index[0])
        df_global['decision'] = df_global['decision'].astype(int)

        if type is 'without_ensemble':
            df_global.to_csv(output_path + 'decisions_without_ensemble.csv', header=True, index=False)

        if type is 'without_ensemble_last_epoch':
            df_global.to_csv(output_path + 'decision_with_last_epoch.csv', header=True, index=False)

