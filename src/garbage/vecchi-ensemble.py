    '''
    ' Calcolo le decisioni finali sull'ensemble el, facendo il long only usando la % 
    ' Viene calcolato sull'ensemble generato da __elimination_ensemble (quindi facendo +1 long -1 short)
    '''
    
    def get_final_decision_from_ensemble_el_longonly(self, perc_agreement, remove_nets=False, validation_return_thr=15):
        remove_nets_str = 'con-rimozione-reti/'

        if remove_nets is False:
            remove_nets_str = 'senza-rimozione-reti/'

        

        if not os.path.isdir(output_path):
            os.makedirs(output_path)

        df_global = pd.DataFrame(columns=['date_time', 'close', 'delta', 'label'])

        validation_input_path = self.ensemble_base_folder + '/' + remove_nets_str + '/validation/ensemble_el/'
        test_input_path = self.ensemble_base_folder + '/' + remove_nets_str + '/test/ensemble_el/'

        dataset = Market(dataset=self.dataset)
        dataset_label = dataset.get_label(freq='1d', columns=['open', 'close', 'delta_current_day', 'delta_next_day'])
        dataset_label = dataset_label.reset_index()
        dataset_label['date_time'] = dataset_label['date_time'].astype(str))

        for index_walk, walk in enumerate(self.walks_list):
            df_ensemble_val = pd.read_csv(validation_input_path + walk)
            df_merge_with_label = pd.merge(df_ensemble_val, dataset_label, how="inner")

            val_idx, romad, return_value = self.get_max_idx_from_validation(df_merge_with_label=df_merge_with_label, validation_return_thr=validation_return_thr, metric='romad')
            df_ensemble = pd.read_csv(test_input_path + walk)

            # mergio con le label, così ho un subset del df con le date che mi servono e la predizione
            df_merge_with_label = pd.merge(df_ensemble, dataset_label, how="inner")

            #df_merge_with_label = df_merge_with_label.set_index('index')
            subset_column = df_merge_with_label[['epoch_' + str(val_idx), 'close', 'delta', 'date_time']]
            subset_column = subset_column.rename(columns={"epoch_" + str(val_idx): "label"})

            df_global = pd.concat([df_global, subset_column], sort=True)

        df_global = df_global.drop_duplicates(subset='date_time', keep="first")

        df_global['label'] = df_global['label'].shift(1)
        #df_global = df_global.drop(columns=['close', 'delta'], axis=1)
        df_global = df_global[['date_time', 'label', 'close', 'delta']]
        df_global = df_global.drop(df_global.index[0])
        df_global['label'] = df_global['label'].astype(int)

        # CALCOLO DI QUANTO DEV'ESSERE IL VALORE DI LABEL PER ESSERE IN LINEA CON LA % DI AGREEMENT
        jolly_number = int((30 * (perc_agreement*100) / 100))
        df_global['label'] = df_global['label'].apply(lambda x: 2 if x > jolly_number else 1)

        df_global.to_csv(output_path + 'decisions_ensemble_el_' + str(perc_agreement) + '.csv', header=True, index=False)

    '''
    ' Calcolo le decisioni finali sull'ensemble el, facendo il long only usando la % 
    ' Viene calcolato sull'ensemble generato da __elimination_ensemble_longonly 
    '(quindi facendo +1 long e basta)
    '''
    def get_final_decision_from_ensemble_el_exclusive(self, num_agreement, remove_nets=False, validation_return_thr=15):
        remove_nets_str = 'con-rimozione-reti/'

        if remove_nets is False:
            remove_nets_str = 'senza-rimozione-reti/'

        output_path = self.final_decision_folder + 'ensemble_el_exclusive/' + remove_nets_str + '/'

        if not os.path.isdir(output_path):
            os.makedirs(output_path)

        df_global = pd.DataFrame(columns=['date_time', 'close', 'delta', 'label'])

        validation_input_path = self.ensemble_base_folder + '/' + remove_nets_str + '/validation/ensemble_el_exclusive/'
        test_input_path = self.ensemble_base_folder + '/' + remove_nets_str + '/test/ensemble_el_exclusive/'
        # dataset wrapper
        dataset = Market(dataset=self.dataset)
        dataset_label = dataset.get_label_next_day(freq='1d', columns=['open', 'close', 'delta'])
        dataset_label = dataset_label.reset_index()
        dataset_label['date_time'] = dataset_label['date_time'].astype(str)

        for index_walk, walk in enumerate(self.walks_list):
            df_ensemble_val = pd.read_csv(validation_input_path + walk)
            df_merge_with_label = pd.merge(df_ensemble_val, dataset_label, how="inner")

            val_idx, romad, return_value = self.get_max_idx_from_validation(df_merge_with_label=df_merge_with_label, validation_return_thr=validation_return_thr, metric='romad')
            df_ensemble = pd.read_csv(test_input_path + walk)

            # mergio con le label, così ho un subset del df con le date che mi servono e la predizione
            df_merge_with_label = pd.merge(df_ensemble, dataset_label, how="inner")
            #df_merge_with_label = df_merge_with_label.set_index('index')
            subset_column = df_merge_with_label[['epoch_' + str(val_idx), 'close', 'delta', 'date_time']]
            subset_column = subset_column.rename(columns={"epoch_" + str(val_idx): "label"})

            df_global = pd.concat([df_global, subset_column], sort=True)

        df_global = df_global.drop_duplicates(subset='date_time', keep="first")

        df_global['label'] = df_global['label'].shift(1)
        #df_global = df_global.drop(columns=['close', 'delta'], axis=1)
        df_global = df_global[['date_time', 'label', 'close', 'delta']]
        df_global = df_global.drop(df_global.index[0])
        df_global['label'] = df_global['label'].astype(int)

        # CALCOLO DI QUANTO DEV'ESSERE IL VALORE DI LABEL PER ESSERE IN LINEA CON LA % DI AGREEMENT
        df_global['label'] = df_global['label'].apply(lambda x: 2 if x > num_agreement else 1)

        df_global.to_csv(output_path + 'decisions_ensemble_exclusive_' + str(num_agreement) + '.csv', header=True, index=False)
