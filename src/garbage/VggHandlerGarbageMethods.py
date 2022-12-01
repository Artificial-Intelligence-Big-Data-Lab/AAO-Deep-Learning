    '''
    '
    '''
    def __train_1D(self, x_train, y_train, x_val, y_val, index_net, index_walk):
        # binarizing labels
        lb = LabelBinarizer()
        y_train = lb.fit_transform(y_train)
        y_val = lb.fit_transform(y_val)

        # [INFO] compiling model...
        model = SmallerVGGNet.build_vgg16_1d_smaller(height=x_train.shape[0], width=x_train.shape[1],
                                    classes=len(lb.classes_), init_var=index_net)

        opt = Adam(lr=self.__init_lr, decay=(self.__init_lr/ self.__epochs))
        model.compile(loss="sparse_categorical_crossentropy", optimizer=opt, metrics=["accuracy"])
        
        # model info
        #model.summary()
        #print(opt)

        #[INFO] training network...
        #filepath = self.__output_folder + 'models/epochs/' +"weights-improvement-{epoch:02d}-{val_acc:.2f}.hdf5"
        
        # Salvo per ogni epoca il modello
        filepath = self.__output_folder + 'models/model_foreach_epoch/' + "walk_" + str(index_walk) + "/net_" + str(index_net) + "/"
        # creo la cartella di output per gli esperimenti
        if not os.path.isdir(filepath):
            os.makedirs(filepath)

        # Se voglio salvare la storia dei modelli creo la callback 
        if self.__save_model_history == True: 
            filename =  "epoch_{epoch:02d}.model"
            # period indica ogni quanto salvare il modello 
            checkpoint = ModelCheckpoint(filepath + filename, monitor='val_loss', verbose=0, save_best_only=False, save_weights_only=False, mode='auto', period=self.__model_history_period)
            callbacks_list = [checkpoint]
            
            H = model.fit(X_train, y_train, batch_size=self.__bs, validation_data=(X_val, y_val), epochs=self.__epochs, verbose=1, callbacks=callbacks_list)
       
        # Non voglio salvare la storia dei modelli ogni tot epoche, salvo solo l'ultimo
        if self.__save_model_history == False:
            H = model.fit(X_train, y_train, batch_size=self.__bs, validation_data=(X_val, y_val), epochs=self.__epochs, verbose=1)
       
        return model, H

    '''
    '
    '''
    def __train_again(self, x_train, y_train, x_val, y_val, index_net, index_walk, model_filename):
        # Carico le immagini
        X_train = x_train.astype('float32') / 255
        X_val = x_val.astype('float32') / 255

        # binarizing labels
        lb = LabelBinarizer()
        y_train = lb.fit_transform(y_train)
        y_val = lb.fit_transform(y_val)

        # [INFO] compiling model...
        #model = SmallerVGGNet.build(height=self.__input_shape[0], width=self.__input_shape[1], depth=self.__input_shape[2],
        #                            classes=len(lb.classes_), init_var=index_net)

        model = load_model(model_filename)
        opt = Adam(lr=self.__init_lr, decay=(self.__init_lr/ self.__epochs))
        model.compile(loss="sparse_categorical_crossentropy", optimizer=opt, metrics=["accuracy"])

        # model info
        #model.summary()
        #print(opt)

        #[INFO] training network...
        #filepath = self.__output_folder + 'models/epochs/' +"weights-improvement-{epoch:02d}-{val_acc:.2f}.hdf5"
        
        # Salvo per ogni epoca il modello
        filepath = self.__output_folder + 'models/model_foreach_epoch/' + "walk_" + str(index_walk) + "/net_" + str(index_net) + "/"
        # creo la cartella di output per gli esperimenti
        if not os.path.isdir(filepath):
            os.makedirs(filepath)

        # Se voglio salvare la storia dei modelli creo la callback 
        if self.__save_model_history == True: 
            filename =  "epoch_{epoch:02d}.model"
            # period indica ogni quanto salvare il modello 
            checkpoint = ModelCheckpoint(filepath + filename, monitor='val_loss', verbose=0, save_best_only=False, save_weights_only=False, mode='auto', period=self.__model_history_period)
            callbacks_list = [checkpoint]
            
            H = model.fit(X_train, y_train, batch_size=self.__bs, validation_data=(x_val, y_val), epochs=self.__epochs, verbose=1, callbacks=callbacks_list)
       
        # Non voglio salvare la storia dei modelli ogni tot epoche, salvo solo l'ultimo
        if self.__save_model_history == False:
            H = model.fit(X_train, y_train, batch_size=self.__bs, validation_data=(x_val, y_val), epochs=self.__epochs, verbose=1)
       
        return model, H


    '''
    '
    '''
    def __train_small(self, x_train, y_train, x_val, y_val):

        # Carico le immagini
        X_train = x_train.astype('float32') / 255
        X_val = x_val.astype('float32') / 255

        # binarizing labels
        lb = LabelBinarizer()
        y_train = lb.fit_transform(y_train)
        y_val = lb.fit_transform(y_val)

        # [INFO] compiling model...
        model = SmallerVGGNet.build_small(height=40, width=40, depth=3)

        # compile model
        opt = SGD(lr=0.001, momentum=0.9)
        model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])

        H = model.fit(X_train, y_train, batch_size=self.__bs, validation_data=(x_val, y_val), epochs=self.__epochs, verbose=1)
        
        return model, H
   '''
    '
    '''
    def get_1d_x_vector(self, index_walk, vector_len, start_date, end_date):
        print("Walk " + str(index_walk) + " - Calcolo il vettore x per le date " + str(start_date) + " - " + str(end_date))

        dataset = Market(dataset=self.predictions_dataset)

        d1_df = dataset.group(freq='1d', nan=False)
        h1_df = dataset.group(freq='1h', nan=False)
        h4_df = dataset.group(freq='4h', nan=False)
        h8_df = dataset.group(freq='8h', nan=False)

        x = np.zeros(shape=(vector_len, 20), dtype=np.uint8)

        # TRAINING
        start_date = pd.to_datetime(start_date)
        end_date = pd.to_datetime(end_date)

        d1_df_selected = Market.get_df_until_data(d1_df, end_date)
        h1_df_selected = Market.get_df_until_data(h1_df, end_date)
        h4_df_selected = Market.get_df_until_data(h4_df, end_date)
        h8_df_selected = Market.get_df_until_data(h8_df, end_date)
        
        index_x = 0

        for i, (idx, row) in enumerate(d1_df_selected.iterrows()):

            if row['date_time'] < start_date:
                continue

            d1_subset = Market.get_df_until_data(d1_df_selected, row['date_time'])
            h1_subset = Market.get_df_until_data(h1_df_selected, row['date_time'])
            h4_subset = Market.get_df_until_data(h4_df_selected, row['date_time'])
            h8_subset = Market.get_df_until_data(h8_df_selected, row['date_time'])
            
            
            d1_df_range = d1_subset.tail(20)
            h1_df_range = h1_subset.tail(20)
            h4_df_range = h4_subset.tail(20)
            h8_df_range = h8_subset.tail(20)

            
            #x[index_x] = d1_df_range['delta'].to_list() + h1_df_range['delta'].to_list() + h4_df_range#['delta'].to_list() + h8_df_range['delta'].to_list() 
            x[index_x] = d1_df_range['delta'].to_list() 
            
            index_x = index_x + 1

        return x

        
    '''
    '
    '''
    def run_1D(self):
        # salvo la prima parte dei log
        self.__start_log()
        
        # Per ogni walk calcolo train, val e test set (x,y)
        for index_walk in range(self.__number_of_walks):

            dataset = Market(dataset=self.predictions_dataset)

            day_df = dataset.get_label_next_day(freq='1d')

            y_train =  Market.get_df_by_data_range(day_df, self.__training_set[index_walk][0], self.__training_set[index_walk][1])['label'].to_list()
            y_val =  Market.get_df_by_data_range(day_df, self.__validation_set[index_walk][0], self.__validation_set[index_walk][1])['label'].to_list()
            y_test =  Market.get_df_by_data_range(day_df, self.__test_set[index_walk][0], self.__test_set[index_walk][1])['label'].to_list()

           
            x_train = self.get_1d_x_vector(index_walk=index_walk, vector_len=len(y_train), start_date=self.__training_set[index_walk][0], end_date=self.__training_set[index_walk][1])
            x_val = self.get_1d_x_vector(index_walk=index_walk, vector_len=len(y_val), start_date=self.__validation_set[index_walk][0], end_date=self.__validation_set[index_walk][1])
            x_test = self.get_1d_x_vector(index_walk=index_walk, vector_len=len(y_test), start_date=self.__test_set[index_walk][0], end_date=self.__test_set[index_walk][1])
           

            x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], 1)
            x_val = x_val.reshape(x_val.shape[0], x_val.shape[1], 1)
            x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], 1)

            # per ogni rete effettuo il fit del modello e salvo .model, .pkl ed i plots
            for index_net in range(self.__number_of_nets):
                
                # Debug
                print("TRAINING - INDEX NET: ", str(index_net), " INDEX WALK: " + str(index_walk))

                # effettuo il training del modello
                model, H = self.__train_1D(x_train=x_train, y_train=y_train, x_val=x_val, y_val=y_val, index_net=index_net, index_walk=index_walk)

                # Salvo il modello ed i grafici della rete
                self.__save_plots(H=H, index_walk=index_walk, index_net=index_net)
                self.__save_models(model=model, H=H, index_walk=index_walk, index_net=index_net)
                
                K.clear_session()
                
                
        # salvo l'ultima parte del log
        self.__end_log()
        

    '''
    '
    '''
    def run_small(self):
        # salvo la prima parte dei log
        self.__start_log()
        
        # Per ogni walk calcolo train, val e test set (x,y)
        for index_walk in range(self.__number_of_walks):

            training, validation, test = self.__get_train_val_test(training_set=self.__training_set[index_walk],
                                                                                        validation_set=self.__validation_set[index_walk],
                                                                                        test_set=self.__test_set[index_walk],
                                                                                        df_label_type='next_day')
            
            # Debug
            print("TRAINING - INDEX WALK: " + str(index_walk) + " with small CNN")

            # effettuo il training del modello
            model, H = self.__train_small(x_train=training.get_x(), y_train=training.get_y(), x_val=validation.get_x(), y_val=validation.get_y())

            # Salvo il modello ed i grafici della rete
            self.__save_plots(H=H, index_walk=index_walk, index_net=index_net)
            self.__save_models(model=model, H=H, index_walk=index_walk, index_net=index_net)
            
            K.clear_session()
                
                
        # salvo l'ultima parte del log
        self.__end_log()

    '''
    '
    '''
    def run_again(self, model_input_folder):
        # salvo la prima parte dei log
        self.__start_log()
        
        # Per ogni walk calcolo train, val e test set (x,y)
        for index_walk in range(self.__number_of_walks):

            training, validation, test = self.__get_train_val_test(training_set=self.__training_set[index_walk],
                                                                                        validation_set=self.__validation_set[index_walk],
                                                                                        test_set=self.__test_set[index_walk],
                                                                                        df_label_type='next_day')
            # per ogni rete effettuo il fit del modello e salvo .model, .pkl ed i plots
            for index_net in range(self.__number_of_nets):
                # Debug
                print("TRAINING - INDEX NET: ", str(index_net), " INDEX WALK: " + str(index_walk))

                # carico il modello per utilizzarlo successivamente per calcolare l'a classe di 'output
                model_filename = self.__output_base_path + model_input_folder + '/' + 'models/walk_' + str(index_walk) + '_net_'+ str(index_net) + '.model'

                # effettuo il training del modello
                model, H = self.__train_again(x_train=training.get_x(), y_train=training.get_y(), x_val=validation.get_x(), y_val=validation.get_y())

                # Salvo il modello ed i grafici della rete
                self.__save_plots(H=H, index_walk=index_walk, index_net=index_net)
                self.__save_models(model=model, H=H, index_walk=index_walk, index_net=index_net)
                
                K.clear_session()
                
                
        # salvo l'ultima parte del log
        self.__end_log()

    '''
    ' Questo metodo stampa all'interno della cartella predictions l'output della rete allenata
    ' E' possibile stampare l'output sia per il training, validation e test. set_type è appunto il parametro
    ' di ingresso per stabilire quale degli output stampati.
    ' Il csv stampato viene generato usando la label del giorno corrente. 
    '''
    def get_predictions_2D(self, set_type):

        # Controllo che il parametro sia valido
        if set_type  is not 'validation' and set_type is not 'test':
            sys.exit('VggHandler.get_predictions: set_type must be validation or test')

        # Dove andrò a salvare il file, dentro la cartella predictions
        folder = self.__output_folder + 'predictions/' + set_type + '/'
            
        # Se le cartella del dataset non esistono, la creo a runtime
        if not os.path.isdir(folder):
            os.makedirs(folder)

        # Per ogni walk calcolo train, val e test set (x,y)
        for index_walk in range(0, self.__number_of_walks):

            # Calcolo tutti i vari set per questo walk
            training, validation, test = self.__get_train_val_test(
                                                                                    training_set=self.__training_set[index_walk],
                                                                                    validation_set=self.__validation_set[index_walk],
                                                                                    test_set=self.__test_set[index_walk],
                                                                                    df_label_type='next_day')

            # A seconda del parametro passato al metodo setto dentro walk_df il set desiderato
            # x_set e y_set li uso come variabile generale, settati con una delle variabili generate sopra.
            # lo faccio per rendere il metodo più generale, così successivamente uso solo x/y_set
            if set_type == "validation":
                walk_df = self.__get_df_by_data_range(df=self.__date_label_df, 
                                                        start_date=self.__validation_set[index_walk][0], 
                                                        end_date=self.__validation_set[index_walk][1])
                x_set = validation.get_x()
                y_set = validation.get_y()

            if set_type == "test":
                walk_df = self.__get_df_by_data_range(df=self.__date_label_df, 
                                                        start_date=self.__test_set[index_walk][0], 
                                                        end_date=self.__test_set[index_walk][1])
                x_set = test.get_x()
                y_set = test.get_y()

            # imposto come sempre date_time come indice del dataframe
            walk_df = walk_df.set_index('date_time')
            
            # imposto il nome del file di output aggiungendogli la path 
            walkname = folder + 'GADF_walk_' + str(index_walk) + ".csv"
            
            # per ogni rete carico modello preallenato e genero le predizioni per ogni giorno
            for index_net in range(0, self.__number_of_nets):
                print("WALK:" + str(index_walk) + ' - NET: ' + str(index_net))

                # Calcolo la grandezza del x_set in modo da crearmi poi un array vuoto di quella dimensione (per metterci dentro le predizioni)
                set_size = x_set.shape[0]
                preds = np.zeros(shape=set_size, dtype=np.uint8)
                
                # carico il modello per utilizzarlo successivamente per calcolare l'a classe di 'output
                model_filename = self.__output_folder + 'models/walk_' + str(index_walk) + '_net_'+ str(index_net) + '.model'
                #model = load_model(model_filename)
                model = load_model(model_filename, custom_objects={'RAdam': RAdam})


                ''' Utilizzo le probabilità, al momento non utilizzato
                probs = np.zeros(shape=set_size, dtype=np.uint8)
                probs = model.predict(x_set)
                print(probs)
                preds = np.argmax(probs, axis=1)
                preds[preds == 0] = -1
                '''

                # Calcolo la classe e genero l'array delle predizioni
                preds = model.predict_classes(x_set)
                
                # Cambio la classe 0 con -1, in quanto le short in tutto il framework sono indicate con -1
                preds[preds == 0] = -1
                
                # aggiungo la colonna net_N con le predizioni appena calcolate 
                walk_df['net_' + str(index_net)] = preds

                # Pulisco la sessione per evitare rallentamenti ad ogni load
                K.clear_session()
            # Salvo le predizioni con tutte le reti incolonnate
            walk_df.to_csv(walkname, header=True, index=True)

    '''
    '
    '''
    def get_predictions_1D(self, set_type):

        # Controllo che il parametro sia valido
        if set_type  is not 'validation' and set_type is not 'test':
            sys.exit('VggHandler.get_predictions: set_type must be validation or test')

        # Dove andrò a salvare il file, dentro la cartella predictions
        folder = self.__output_folder + 'predictions/' + set_type + '/'
            
        # Se le cartella del dataset non esistono, la creo a runtime
        if not os.path.isdir(folder):
            os.makedirs(folder)

        # Per ogni walk calcolo train, val e test set (x,y)
        for index_walk in range(0, self.__number_of_walks):
            
            dataset = Market(dataset=self.predictions_dataset)

            day_df = dataset.get_label_next_day(freq='1d')

            y_train =  Market.get_df_by_data_range(day_df, self.__training_set[index_walk][0], self.__training_set[index_walk][1])['label'].to_list()
            y_val =  Market.get_df_by_data_range(day_df, self.__validation_set[index_walk][0], self.__validation_set[index_walk][1])['label'].to_list()
            y_test =  Market.get_df_by_data_range(day_df, self.__test_set[index_walk][0], self.__test_set[index_walk][1])['label'].to_list()

           
            x_train = self.get_1d_x_vector(index_walk=index_walk, vector_len=len(y_train), start_date=self.__training_set[index_walk][0], end_date=self.__training_set[index_walk][1])
            x_val = self.get_1d_x_vector(index_walk=index_walk, vector_len=len(y_val), start_date=self.__validation_set[index_walk][0], end_date=self.__validation_set[index_walk][1])
            x_test = self.get_1d_x_vector(index_walk=index_walk, vector_len=len(y_test), start_date=self.__test_set[index_walk][0], end_date=self.__test_set[index_walk][1])

            x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], 1)
            x_val = x_val.reshape(x_val.shape[0], x_val.shape[1], 1)
            x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], 1)

            # A seconda del parametro passato al metodo setto dentro walk_df il set desiderato
            # x_set e y_set li uso come variabile generale, settati con una delle variabili generate sopra.
            # lo faccio per rendere il metodo più generale, così successivamente uso solo x/y_set
            if set_type == "validation":
                walk_df = self.__get_df_by_data_range(df=self.__date_label_df, 
                                                        start_date=self.__validation_set[index_walk][0], 
                                                        end_date=self.__validation_set[index_walk][1])
                x_set = x_val
                y_set = y_val

            if set_type == "test":
                walk_df = self.__get_df_by_data_range(df=self.__date_label_df, 
                                                        start_date=self.__test_set[index_walk][0], 
                                                        end_date=self.__test_set[index_walk][1])
                x_set = x_test
                y_set = y_test

            # imposto come sempre date_time come indice del dataframe
            walk_df = walk_df.set_index('date_time')

            # imposto il nome del file di output aggiungendogli la path 
            walkname = folder + 'GADF_walk_' + str(index_walk) + ".csv"
            
            # per ogni rete carico modello preallenato e genero le predizioni per ogni giorno
            for index_net in range(0, self.__number_of_nets):
                print("WALK:" + str(index_walk) + ' - NET: ' + str(index_net))
                
                # Calcolo la grandezza del x_set in modo da crearmi poi un array vuoto di quella dimensione (per metterci dentro le predizioni)
                set_size = x_set.shape[0]
                preds = np.zeros(shape=set_size, dtype=np.uint8)
                
                # carico il modello per utilizzarlo successivamente per calcolare l'a classe di 'output
                model_filename = self.__output_folder + 'models/walk_' + str(index_walk) + '_net_'+ str(index_net) + '.model'
                model = load_model(model_filename)

                # Calcolo la classe e genero l'array delle predizioni
                preds = model.predict_classes(x_set)
                
                # Cambio la classe 0 con -1, in quanto le short in tutto il framework sono indicate con -1
                preds[preds == 0] = -1
                
                # aggiungo la colonna net_N con le predizioni appena calcolate 
                walk_df['net_' + str(index_net)] = preds

                # Pulisco la sessione per evitare rallentamenti ad ogni load
                K.clear_session()
            # Salvo le predizioni con tutte le reti incolonnate
            walk_df.to_csv(walkname, header=True, index=True)
    '''
    '
    '''
    def get_predictions_foreach_epoch(self, set_type):

        # Controllo che il parametro sia valido
        if set_type  is not 'validation' and set_type is not 'test':
            sys.exit('VggHandler.get_predictions: set_type must be validation or test')

        # Dove andrò a salvare il file, dentro la cartella predictions
        folder = self.__output_folder + 'predictions_foreach_model/' + set_type + '/'
            
        # Se le cartella del dataset non esistono, la creo a runtime
        if not os.path.isdir(folder):
            os.makedirs(folder)

        # Per ogni walk calcolo train, val e test set (x,y)
        for index_walk in range(0, self.__number_of_walks):

            # Calcolo tutti i vari set per questo walk
            training, validation, test = self.__get_train_val_test(
                                                                    training_set=self.__training_set[index_walk],
                                                                    validation_set=self.__validation_set[index_walk],
                                                                    test_set=self.__test_set[index_walk],
                                                                    df_label_type='next_day')

            # A seconda del parametro passato al metodo setto dentro walk_df il set desiderato
            # x_set e y_set li uso come variabile generale, settati con una delle variabili generate sopra.
            # lo faccio per rendere il metodo più generale, così successivamente uso solo x/y_set
            if set_type == "validation":
                walk_df = self.__get_df_by_data_range(df=self.__date_label_df, 
                                                        start_date=self.__validation_set[index_walk][0], 
                                                        end_date=self.__validation_set[index_walk][1])
                x_set = validation.get_x()
                y_set = validation.get_y()

            if set_type == "test":
                walk_df = self.__get_df_by_data_range(df=self.__date_label_df, 
                                                        start_date=self.__test_set[index_walk][0], 
                                                        end_date=self.__test_set[index_walk][1])
                x_set = test.get_x()
                y_set = test.get_y()

            # imposto come sempre date_time come indice del dataframe
            walk_df = walk_df.set_index('date_time')
            
            # imposto il nome del file di output aggiungendogli la path 
            walkname = folder + 'GADF_walk_' + str(index_walk) + ".csv"
            
            index_net = 0
                
            starting = self.__model_history_period
            acc = self.__model_history_period

            while starting <= self.__epochs:
                    # imposto il nome del file di output aggiungendogli la path 
                walkname = folder + 'GADF_epochs_' + str(starting) + ".csv"
                print("WALK:" + str(index_walk) + ' - NET: ' + str(index_net) + ' EPOCH: ' +  str(starting))

                # Calcolo la grandezza del x_set in modo da crearmi poi un array vuoto di quella dimensione (per metterci dentro le predizioni)
                set_size = x_set.shape[0]
                preds = np.zeros(shape=set_size, dtype=np.uint8)
                
                # carico il modello per utilizzarlo successivamente per calcolare l'a classe di 'output
                model_filename = self.__output_folder + 'models/model_foreach_epoch/walk_' + str(index_walk) + '/net_'+ str(index_net) + '/epoch_' + str(starting) + '.model'
                model = load_model(model_filename)


                # Calcolo la classe e genero l'array delle predizioni
                preds = model.predict_classes(x_set)
                
                # Cambio la classe 0 con -1, in quanto le short in tutto il framework sono indicate con -1
                preds[preds == 0] = -1
                
                # aggiungo la colonna net_N con le predizioni appena calcolate 
                walk_df['net_' + str(index_net)] = preds

                # Pulisco la sessione per evitare rallentamenti ad ogni load
                K.clear_session()

                starting += acc
                # Salvo le predizioni con tutte le reti incolonnate
                walk_df.to_csv(walkname, header=True, index=True)

    