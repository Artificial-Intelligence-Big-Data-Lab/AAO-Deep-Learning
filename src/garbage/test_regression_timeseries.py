# IMPORTING IMPORTANT LIBRARIES
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np 
import math
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import Dense, Activation, Input, GRU, Embedding
from keras.layers import LSTM
from classes.Market import Market

from keras.optimizers import RMSprop
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard, ReduceLROnPlateau
from keras.initializers import RandomUniform

# Definisco i data range per il training set di 11 walk
training_set = [
                ['2000-02-01', '2009-01-30'],
                ['2000-08-01', '2009-07-31'],
                ['2001-02-01', '2010-01-31'],
                ['2001-08-01', '2010-07-30'],
                ['2002-02-01', '2011-01-31'],
                ['2002-08-01', '2011-07-31'],
                ['2003-02-02', '2012-01-31'],
                ['2003-08-01', '2012-07-31'],
                ['2004-02-01', '2013-01-31'],
                ['2004-08-01', '2013-07-31'],
                ['2005-02-01', '2014-01-31']
            ]

# Definisco i data range per il validation set di 11 walk
validation_set = [
                ['2009-02-01', '2009-07-31'],
                ['2009-08-02', '2010-01-31'],
                ['2010-02-01', '2010-07-30'],
                ['2010-08-01', '2011-01-31'],
                ['2011-02-01', '2011-07-31'], 
                ['2011-08-01', '2012-01-31'],
                ['2012-02-01', '2012-07-31'],
                ['2012-08-01', '2013-01-31'],
                ['2013-02-01', '2013-07-31'],
                ['2013-08-01', '2014-01-31'],
                ['2014-02-02', '2014-07-31']   
            ]

# Definisco i data range per il test set di 11 walk
test_set = [
            ['2009-08-02', '2010-01-31'],
            ['2010-02-01', '2010-07-30'],
            ['2010-08-01', '2011-01-31'],
            ['2011-02-01', '2011-07-31'],
            ['2011-08-01', '2012-01-31'], 
            ['2012-02-01', '2012-07-31'],
            ['2012-08-01', '2013-01-31'],
            ['2013-02-01', '2013-07-31'],
            ['2013-08-01', '2014-01-31'],
            ['2014-02-02', '2014-07-31'],
            ['2014-08-01', '2015-01-30'] 
        ]

# FUNCTION TO CREATE 1D DATA INTO TIME SERIES DATASET
def new_dataset(dataset, step_size):
  data_X, data_Y = [], []
  for i in range(len(dataset)-step_size-1):
    a = dataset[i:(i+step_size), 0]
    data_X.append(a)
    data_Y.append(dataset[i + step_size, 0])
  return np.array(data_X), np.array(data_Y) 

def get_label_next_day_using_close(df):
    
    df['label'] = -1

    # Calcolo per ogni riga la Label, ovvero se il giorno dopo il mercato
    # salira' o scendera'
    df['label'] = (df['predicted'].diff().shift(-1) > 1).astype(int)
    
    # Converto lo 0 in -1, visto che lo 0 lo interpreteremo come hold
    df.loc[df['label'] == 0, ['label']] = -1
    # Invece di mettere la label sul giorno precedente, la mette sul giorno corrente. Puo' essere utile per altri scopi ma per ora e' deprecata
    #df['label'] = (df['close'].diff().bfill() > 0).astype(int)
    
    #columns.append('label')
    
    df = df.filter(['label'], axis=1) 

    return df

# Custom loss function
def loss_mse_warmup(y_true, y_pred):
    """
    Calculate the Mean Squared Error between y_true and y_pred,
    but ignore the beginning "warmup" part of the sequences.
    
    y_true is the desired output.
    y_pred is the model's output.
    """

    # The shape of both input tensors are:
    # [batch_size, sequence_length, num_y_signals].

    # Ignore the "warmup" parts of the sequences
    # by taking slices of the tensors.
    y_true_slice = y_true[:, warmup_steps:, :]
    y_pred_slice = y_pred[:, warmup_steps:, :]

    # These sliced tensors both have this shape:
    # [batch_size, sequence_length - warmup_steps, num_y_signals]

    # Calculate the MSE loss for each value in these tensors.
    # This outputs a 3-rank tensor of the same shape.
    loss = tf.losses.mean_squared_error(labels=y_true_slice,
                                        predictions=y_pred_slice)

    # Keras may reduce this across the first axis (the batch)
    # but the semantics are unclear, so to be sure we use
    # the loss across the entire tensor, we reduce it to a
    # single scalar with the mean function.
    loss_mean = tf.reduce_mean(loss)

    return loss_mean

# FOR REPRODUCIBILITY
np.random.seed(7)

# IMPORTING DATASET 
dataset = Market(dataset='sp500')
dataset = dataset.group(freq='1d', nan=False)
dataset = dataset.reset_index()

# CREATING OWN INDEX FOR FLEXIBILITY
obs = np.arange(1, len(dataset) + 1, 1)

# TAKING DIFFERENT INDICATORS FOR PREDICTION
dataset['OHLC'] = dataset[['high', 'low', 'close', 'open']].mean(axis = 1)
dataset['HLC'] = dataset[['high', 'low', 'close']].mean(axis = 1) # Non viene utilizzato!

''' #UTILE
# PLOTTING ALL INDICATORS IN ONE PLOT
plt.plot(dataset['date_time'].to_list(), dataset['OHLC'], 'r', label = 'OHLC avg')
plt.plot(dataset['date_time'].to_list(), dataset['HLC'], 'b', label = 'HLC avg')
plt.plot(dataset['date_time'].to_list(), dataset['close'], 'g', label = 'Closing price')
plt.legend(loc = 'upper right')
plt.show()
'''

# PREPARATION OF TIME SERIES DATASE
#OHLC_avg = np.reshape(dataset['OHLC'].values, (200,1)) # 1664

scaler = MinMaxScaler(feature_range=(0, 1))
dataset['OHLC_orig'] = dataset[['OHLC']]
dataset['OHLC'] = scaler.fit_transform(dataset[['OHLC']])

dataset['close_transformed'] = scaler.fit_transform(dataset[['close']])


# TRAIN-TEST SPLIT
#train_len = int(len(OHLC_avg) * 0.9)
#train_OHLC, test_OHLC = OHLC_avg[0:train_len,:], OHLC_avg[train_len:len(OHLC_avg),:]

step_size = 10

train = Market.get_df_by_data_range(df=dataset, start_date=training_set[1][0], end_date=validation_set[1][1])['close_transformed'].to_list()
val = Market.get_df_by_data_range(df=dataset, start_date=test_set[1][0], end_date=test_set[1][1])['close_transformed'].to_list()

train = np.reshape(train, (len(train),1)) 
val = np.reshape(val, (len(val),1)) 

# TIME-SERIES DATASET (FOR TIME T, VALUES FOR TIME T+1)
trainX, trainY = new_dataset(train, step_size)
testX, testY = new_dataset(val, step_size)

num_y_signals = len(trainY) + len(testY)

# RESHAPING TRAIN AND TEST DATA
trainX = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
testX = np.reshape(testX, (testX.shape[0], 1, testX.shape[1]))

# LSTM MODEL
model = Sequential()
model.add(LSTM(32, input_shape=(1, step_size), return_sequences = True))
model.add(LSTM(16))
model.add(Dense(1))
model.add(Activation('linear'))
model.compile(loss='mean_squared_error', optimizer='adagrad') # Try SGD, adam, adagrad and compare!!!
model.fit(trainX, trainY, epochs=1, batch_size=1, verbose=1)


# SECONDA RETE
#model = Sequential()
#model.add(GRU(units=512, return_sequences=True, input_shape=(None, (1, step_size))))
#model.add(Dense(num_y_signals, activation='sigmoid'))
#model.add(Dense(num_y_signals, activation='linear', kernel_initializer=RandomUniform(minval=-0.05, maxval=0.05)))
#optimizer = RMSprop(lr=1e-3)
#model.compile(loss=loss_mse_warmup, optimizer=optimizer)
#model.fit(trainX, trainY, epochs=10, batch_size=1, verbose=1)
#model.fit_generator(generator=generator,
#                    epochs=20,
#                    steps_per_epoch=100,
#                    validation_data=validation_data,
#                    callbacks=callbacks)

# PREDICTION
trainPredict = model.predict(trainX)
test_predict = model.predict(testX)


# DE-NORMALIZING FOR PLOTTING
trainPredict = scaler.inverse_transform(trainPredict)
trainY = scaler.inverse_transform([trainY])
test_predict = scaler.inverse_transform(test_predict)
testY = scaler.inverse_transform([testY])

# TRAINING RMSE
trainScore = math.sqrt(mean_squared_error(trainY[0], trainPredict[:,0]))
print('Train RMSE: %.2f' % (trainScore))
# TEST RMSE
testScore = math.sqrt(mean_squared_error(testY[0], test_predict[:,0]))
print('Test RMSE: %.2f' % (testScore))



date_df = Market.get_df_by_data_range(df=dataset, start_date=test_set[1][0], end_date=test_set[1][1])

print(date_df.shape)
print(test_predict.shape)
# Lista delle date per il plot
date_df = date_df.iloc[11:]
date = date_df['date_time'].to_list()
# Lista del close value originale per il plot
original = date_df['close'].to_list()

date_df['predicted'] = test_predict

date_df = date_df.set_index('date_time')
prova = get_label_next_day_using_close(df=date_df)

prova.to_csv('predizioni_da_regression.csv', header=True, index=True)

plt.plot(date, test_predict, 'b', label = 'Predicted validation set values')
plt.plot(date, original, 'r', label = 'Original validation set values')
plt.legend(loc = 'upper right')
plt.xlabel('Time in Days')
plt.ylabel('OHLC Value')
plt.show()
