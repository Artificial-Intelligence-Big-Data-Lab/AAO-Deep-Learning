from utils import *

import pandas as pd
import matplotlib.pylab as plt

from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.recurrent import LSTM, GRU
from keras.layers import Convolution1D, MaxPooling1D, AtrousConvolution1D, RepeatVector
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, CSVLogger
from keras.layers.wrappers import Bidirectional
from keras import regularizers
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import *
from keras.optimizers import RMSprop, Adam, SGD, Nadam
from keras.initializers import *
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import seaborn as sns

sns.despine()

def get_df_by_data_range(df, start_date, end_date):
    # Search mask
    mask = (df['date'] >= start_date) & (df['date'] <= end_date)
    # Get the subset of sp500
    return df.loc[mask]

def group(df, freq, nan=False):
    df['date_time'] = pd.to_datetime(df['date'] + ' ' + df['time'], format='%Y-%m-%d %H:%M:%S')

    # Get first Open, Last Close, Highest value of "High" and Lower value of "Low", and sum(Volume)
    grouped = df.drop(['date', 'time'], axis=1).groupby(pd.Grouper(key='date_time', freq=freq), sort=True).agg({
        'open': 'first',
        'close': 'last',
        'high': 'max',
        'low': 'min',
        'up': 'sum',
        'down': 'sum',
        'volume': 'sum'
    })
    
    if nan == True:
        return grouped
    else:
        return grouped.dropna()

dataset = 'AAPL1216'
dataset = 'sp500'

if dataset == 'AAPL1216':
    data_original = pd.read_csv('./data/' + dataset + '.csv')[::-1]
    openp = data_original.ix[:, 'Open'].tolist()
    highp = data_original.ix[:, 'High'].tolist()
    lowp = data_original.ix[:, 'Low'].tolist()
    closep = data_original.ix[:, 'Adj Close'].tolist()
    volumep = data_original.ix[:, 'Volume'].tolist()
else:
    data_original = pd.read_csv('./data/' + dataset + '.csv')
    
    data_original = get_df_by_data_range(data_original, '2003-01-03', '2013-12-31')

    data_original = group(data_original, '1d')
    
    openp = data_original.ix[:, 'open'].tolist()
    closep = data_original.ix[:, 'close'].tolist()
    highp = data_original.ix[:, 'high'].tolist()
    lowp = data_original.ix[:, 'low'].tolist()
    volumep = data_original.ix[:, 'volume'].tolist()

# data_chng = data_original.ix[:, 'Adj Close'].pct_change().dropna().tolist()

WINDOW = 10
EMB_SIZE = 5
STEP = 1
FORECAST = 1

X, Y = [], []
for i in range(0, len(data_original), STEP): 
    try:
        o = openp[i:i+WINDOW]
        h = highp[i:i+WINDOW]
        l = lowp[i:i+WINDOW]
        c = closep[i:i+WINDOW]
        v = volumep[i:i+WINDOW]

        o = (np.array(o) - np.mean(o)) / np.std(o)
        h = (np.array(h) - np.mean(h)) / np.std(h)
        l = (np.array(l) - np.mean(l)) / np.std(l)
        c = (np.array(c) - np.mean(c)) / np.std(c)
        v = (np.array(v) - np.mean(v)) / np.std(v)

        x_i = closep[i:i+WINDOW]
        y_i = closep[i+WINDOW+FORECAST]  

        last_close = x_i[-1]
        next_close = y_i

        if last_close < next_close:
            y_i = 0
        else:
            y_i = 1

        x_i = np.column_stack((o, h, l, c, v))

    except Exception as e:
        break

    X.append(x_i)
    Y.append(y_i)

X, Y = np.array(X), np.array(Y)

X_train, X_test, Y_train, Y_test = create_Xt_Yt(X, Y)

np.set_printoptions(threshold=np.inf)

X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], EMB_SIZE))
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], EMB_SIZE))


model = Sequential()
model.add(Convolution1D(input_shape = (WINDOW, EMB_SIZE),
                        nb_filter=16,
                        filter_length=4,
                        border_mode='same'))
model.add(BatchNormalization())
model.add(LeakyReLU())
model.add(Dropout(0.75))

model.add(Convolution1D(nb_filter=8,
                        filter_length=4,
                        border_mode='same'))
model.add(BatchNormalization())
model.add(LeakyReLU())
model.add(Dropout(0.75))

model.add(Flatten())

model.add(Dense(64))
model.add(BatchNormalization())
model.add(LeakyReLU())


model.add(Dense(1))
model.add(Activation('softmax'))

opt = Nadam(lr=0.001)

reduce_lr = ReduceLROnPlateau(monitor='val_acc', factor=0.9, patience=30, min_lr=0.000001, verbose=1)
checkpointer = ModelCheckpoint(filepath="lolkek.hdf5", verbose=1, save_best_only=True)


model.compile(optimizer=opt, loss='sparse_categorical_crossentropy',metrics=['accuracy'])

history = model.fit(X_train, Y_train, 
          nb_epoch = 1000, 
          batch_size = 64, 
          verbose=1, 
          validation_data=(X_test, Y_test),
          callbacks=[reduce_lr, checkpointer],
          shuffle=False)

model.load_weights("lolkek.hdf5")
pred = model.predict(np.array(X_test))

#print("CIAOOO")

#print(pred)

C = confusion_matrix([np.argmax(y) for y in Y_test], [np.argmax(y) for y in pred])

print(C / C.astype(np.float).sum(axis=1))

# Classification
# [[ 0.75510204  0.24489796]
#  [ 0.46938776  0.53061224]]


#for i in range(len(pred)):
    #print(Y_test[i], pred[i])


plt.figure(figsize=(15,5))

plt.subplot(1, 2, 1)

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title(dataset + ' Model Loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='best')

plt.subplot(1, 2, 2)

plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title(dataset + ' Model Accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='best')

plt.show()