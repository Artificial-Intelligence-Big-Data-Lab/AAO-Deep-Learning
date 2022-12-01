from keras import backend as K
from keras import losses
import tensorflow as tf
from keras.regularizers import l2
import numpy as np 
import logging

def stop_loss_custom(weights):

    #def my_func(arg):
    #    arg = tf.convert_to_tensor(arg, dtype=tf.float32)
    #    return tf.matmul(arg, arg) + arg

    def loss_function(y_true,y_pred):

        '''
        x = np.zeros((3,3))
        y = np.zeros((3,3))
        w = np.ones(3)

        x[0][2] = 1
        x[1][1] = 1
        x[2][1] = 1

        #print(x)

        y[0][0] = 1
        y[1][1] = 1
        y[2][0] = 1

        w[1] = 2
        w[2] = 5

        y_true = x
        y_pred = y
        '''

        
        #sess = tf.InteractiveSession()
        #y_true = my_func(y_true)
        #y_pred = my_func(y_pred)
        #sess = K.get_session()
        #y_true = y_true.eval(session=sess)
        #y_pred = y_pred.eval(session=sess)

        #y_true = y_true.numpy()
        #y_pred = y_pred.numpy()

        #loss = np.zeros(y_true.shape[1])

        #for i in range(0, y_true.shape[1]):
        #    y_pred[i] = np.abs(K.sum((y_true[i] - y_pred[i]) * np.array([0,1,2]))) * w[i]

        #mean_loss = K.constant(K.eval(1.))

        original_pred = y_pred

        y_true = np.ones(2791)
        y_pred = np.ones(2791)


        #y_true = K.eval(y_true)
        #y_pred = K.eval(y_pred)

        loss = np.zeros(2791)

        for i in range(0, 2791):
            loss[i] = np.abs(np.sum((y_true[i] - y_pred[i]) * np.array([0,1,2]))) * weights[i]

        mean_loss = np.sum(loss)

        return K.sum(mean_loss + original_pred)
    return loss_function