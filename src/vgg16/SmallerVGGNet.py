# import the necessary packages
from keras.models import Sequential
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Conv2D, Conv1D
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D, Convolution1D, MaxPooling1D, ZeroPadding1D
from keras.layers.core import Activation
from keras.layers.core import Flatten
from keras.layers.core import Dropout
from keras.layers.core import Dense
from keras.initializers import *
from keras.optimizers import SGD
from keras.layers import Input
from keras.layers.merge import concatenate
from keras.models import Model
#width : The image width dimension.
#height : The image height dimension.
#depth : The depth of the image also known as the number of channels.
#classes : The number of classes in our dataset (which will affect the last layer of our model).
class SmallerVGGNet:
    '''
        0: Orthogonal(gain=1.0, seed=None),
        1: lecun_uniform(seed=None),
        2: VarianceScaling(scale=1.0, mode='fan_in', distribution='normal', seed=None),
        3: RandomNormal(mean=0.0, stddev=0.05, seed=None),
        4: RandomUniform(minval=-0.05, maxval=0.05, seed=None),
        5: TruncatedNormal(mean=0.0, stddev=0.05, seed=None),
        6: glorot_normal(seed=None),
        7: glorot_uniform(seed=None),
        8: he_normal(seed=None),
        9: he_uniform(seed=None), 
        10: Orthogonal(gain=1.0, seed=42),
        11: lecun_uniform(seed=42),
        12: VarianceScaling(scale=1.0, mode='fan_in', distribution='normal', seed=42),
        13: RandomNormal(mean=0.0, stddev=0.05, seed=42),
        14: RandomUniform(minval=-0.05, maxval=0.05, seed=42),
        15: TruncatedNormal(mean=0.0, stddev=0.05, seed=42),
        16: glorot_normal(seed=42),
        17: glorot_uniform(seed=42),
        18: he_normal(seed=42),
        19: he_uniform(seed=42)
    ''' 

    switcher = {
        0: Orthogonal(gain=1.0, seed=42),
        1: Orthogonal(gain=1.0, seed=42),
        2: Orthogonal(gain=1.0, seed=42),
        3: Orthogonal(gain=1.0, seed=42),
        4: Orthogonal(gain=1.0, seed=42),
        5: Orthogonal(gain=1.0, seed=42),
        6: Orthogonal(gain=1.0, seed=42),
        7: Orthogonal(gain=1.0, seed=42),
        8: Orthogonal(gain=1.0, seed=42),
        9: Orthogonal(gain=1.0, seed=42),
        10: Orthogonal(gain=1.0, seed=42),
        11: Orthogonal(gain=1.0, seed=42),
        12: Orthogonal(gain=1.0, seed=42),
        13: Orthogonal(gain=1.0, seed=42),
        14: Orthogonal(gain=1.0, seed=42),
        15: Orthogonal(gain=1.0, seed=42),
        16: Orthogonal(gain=1.0, seed=42),
        17: Orthogonal(gain=1.0, seed=42),
        18: Orthogonal(gain=1.0, seed=42),
        19: Orthogonal(gain=1.0, seed=42),
        20: Orthogonal(gain=1.0, seed=42),
        21: Orthogonal(gain=1.0, seed=42),
        22: Orthogonal(gain=1.0, seed=42),
        23: Orthogonal(gain=1.0, seed=42),
        24: Orthogonal(gain=1.0, seed=42),
        25: Orthogonal(gain=1.0, seed=42),
        26: Orthogonal(gain=1.0, seed=42),
        27: Orthogonal(gain=1.0, seed=42),
        28: Orthogonal(gain=1.0, seed=42),
        29: Orthogonal(gain=1.0, seed=42),
        30: Orthogonal(gain=1.0, seed=42)
    }


    @staticmethod
    def build_vgg16(height, width, depth, classes):
        inputShape = (height, width, depth)

        model = Sequential()
        model.add(ZeroPadding2D((1,1),input_shape=inputShape))
        model.add(Convolution2D(64, 3, 3, activation='relu'))
        model.add(ZeroPadding2D((1,1)))
        model.add(Convolution2D(64, 3, 3, activation='relu'))
        model.add(MaxPooling2D((2,2), strides=(2,2)))

        model.add(ZeroPadding2D((1,1)))
        model.add(Convolution2D(128, 3, 3, activation='relu'))
        model.add(ZeroPadding2D((1,1)))
        model.add(Convolution2D(128, 3, 3, activation='relu'))
        model.add(MaxPooling2D((2,2), strides=(2,2)))

        model.add(ZeroPadding2D((1,1)))
        model.add(Convolution2D(256, 3, 3, activation='relu'))
        model.add(ZeroPadding2D((1,1)))
        model.add(Convolution2D(256, 3, 3, activation='relu'))
        model.add(ZeroPadding2D((1,1)))
        model.add(Convolution2D(256, 3, 3, activation='relu'))
        model.add(MaxPooling2D((2,2), strides=(2,2)))

        model.add(ZeroPadding2D((1,1)))
        model.add(Convolution2D(512, 3, 3, activation='relu'))
        model.add(ZeroPadding2D((1,1)))
        model.add(Convolution2D(512, 3, 3, activation='relu'))
        model.add(ZeroPadding2D((1,1)))
        model.add(Convolution2D(512, 3, 3, activation='relu'))
        model.add(MaxPooling2D((2,2), strides=(2,2)))

        model.add(ZeroPadding2D((1,1)))
        model.add(Convolution2D(512, 3, 3, activation='relu'))
        model.add(ZeroPadding2D((1,1)))
        model.add(Convolution2D(512, 3, 3, activation='relu'))
        model.add(ZeroPadding2D((1,1)))
        model.add(Convolution2D(512, 3, 3, activation='relu'))
        model.add(MaxPooling2D((2,2), strides=(2,2)))

        model.add(Flatten())
        model.add(Dense(4096, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(4096, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(1000, activation='softmax'))

         # softmax classifier
        model.add(Dense(classes))
        model.add(Activation("softmax"))

        # return the constructed network architecture
        return model
    
    @staticmethod
    def build_vgg16_2d_smaller(height, width, depth, classes, init_var):

        # initialize the model along with the input shape to be
        # "channels last" and the channels dimension itself
        model = Sequential()
        inputShape = (height, width, depth)
        chanDim = -1

        # CONV => RELU => POOL
        model.add(Conv2D(32, (3, 3), padding="same", input_shape=inputShape, kernel_initializer=SmallerVGGNet.switcher.get(init_var)))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=chanDim))
        model.add(MaxPooling2D(pool_size=(3, 3)))
        model.add(Dropout(0.25))


        # (CONV => RELU) * 2 => POOL
        model.add(Conv2D(64, (3, 3), padding="same", kernel_initializer=SmallerVGGNet.switcher.get(init_var)))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=chanDim))

        model.add(Conv2D(64, (3, 3), padding="same", kernel_initializer=SmallerVGGNet.switcher.get(init_var)))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=chanDim))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))

        # (CONV => RELU) * 2 => POOL
        model.add(Conv2D(128, (3, 3), padding="same", kernel_initializer=SmallerVGGNet.switcher.get(init_var)))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=chanDim))

        model.add(Conv2D(128, (3, 3), padding="same", kernel_initializer=SmallerVGGNet.switcher.get(init_var)))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=chanDim))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))

        # first (and only) set of FC => RELU layers
        model.add(Flatten())
        model.add(Dense(1024, kernel_initializer=SmallerVGGNet.switcher.get(init_var)))
        model.add(Activation("relu"))
        model.add(BatchNormalization())
        model.add(Dropout(0.5))

        # softmax classifier
        model.add(Dense(classes))
        model.add(Activation("softmax"))

        # return the constructed network architecture
        return model

    @staticmethod
    def build_vgg16_1d_smaller(height, width, classes, init_var):

        # initialize the model along with the input shape to be
        # "channels last" and the channels dimension itself
        model = Sequential()
        inputShape = (width, 1)

        chanDim = -1

        #model.add(Input(shape=(width,1)))
        #model.add(Flatten())

        # CONV => RELU => POOL
        model.add(Conv1D(32, 3, padding="same", input_shape=inputShape, kernel_initializer=SmallerVGGNet.switcher.get(init_var)))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=chanDim))
        model.add(MaxPooling1D(pool_size=3))
        model.add(Dropout(0.25))

        # (CONV => RELU) * 2 => POOL
        model.add(Conv1D(64, 3, padding="same", kernel_initializer=SmallerVGGNet.switcher.get(init_var)))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=chanDim))

        model.add(Conv1D(64, 3, padding="same", kernel_initializer=SmallerVGGNet.switcher.get(init_var)))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=chanDim))
        model.add(MaxPooling1D(pool_size=2))
        model.add(Dropout(0.25))

        # (CONV => RELU) * 2 => POOL
        model.add(Conv1D(128, 3, padding="same", kernel_initializer=SmallerVGGNet.switcher.get(init_var)))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=chanDim))

        model.add(Conv1D(128, 3, padding="same", kernel_initializer=SmallerVGGNet.switcher.get(init_var)))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=chanDim))
        model.add(MaxPooling1D(pool_size=2))
        model.add(Dropout(0.25))

        # first (and only) set of FC => RELU layers
        model.add(Flatten())
        model.add(Dense(1024, kernel_initializer=SmallerVGGNet.switcher.get(init_var)))
        model.add(Activation("relu"))
        model.add(BatchNormalization())
        model.add(Dropout(0.5))

        # softmax classifier
        model.add(Dense(classes))
        model.add(Activation("softmax"))

        # return the constructed network architecture
        return model

    @staticmethod
    def build_small_2d(height, width, depth, init_var):
        inputShape = (height, width, depth)

        model = Sequential()
        
        model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer=SmallerVGGNet.switcher.get(0), padding='same', input_shape=inputShape))
        model.add(MaxPooling2D((2, 2)))
        
        model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer=SmallerVGGNet.switcher.get(0), padding='same'))
        model.add(MaxPooling2D((2, 2)))
        model.add(Flatten())

        model.add(Dense(128, activation='relu', kernel_initializer=SmallerVGGNet.switcher.get(0)))
        #model.add(Dense(1, activation='sigmoid'))
        model.add(Dense(3, activation='softmax'))
        
        return model


    @staticmethod
    def build_small_2d_rettangolo(height, width, depth, init_var):
        inputShape = (height, width, depth)

        model = Sequential()
        
        model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer=SmallerVGGNet.switcher.get(init_var), padding='same', input_shape=inputShape))
        model.add(MaxPooling2D((2, 2)))
        
        model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer=SmallerVGGNet.switcher.get(init_var), padding='same'))
        model.add(MaxPooling2D((2, 2)))

        model.add(Conv2D(128, (3, 3), activation='relu', kernel_initializer=SmallerVGGNet.switcher.get(init_var), padding='same'))
        model.add(MaxPooling2D((2, 2)))

        model.add(Flatten())

        model.add(Dense(128, activation='relu', kernel_initializer=SmallerVGGNet.switcher.get(init_var)))
        #model.add(Dense(1, activation='sigmoid'))
        model.add(Dense(3, activation='softmax'))
        
        return model

    @staticmethod
    def build_anse_2d(height, width, depth, init_var):
        chanDim = -1

        inputShape = (height, width, depth)

        model = Sequential()
        
        model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer=SmallerVGGNet.switcher.get(init_var), padding='same', input_shape=inputShape))
        model.add(BatchNormalization(axis=chanDim))
        model.add(MaxPooling2D((2, 2)))
        model.add(Dropout(0.10))

        
        model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer=SmallerVGGNet.switcher.get(init_var), padding='same'))
        model.add(BatchNormalization(axis=chanDim))
        model.add(MaxPooling2D((2, 2)))
        model.add(Dropout(0.15))

        model.add(Conv2D(128, (3, 3), activation='relu', kernel_initializer=SmallerVGGNet.switcher.get(init_var), padding='same'))
        model.add(BatchNormalization(axis=chanDim))
        model.add(MaxPooling2D((2, 2)))
        model.add(Dropout(0.20))
        
        model.add(Flatten())

        model.add(Dense(1024, activation='relu', kernel_initializer=SmallerVGGNet.switcher.get(init_var)))
        model.add(Dense(2, activation='softmax'))
        #model.add(Dense(1, activation='sigmoid'))
        
        return model

    @staticmethod
    def build_anse_v2(height, width, depth, init_var):
        chanDim = -1

        input_shape = (height, width, depth)

        model = Sequential()
        # The first two layers with 32 filters of window size 3x3
        model.add(Conv2D(32, (3, 3), padding='same', activation='relu', input_shape=input_shape))
        model.add(Conv2D(32, (3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))

        model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
        model.add(Conv2D(64, (3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))

        model.add(Conv2D(128, (3, 3), padding='same', activation='relu'))
        model.add(Conv2D(128, (3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))

        model.add(Flatten())
        model.add(Dense(512, activation='relu'))
        model.add(Dense(2, activation='softmax'))
        
        return model
    
    @staticmethod
    def build_anse_2d_deeper(height, width, depth, init_var):
        inputShape = (height, width, depth)

        model = Sequential()
        
        model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer=SmallerVGGNet.switcher.get(init_var), padding='same', input_shape=inputShape))
        model.add(MaxPooling2D((2, 2)))
      
        model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer=SmallerVGGNet.switcher.get(init_var), padding='same'))
        model.add(MaxPooling2D((2, 2)))

        model.add(Conv2D(128, (3, 3), activation='relu', kernel_initializer=SmallerVGGNet.switcher.get(init_var), padding='same'))
        model.add(MaxPooling2D((2, 2)))

        model.add(Flatten())

        model.add(Dense(1024, activation='relu', kernel_initializer=SmallerVGGNet.switcher.get(init_var)))
        model.add(Dense(2, activation='softmax'))
        #model.add(Dense(1, activation='relu'))
        
        return model

    @staticmethod
    def build_small_1d(height, width, init_var):
        inputShape = (width, height)

        model = Sequential()
        
        model.add(Conv1D(32, 3, activation='relu', kernel_initializer=SmallerVGGNet.switcher.get(0), padding='same', input_shape=inputShape))

        if width > 1:
            model.add(MaxPooling1D(2))
        
        model.add(Conv1D(64, 3, activation='relu', kernel_initializer=SmallerVGGNet.switcher.get(0), padding='same'))
        if width > 1:
            model.add(MaxPooling1D(2))
        model.add(Flatten())

        model.add(Dense(128, activation='relu', kernel_initializer=SmallerVGGNet.switcher.get(0)))
        #model.add(Dense(1, activation='sigmoid'))
        model.add(Dense(2, activation='softmax'))
        return model

    @staticmethod
    def build_parallel_2d(height, width, depth, init_var):
        inputShape = (height, width, depth)

        input_a= Input(shape=inputShape)
        input_b= Input(shape=inputShape)
        input_c= Input(shape=inputShape)
        input_d= Input(shape=inputShape)

        # first channel
        conv1_a = Conv2D(32, (3, 3), activation='relu', kernel_initializer=SmallerVGGNet.switcher.get(0), padding='same', data_format="channels_last")(input_a)
        pooling1_a = MaxPooling2D((2, 2))(conv1_a)
        
        conv2_a = Conv2D(64, (3, 3), activation='relu', kernel_initializer=SmallerVGGNet.switcher.get(0), padding='same')(pooling1_a)
        pooling2_a = MaxPooling2D((2, 2))(conv2_a)

        upper_left_cnn = Flatten()(pooling2_a)

        # second channel
        conv1_b = Conv2D(32, (3, 3), activation='relu', kernel_initializer=SmallerVGGNet.switcher.get(0), padding='same', data_format="channels_last")(input_b)
        pooling1_b = MaxPooling2D((2, 2))(conv1_b)
        
        conv2_b = Conv2D(64, (3, 3), activation='relu', kernel_initializer=SmallerVGGNet.switcher.get(0), padding='same')(pooling1_b)
        pooling2_b = MaxPooling2D((2, 2))(conv2_b)

        upper_right_cnn = Flatten()(pooling2_b)

        # third channel
        conv1_c = Conv2D(32, (3, 3), activation='relu', kernel_initializer=SmallerVGGNet.switcher.get(0), padding='same', data_format="channels_last")(input_c)
        pooling1_c = MaxPooling2D((2, 2))(conv1_c)
        
        conv2_c = Conv2D(64, (3, 3), activation='relu', kernel_initializer=SmallerVGGNet.switcher.get(0), padding='same')(pooling1_c)
        pooling2_c = MaxPooling2D((2, 2))(conv2_c)

        lower_left_cnn = Flatten()(pooling2_c)

        # last channel
        conv1_d = Conv2D(32, (3, 3), activation='relu', kernel_initializer=SmallerVGGNet.switcher.get(0), padding='same', data_format="channels_last")(input_d)
        pooling1_d = MaxPooling2D((2, 2))(conv1_d)
        
        conv2_d = Conv2D(64, (3, 3), activation='relu', kernel_initializer=SmallerVGGNet.switcher.get(0), padding='same')(pooling1_d)
        pooling2_d = MaxPooling2D((2, 2))(conv2_d)

        lower_right_cnn = Flatten()(pooling2_d)

        # merge tutti i blocchi
        merge = concatenate([upper_left_cnn, upper_right_cnn, lower_left_cnn, lower_right_cnn])

        #merge senza 1 blocco 
        #merge = concatenate([upper_left_cnn, upper_right_cnn, lower_left_cnn])
        
        #merge senza 2 blocchi
        #merge = concatenate([upper_left_cnn, upper_right_cnn])

        # quando utilizzo 1 blocco
        #dense = Dense(128, activation='relu', kernel_initializer=SmallerVGGNet.switcher.get(0))(upper_left_cnn)

        # merge 2 o più blocchi
        dense = Dense(128, activation='relu', kernel_initializer=SmallerVGGNet.switcher.get(0))(merge)
        output = Dense(3, activation='softmax')(dense)

        # model tutti i blocchi
        model = Model(inputs=[input_a, input_b, input_c, input_d], outputs=output)

        # model senza 1 blocco 
        #model = Model(inputs=[input_a, input_b, input_c], outputs=output)
        
        # model senza 2 blocchi
        #model = Model(inputs=[input_a, input_b], outputs=output)

        # model solo 1 blocco
        #model = Model(inputs=[input_a], outputs=output)

        return model


    '''
    ' BINARY 
    '''
    @staticmethod
    def build_small_2d_binary(height, width, depth, init_var):
        inputShape = (height, width, depth)

        model = Sequential()
        
        model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer=SmallerVGGNet.switcher.get(0), padding='same', input_shape=inputShape))
        model.add(MaxPooling2D((2, 2)))
        
        model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer=SmallerVGGNet.switcher.get(0), padding='same'))
        model.add(MaxPooling2D((2, 2)))
        model.add(Flatten())

        model.add(Dense(128, activation='relu', kernel_initializer=SmallerVGGNet.switcher.get(0)))
        #model.add(Dense(1, activation='sigmoid'))
        model.add(Dense(2, activation='softmax'))
        
        return model

    @staticmethod
    def build_small_2d_v2_binary(height, width, depth, init_var):
        inputShape = (height, width, depth)

        model = Sequential()
        
        model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer=SmallerVGGNet.switcher.get(0), padding='same', input_shape=inputShape))
        model.add(MaxPooling2D((2, 2)))
        
        model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer=SmallerVGGNet.switcher.get(0), padding='same'))
        model.add(MaxPooling2D((2, 2)))

        model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer=SmallerVGGNet.switcher.get(0), padding='same'))
        model.add(MaxPooling2D((2, 2)))

        model.add(Flatten())

        model.add(Dense(128, activation='relu', kernel_initializer=SmallerVGGNet.switcher.get(0)))
        #model.add(Dense(1, activation='sigmoid'))
        model.add(Dense(2, activation='softmax'))
        
        return model


    @staticmethod
    def build_small_2d_v3_binary(height, width, depth, init_var):
        inputShape = (height, width, depth)

        model = Sequential()
        
        model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer=SmallerVGGNet.switcher.get(0), padding='same', input_shape=inputShape))
        model.add(MaxPooling2D((2, 2)))
        
        model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer=SmallerVGGNet.switcher.get(0), padding='same'))
        model.add(MaxPooling2D((2, 2)))

        model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer=SmallerVGGNet.switcher.get(0), padding='same'))
        model.add(MaxPooling2D((2, 2)))
        
        model.add(Conv2D(128, (3, 3), activation='relu', kernel_initializer=SmallerVGGNet.switcher.get(0), padding='same'))
        model.add(MaxPooling2D((2, 2)))

        model.add(Flatten())

        model.add(Dense(128, activation='relu', kernel_initializer=SmallerVGGNet.switcher.get(0)))
        #model.add(Dense(1, activation='sigmoid'))
        model.add(Dense(2, activation='softmax'))
        
        return model

    @staticmethod
    def build_small_2d_v4_binary(height, width, depth, init_var):
        inputShape = (height, width, depth)

        model = Sequential()
        
        model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer=SmallerVGGNet.switcher.get(0), padding='same', input_shape=inputShape))
        
        model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer=SmallerVGGNet.switcher.get(0), padding='same'))
        model.add(MaxPooling2D((2, 2)))

        model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer=SmallerVGGNet.switcher.get(0), padding='same'))

        model.add(Conv2D(128, (3, 3), activation='relu', kernel_initializer=SmallerVGGNet.switcher.get(0), padding='same'))
        model.add(MaxPooling2D((2, 2)))

        model.add(Flatten())

        model.add(Dense(128, activation='relu', kernel_initializer=SmallerVGGNet.switcher.get(0)))
        model.add(Dropout(0.25))
        #model.add(Dense(1, activation='sigmoid'))
        model.add(Dense(2, activation='softmax'))
        
        return model

    @staticmethod
    def build_alexNet(width, height, depth):
        input_shape = (width, height, depth)

        model = Sequential()

        # 1st Convolutional Layer
        model.add(Conv2D(filters=96, input_shape=input_shape, kernel_size=(11,11), strides=(4,4), padding='valid'))
        model.add(Activation('relu'))
        # Max Pooling
        model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='valid'))

        # 2nd Convolutional Layer
        model.add(Conv2D(filters=256, kernel_size=(5,5), strides=(1,1), padding='valid'))
        model.add(Activation('relu'))
        # Max Pooling
        model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='valid'))

        # 3rd Convolutional Layer
        model.add(Conv2D(filters=384, kernel_size=(3,3), strides=(1,1), padding='valid'))
        model.add(Activation('relu'))

        # 4th Convolutional Layer
        model.add(Conv2D(filters=384, kernel_size=(3,3), strides=(1,1), padding='valid'))
        model.add(Activation('relu'))

        # 5th Convolutional Layer
        model.add(Conv2D(filters=256, kernel_size=(3,3), strides=(1,1), padding='valid'))
        model.add(Activation('relu'))
        # Max Pooling
        model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='valid'))

        # Passing it to a Fully Connected layer
        model.add(Flatten())
        # 1st Fully Connected Layer
        model.add(Dense(4096, input_shape=input_shape)) # prima per input shape c'èera (224*224*3,)
        model.add(Activation('relu'))
        # Add Dropout to prevent overfitting
        model.add(Dropout(0.4))

        # 2nd Fully Connected Layer
        model.add(Dense(4096))
        model.add(Activation('relu'))
        # Add Dropout
        model.add(Dropout(0.4))

        # 3rd Fully Connected Layer
        model.add(Dense(1000))
        model.add(Activation('relu'))
        # Add Dropout
        model.add(Dropout(0.4))

        # Output Layer
        model.add(Dense(2))
        model.add(Activation('softmax'))

        return model

    @staticmethod
    def build_alexNet_ternary(width, height, depth):
        input_shape = (width, height, depth)

        model = Sequential()

        # 1st Convolutional Layer
        model.add(Conv2D(filters=96, input_shape=input_shape, kernel_size=(11,11), strides=(4,4), padding='valid'))
        model.add(Activation('relu'))
        # Max Pooling
        model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='valid'))

        # 2nd Convolutional Layer
        model.add(Conv2D(filters=256, kernel_size=(5,5), strides=(1,1), padding='valid'))
        model.add(Activation('relu'))
        # Max Pooling
        model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='valid'))

        # 3rd Convolutional Layer
        model.add(Conv2D(filters=384, kernel_size=(3,3), strides=(1,1), padding='valid'))
        model.add(Activation('relu'))

        # 4th Convolutional Layer
        model.add(Conv2D(filters=384, kernel_size=(3,3), strides=(1,1), padding='valid'))
        model.add(Activation('relu'))

        # 5th Convolutional Layer
        model.add(Conv2D(filters=256, kernel_size=(3,3), strides=(1,1), padding='valid'))
        model.add(Activation('relu'))
        # Max Pooling
        model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='valid'))

        # Passing it to a Fully Connected layer
        model.add(Flatten())
        # 1st Fully Connected Layer
        model.add(Dense(4096, input_shape=input_shape)) # prima per input shape c'èera (224*224*3,)
        model.add(Activation('relu'))
        # Add Dropout to prevent overfitting
        model.add(Dropout(0.4))

        # 2nd Fully Connected Layer
        model.add(Dense(4096))
        model.add(Activation('relu'))
        # Add Dropout
        model.add(Dropout(0.4))

        # 3rd Fully Connected Layer
        model.add(Dense(1000))
        model.add(Activation('relu'))
        # Add Dropout
        model.add(Dropout(0.4))

        # Output Layer
        model.add(Dense(3))
        model.add(Activation('softmax'))

        return model

    @staticmethod
    def build_alexNet_1D(width, height):
        input_shape = (width, height)

        model = Sequential()

        # 1st Convolutional Layer
        model.add(Conv1D(filters=96, input_shape=input_shape, kernel_size=11, strides=4, padding='valid'))
        model.add(Activation('relu'))
        # Max Pooling
        model.add(MaxPooling1D(pool_size=2, strides=2, padding='valid'))

        # 2nd Convolutional Layer
        model.add(Conv1D(filters=256, kernel_size=11, strides=11, padding='valid'))
        model.add(Activation('relu'))
        # Max Pooling
        model.add(MaxPooling1D(pool_size=2, strides=2, padding='valid'))

        # 3rd Convolutional Layer
        model.add(Conv1D(filters=384, kernel_size=3, strides=1, padding='valid'))
        model.add(Activation('relu'))

        # 4th Convolutional Layer
        model.add(Conv1D(filters=384, kernel_size=3, strides=1, padding='valid'))
        model.add(Activation('relu'))

        # 5th Convolutional Layer
        model.add(Conv1D(filters=256, kernel_size=3, strides=1, padding='valid'))
        model.add(Activation('relu'))
        # Max Pooling
        model.add(MaxPooling1D(pool_size=2, strides=2, padding='valid'))

        # Passing it to a Fully Connected layer
        model.add(Flatten())
        # 1st Fully Connected Layer
        model.add(Dense(4096, input_shape=input_shape)) # prima per input shape c'èera (224*224*3,)
        model.add(Activation('relu'))
        # Add Dropout to prevent overfitting
        model.add(Dropout(0.4))

        # 2nd Fully Connected Layer
        model.add(Dense(4096))
        model.add(Activation('relu'))
        # Add Dropout
        model.add(Dropout(0.4))

        # 3rd Fully Connected Layer
        model.add(Dense(1000))
        model.add(Activation('relu'))
        # Add Dropout
        model.add(Dropout(0.4))

        # Output Layer
        model.add(Dense(2))
        model.add(Activation('softmax'))

        return model