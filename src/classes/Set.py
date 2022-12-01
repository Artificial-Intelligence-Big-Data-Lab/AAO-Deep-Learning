import numpy as np
from keras.utils import to_categorical
from sklearn.preprocessing import LabelBinarizer, MultiLabelBinarizer
from collections import Counter
import math 
from tqdm import tqdm

class Set:

    date_time = ''
    x = []
    y_current_day = []
    y_next_day = []
    delta_current_day = ''
    delta_next_day = ''

    close_value = ''

    def __init__(self, date_time=None, x=None, y_current_day=None, y_next_day=None, delta_current_day=None, delta_next_day=None, close=None, balance_binary=False):
        self.date_time = date_time
        self.x = x
        self.y_current_day = [int(i) for i in y_current_day]
        self.y_next_day = [int(i) for i in y_next_day]

        self.delta_current_day = delta_current_day

        self.delta_next_day = delta_next_day
        
        self.close = close

        counter = Counter(self.y_next_day)
        print("Label Balancing before balance:", counter)
        
        if balance_binary == True:
            self.balance_binary_dataset()

            counter = Counter(self.y_next_day)
            print("Label Balancing after balance:", counter)

    
    '''
    ' Restituisce la matrice delle img
    ' normalized=True serve per normalizzare i valori tra [0, 1]
    '''
    def get_x(self, normalized=True, multivariate=False):

        if multivariate == True: 
            return self.get_multivariate_x(normalized=normalized)
        else:
            if normalized is True:
                return self.x.astype('float32') / 255
            else:
                return self.x
    
    '''
    '
    '''
    def get_y(self, type='classic', referred_to='next_day'):
        if referred_to == 'next_day':
            y = self.y_next_day

        if referred_to == 'current_day':
            y = self.y_current_day

        if type == 'classic':
            return y
        
        if type == 'categorical':
            y = to_categorical([int(i) for i in y])
            return y
        
        if type == 'binary':
            lb = LabelBinarizer()
            y = lb.fit_transform(y)
            return y

        if type == 'multi_label_binary':
            mlb = MultiLabelBinarizer()
            y = mlb.fit_transform(y)
            return y
    
    '''
    '
    '''
    def get_date_time(self):
        return self.date_time

    '''
    '
    '''
    def get_delta_current_day(self):
        return self.delta_current_day

    '''
    '
    '''
    def get_delta_next_day(self):
        return self.delta_next_day

    '''
    '
    '''
    def get_close(self):
        return self.close  

    '''
    '
    '''
    def get_multivariate_x(self, normalized=True):

        imgs = self.x
        x1 = []
        x2 = []
        x3 = []
        x4 = []

        for i in range(0, len(self.x)):

            if normalized is True:
                img1 = imgs[i][0:20, 0:20].astype('float32') / 255
                img2 = imgs[i][20:40, 0:20].astype('float32') / 255
                img3 = imgs[i][0:20, 20:40].astype('float32') / 255
                img4 = imgs[i][20:40, 20:40].astype('float32') / 255
            else:
                img1 = imgs[i][0:20, 0:20].astype('float32') 
                img2 = imgs[i][20:40, 0:20].astype('float32')
                img3 = imgs[i][0:20, 20:40].astype('float32')
                img4 = imgs[i][20:40, 20:40].astype('float32')

            x1.append(img1) # 1h 
            x2.append(img2) # 4h
            x3.append(img3) # 8h
            x4.append(img4) # 1d
        
        # tutti i blocchi
        return [np.asarray(x1), np.asarray(x2), np.asarray(x3), np.asarray(x4)]
        
        # senza blocco 1d
        #return [np.asarray(x1), np.asarray(x2), np.asarray(x3)]

        # senza blocco 8h
        #return [np.asarray(x1), np.asarray(x2), np.asarray(x4)]

        # senza blocco 4h
        #return [np.asarray(x1), np.asarray(x3), np.asarray(x4)]

        # senza blocco 1h
        #return [np.asarray(x2), np.asarray(x3), np.asarray(x4)]

        # senza blocco 1d 8h
        #return [np.asarray(x1), np.asarray(x2)]

        # senza blocco 1h 4h
        #return [np.asarray(x3), np.asarray(x4)]


        # solo blocco 1h
        #return [np.asarray(x1)]

        #### EXP WEEKEND
        # solo blocco 4h
        #return [np.asarray(x2)]

        # solo blocco 8h
        #return [np.asarray(x3)]

        # solo blocco 1d
        #return [np.asarray(x4)]

    '''
    ' Bilancio la classe minoritaria
    ' in caso di label binaria
    '''
    def balance_binary_dataset(self):

        counter = Counter(self.y_next_day)

        class_max = -1
        class_min = -1

        if counter[0] > counter[1]:
            class_max = 0
            class_min = 1
        else:
            class_max = 1
            class_min = 0

        difference = math.floor(counter[class_max] / counter[class_min])

        list_of_max = [i for i, e in enumerate(self.y_next_day) if e == class_max]
        list_of_min = [i for i, e in enumerate(self.y_next_day) if e == class_min]

        n_items = self.x.shape[0] + (len(list_of_max) - len(list_of_min))
        new_x = np.zeros(shape=(n_items, self.x.shape[1], self.x.shape[2], self.x.shape[3]))

        for i, e in enumerate(self.x):
            new_x[i] = self.x[i]

        external_index = self.x.shape[0]

        print("Balancing dataset...")
        for i in tqdm(range(0, difference)): 
            for j in list_of_min:
                if external_index == new_x.shape[0]:
                    continue

                #self.x = np.append(self.x, [self.x[j]], axis=0)
                new_x[external_index] = self.x[j]

                self.y_next_day.append(class_min)
                self.y_current_day.append(class_min)
                self.date_time.append(self.date_time[j])

                external_index += 1

                counter = Counter(self.y_next_day)
                if(counter[0] == counter[1]):
                    self.x = new_x
                    break