from keras.utils import to_categorical
from sklearn.preprocessing import LabelBinarizer, MultiLabelBinarizer

class Set:

    date_time = ''
    x = []
    y_current_day = []
    y_next_day = []
    delta_current_day = ''
    delta_next_day = ''

    close_value = ''

    def __init__(self, date_time=None, x=None, y_current_day=None, y_next_day=None, delta_current_day=None, delta_next_day=None, close=None):
        self.date_time = date_time
        self.x = x
        self.y_current_day = [int(i) for i in y_current_day]
        self.y_next_day = [int(i) for i in y_next_day]

        self.delta_current_day = delta_current_day

        self.delta_next_day = delta_next_day
        
        self.close = close

    '''
    ' Restituisce la matrice delle img
    ' normalized=True serve per normalizzare i valori tra [0, 1]
    '''
    def get_x(self, normalized=True):
        if normalized is True:
            return self.x.astype('float32') / 255
        else:
            return self.x
    
    '''
    '
    '''
    def get_y(self, type='classic', referred_to='next_day'):
        if referred_to is 'next_day':
            y = self.y_next_day

        if referred_to is 'current_day':
            y = self.y_current_day

        if type is 'classic':
            return y
        
        if type is 'categorical':
            y = to_categorical([int(i) for i in y])
            return y
        
        if type is 'binary':
            lb = LabelBinarizer()
            y = lb.fit_transform(y)
            return y

        if type is 'multi_label_binary':
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