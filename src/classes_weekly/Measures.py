import pandas as pd 
import numpy as np 
from datetime import timedelta
from classes.Market import Market

class Measures:

    long_value = 2
    hold_value = 1
    short_value = 0


    '''
    ' Restituisce il bilanciamento di un dataset
    ' controllando la percentuale di elementi sopra lo zero, uguali a zero
    ' o sotto lo zero
    '
    ' RETURN coverage (obj)
    ' coverage = {
            'short': ,
            'hold': ,
            'long': 
        }
    '
    '''
    @staticmethod
    def get_delta_coverage(delta):
        len_delta = len(delta)

        coverage = {
            'short': sum(1 for i in delta if i < 0) / len_delta,
            'hold': sum(1 for i in delta if i == 0) / len_delta,
            'long':  sum(1 for i in delta if i > 0) / len_delta,
            'short_count': sum(1 for i in delta if i < 0),
            'hold_count': sum(1 for i in delta if i == 0),
            'long_count': sum(1 for i in delta if i > 0)
        }

        return coverage
    

    '''
    ' y_pred dev'essere il vettore delle predizioni (quindi riferite al giorno successivo)
    ' delta dev'essere il vettore dei delta già shiftati di un giorno indietro (-1), in modo da combaciare
    '   con la label
    ' multiplier è il moltiplicatore del return
    ' type può essere long_short, long_only o short_only
    '
    ' RETURN: pred
    '''
    @staticmethod
    def pred_modifier(y_pred, type='long_short'):

        pred = y_pred

        if type is 'long_short':
            pred = [-1 if x==Measures.short_value else x for x in pred]
            pred = [0 if x==Measures.hold_value else x for x in pred] 
            pred = [1 if x==Measures.long_value else x for x in pred]

        if type is 'long_only':
            pred = [0 if x==Measures.short_value else x for x in pred]
            pred = [0 if x==Measures.hold_value else x for x in pred] 
            pred = [1 if x==Measures.long_value else x for x in pred]

        if type is 'short_only':
            pred = [-1 if x==Measures.short_value else x for x in pred]
            pred = [0 if x==Measures.hold_value else x for x in pred] 
            pred = [0 if x==Measures.long_value else x for x in pred]

        return pred

    '''
    ' y_pred dev'essere il vettore delle predizioni (quindi riferite al giorno successivo)
    ' delta dev'essere il vettore dei delta già shiftati di un giorno indietro (-1), in modo da combaciare
    '   con la label
    ' multiplier è il moltiplicatore del return
    ' type può essere long_short, long_only o short_only
    ' Calcola il return, il romad e il max dd dato un vettore di predizioni ed un delta associato
    ' Restituisce anche i punti i, j dove si verifica il mdd
    '
    ' RETURN: equity_line, global_return, mdd, romad, i, j
    '''
    @staticmethod
    def get_equity_return_mdd_romad(df, multiplier, type='long_short', penalty=32, stop_loss=1000, delta_to_use='delta_current_day'):

        if stop_loss > 0: 
            df = Market.get_delta_penalty_stop_loss(df=df.copy(), stop_loss=stop_loss, penalty=penalty, multiplier=multiplier, delta_to_use=delta_to_use)
            y_pred = df['decision'].tolist()
            delta = df['delta_penalty_stop_loss'].tolist()
        else: 
            df = Market.get_delta_penalty(df=df.copy(), penalty=penalty, multiplier=multiplier, delta_to_use=delta_to_use)
            y_pred = df['decision'].tolist()
            delta = df['delta_penalty'].tolist()

        # modifico il vettore delle predizioni se mi serve long  + short, long only o short only
        y_pred = Measures.pred_modifier(y_pred, type) 

        equity_line = np.add.accumulate( np.multiply(y_pred, delta) * multiplier)

        
        #np.savetxt('equity.txt', equity_line, fmt='%f', delimiter='\n') 
        #print("equity ls")
        #input()
        if len(equity_line) > 0:
            global_return = equity_line[-1] 
            #print("global return", global_return)
            #input()
            cumulative = np.maximum.accumulate(equity_line) - equity_line

            i, j = 0, 0
            mdd = np.nan
            if not all(cumulative == 0):
                i = np.argmax(cumulative)
                j = np.argmax(equity_line[:i])

                mdd =  equity_line[j] - equity_line[i]
                romad = global_return / mdd
            else:
                mdd =  equity_line[j] - equity_line[i]
                romad = 0
        else:
            global_return = 0
            mdd = 0
            romad = 0
            i = 0
            j = 0
        

        return equity_line, global_return, mdd, romad, i, j

    '''
    ' Calcolo return mdd, romad di una serie (close)
    ' per il buy & hold
    '
    ' RETURN: equity_line, global_return, mdd, romad, i, j 
    '''
    @staticmethod
    def get_return_mdd_romad_bh(close, multiplier):

        close = [i * multiplier for i in close] 
        global_return = close[-1] - close[0]

        cumulative = np.maximum.accumulate(close) - close
        
        i, j = 0, 0
        mdd = np.nan
        if not all(cumulative == 0):
            i = np.argmax(cumulative)
            j = np.argmax(close[:i])

            mdd =  close[j] - close[i]
            romad = global_return / mdd
        else:
            romad = 0

        return close, global_return, mdd, romad, i, j
        
    '''
    ' Dato un vettore di predizioni ed un vettore di delta
    ' calcola quante operazioni vengono fatte, quante sono giuste
    ' qual è la precision e la coverage
    '
    ' RETURN: long, short, hold, general (obj)
    ' [long, short] = {
            "precision": , 
            "count": ,
            "guessed": ,
            "coverage": 
            }
    ' hold = {
            "count": ,
            "coverage": 
            } 
    '
    '''
    @staticmethod
    def get_precision_count_coverage(df, multiplier, delta_to_use, stop_loss=1000, penalty=32): 
        if stop_loss > 0: 
            df = Market.get_delta_penalty_stop_loss(df=df.copy(), stop_loss=stop_loss, penalty=penalty, multiplier=multiplier, delta_to_use=delta_to_use)
            y_pred = df['decision'].tolist()
            delta = df['delta_penalty_stop_loss'].tolist()
        else: 
            df = Market.get_delta_penalty(df=df.copy(), penalty=penalty, multiplier=multiplier, delta_to_use=delta_to_use)
            y_pred = df['decision'].tolist()
            delta = df['delta_penalty'].tolist()

        long_count, long_guessed = 0, 0

        short_count, short_guessed = 0, 0

        hold_count = 0

        for i, val in enumerate(y_pred):
            # Long
            if val == Measures.long_value:
                long_count += 1
                if delta[i] >= 0:
                    long_guessed += 1
            # Short 
            elif val == Measures.short_value:
                short_count += 1
                if delta[i] < 0:
                    short_guessed += 1
            
            #elif val == 1.:
            #    hold_count += 1
            #    if delta[i] > -0.2 and delta[i] < 0.2:
            #        hold_guessed += 1
                    
        hold_count = len(y_pred) - long_count - short_count

        # percentuale di long e shorts azzeccate
        long_precision = 0 if long_count == 0 else long_guessed / long_count
        short_precision = 0 if short_count == 0 else short_guessed / short_count

        random_val = Measures.get_delta_coverage(delta=df[delta_to_use])

        if len(y_pred) > 0:
            # percentuale di operazioni di long e shorts sul totale di operazioni fatte
            long_coverage = long_count / (len(y_pred))
            short_coverage = short_count / (len(y_pred))
            hold_coverage = hold_count / (len(y_pred))
        else: 
            long_coverage = 0
            short_coverage = 0
            hold_coverage = 0

        long = {
            "precision": long_precision, 
            "count": long_count,
            "guessed": long_guessed,
            "coverage": long_coverage,
            "random_perc": random_val['long'],
            "random_count": random_val['long_count']
            }
        
        short = {
            "precision": short_precision, 
            "count": short_count,
            "guessed": short_guessed,
            "coverage": short_coverage,
            "random_perc": random_val['short'],
            "random_count": random_val['short_count']
            }
        hold = {
            "count": hold_count,
            "coverage": hold_coverage,
            "random_perc": random_val['hold'],
            "random_count": random_val['hold_count']
            }

        general = {
            "total_trade": long['count'] + short['count'], 
            "total_guessed_trade": long['guessed'] + short['guessed'],
            "total_operation": long['count'] + short['count'] + hold['count'] 

        }

        return long, short, hold, general

    '''
    '
    '''
    @staticmethod
    def get_precision_over_coverage(df, multiplier, stop_loss, penalty, delta_to_use='delta_next_day'):
        delta = df[delta_to_use].tolist()

        coverage = Measures.get_delta_coverage(delta=delta)
        long, short, hold, general = Measures.get_precision_count_coverage(df=df, multiplier=multiplier, stop_loss=stop_loss, penalty=penalty, delta_to_use=delta_to_use)

        
        long_poc = np.divide(long['precision'], coverage['long'])
        short_poc = np.divide(short['precision'], coverage['short'])

        if long_poc != 0.0:
            long_poc = round((long_poc - 1 ) * 100, 3)
        
        if short_poc != 0.0:
            short_poc = round((short_poc - 1 ) * 100, 3)
        
        # Evito di sfanculare i plot
        if long_poc > 30.0: 
            long_poc = 30.0
        
        if short_poc > 30.0: 
            short_poc = 30.0

        if long_poc < -30.0: 
            long_poc = -30.0
        
        if short_poc < -30.0: 
            short_poc = -30.0


        '''
        print("long precision:", long['precision'])
        print("long coverage:", coverage['long'])
        print("long poc:", long_poc)
        
        print("short precision:", short['precision'])
        print("short coverage:", coverage['short'])
        print("short poc:", short_poc)
        input()
        '''

        return long_poc, short_poc
