import pandas as pd 
import numpy as np 
from datetime import timedelta
from classes.Market import Market
from sklearn.metrics import accuracy_score, balanced_accuracy_score
import time

class Measures:

    '''
    ' classic intraday
    ''
    long_value = 2
    hold_value = 1
    short_value = 0
    '''

    '''
    ' anselmo
    long_value = 1
    hold_value = 0
    short_value = 2
    '''

    ''' Binary Classification'''
    long_value = 1
    hold_value = -2
    short_value = 0    

    '''
    ' Sovrascrivo i valori delle operazioni
    ''
    def __init__(self, long_value=2, short_value=0, hold_value=1):
        self.long_value = long_value
        self.short_value = short_value
        self.hold_value = hold_value
    '''
    
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
    '
    '''
    @staticmethod
    def get_binary_coverage(y):
        down = y.count(0)
        up = y.count(1)
        tot = down + up 

        coverage = {
            'down_perc': round((down / tot) * 100, 2), 
            'up_perc': round((up / tot) * 100, 2), 
            'down_count': down,
            'up_count': up, 
            'tot_cout': tot
        }

        return coverage
    
    '''
    '
    '''
    @staticmethod
    def get_binary_delta_precision(y, delta, delta_val=-25, multiplier=50):
        tot_len = len(delta)

        delta_val = delta_val / multiplier

        up_count, up_guessed = 0, 0
        down_count, down_guessed = 0, 0
    
        for i, val in enumerate(y):
            # Up
            if val == Measures.long_value:
                up_count += 1
                if delta[i] >= delta_val:
                    up_guessed += 1

            if val == 0:
                down_count += 1
                if delta[i] < delta_val:
                    down_guessed += 1

        up_precision = 0 if up_count == 0 else (up_guessed / up_count) * 100
        down_precision = 0 if down_count == 0 else (down_guessed / down_count) * 100
        up_random = (sum(i > delta_val for i in delta) / tot_len) * 100
        down_random = (sum(i < delta_val for i in delta) / tot_len) * 100

        precisions = {
            'up': up_precision,
            'down': down_precision,
            'random_up':  up_random,
            'random_down': down_random,
        }

        return precisions

    '''
    ' y_pred dev'essere il vettore delle predizioni (quindi riferite al giorno successivo)
    ' delta dev'essere il vettore dei delta già shiftati di un giorno indietro (-1), in modo da combaciare
    '   con la label
    ' multiplier è il moltiplicatore del return
    ' type può essere long_short, long_only o short_only
    '
    ' type:  
    ' long_short: converte tutte le operazioni
    ' long_only: converte solo le long, il resto diventa hold
    ' short_only: converte solo le short, il resto diventa hold
    ' bh_long: converte tutte le operazioni in long, simulando un bh intraday
    ' RETURN: pred
    '''
    @staticmethod
    def pred_modifier(y_pred, type='long_short'):

        pred = y_pred
        
        #if type == 'bh_long':
        #pred = [1 for x in pred]

        # BINARY
        if type == 'long_only':
            pred = [0 if x==Measures.short_value else x for x in pred]
            pred = [0 if x==Measures.hold_value else x for x in pred] 
            pred = [1 if x==Measures.long_value else x for x in pred]

        if type == 'long_short':
            pred = [-1 if x==Measures.short_value else x for x in pred]
            pred = [0 if x==Measures.hold_value else x for x in pred] 
            pred = [1 if x==Measures.long_value else x for x in pred]

        if type == 'short_only':
            pred = [-1 if x==Measures.short_value else x for x in pred]
            pred = [0 if x==Measures.hold_value else x for x in pred] 
            pred = [0 if x==Measures.long_value else x for x in pred]
        
        if type == 'bh_long':
            pred = [1 for x in pred]


        ''' ternary
        if type == 'long_short':
            pred = [-1 if x==Measures.short_value else x for x in pred]
            pred = [0 if x==Measures.hold_value else x for x in pred] 
            pred = [1 if x==Measures.long_value else x for x in pred]

        if type == 'long_only':
            pred = [0 if x==Measures.short_value else x for x in pred]
            pred = [0 if x==Measures.hold_value else x for x in pred] 
            pred = [1 if x==Measures.long_value else x for x in pred]

        if type == 'short_only':
            pred = [-1 if x==Measures.short_value else x for x in pred]
            pred = [0 if x==Measures.hold_value else x for x in pred] 
            pred = [0 if x==Measures.long_value else x for x in pred]
        
        if type == 'bh_long':
            pred = [1 for x in pred]
        '''
        return pred

    '''
    ' df['decision'] dev'essere il vettore delle predizioni (quindi riferite al giorno successivo)
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
    def get_equity_return_mdd_romad(df, multiplier, type='long_short', penalty=32, stop_loss=1000, delta_to_use='delta_current_day', compact_results=False):

        if type is 'bh_long':
            df['decision'] = Measures.long_value
            
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
        
        global_return, mdd, romad, i, j = Measures.get_return_mdd_romad_from_equity(equity_line=equity_line)  

        if compact_results == False:
            return equity_line, global_return, mdd, romad, i, j
        else: 
            return {
                'equity_line': equity_line, 
                'return': global_return, 
                'mdd': mdd, 
                'romad': romad, 
                'i': i, 
                'j': j
            }

    '''
    ' restituisce il vettore con i return giornalieri 
    ' calcolati su un vettore di decisioni dentro df['decision'] che può 
    ' riferirsi al giorno successivo o al giorno corrente
    '''
    @staticmethod
    def get_daily_returns(df, multiplier, type='long_short', penalty=32, stop_loss=1000, delta_to_use='delta_current_day'):

        if type is 'bh_long':
            df['decision'] = Measures.long_value
            
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

        daily_returns = np.multiply(y_pred, delta) * multiplier

        return daily_returns

    '''
    ' restituisce sharpe ratio 
    ' calcolato su un vettore di decisioni dentro df['decision'] che può 
    ' riferirsi al giorno successivo o al giorno corrente
    '''
    @staticmethod
    def get_sharpe_ratio(df, multiplier, type='long_short', penalty=32, stop_loss=1000, delta_to_use='delta_current_day', risk_free=0):

        if type is 'bh_long':
            df['decision'] = Measures.long_value
            
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

        daily_returns = np.multiply(y_pred, delta) * multiplier
        
        
        all_zeros = not np.any(daily_returns)
        
        if all_zeros == True or daily_returns.shape[0] < 2:
            return 0, 0 
        else: 

            sharpe_ratio = (daily_returns.mean() - risk_free) / daily_returns.std()
            annualized_sharpe_ratio = ((daily_returns.mean() - risk_free) / daily_returns.std()) * np.sqrt(252) # 240 / 252

        #print(type, sharpe_ratio, annualized_sharpe_ratio)
        #input()
        return sharpe_ratio, annualized_sharpe_ratio

    '''
    '
    '''
    @staticmethod
    def get_sortino_ratio(df, multiplier, type='long_short', penalty=32, stop_loss=1000, delta_to_use='delta_current_day', risk_free=0):

        if type is 'bh_long':
            df['decision'] = Measures.long_value
            
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

        daily_returns = np.multiply(y_pred, delta) * multiplier

        neg_daily_returns = []
        neg_daily_returns = [i for i in daily_returns if i  < 0]
        # per fare .std serve che sia un array np
        neg_daily_returns = np.array(neg_daily_returns)

        all_zeros = not np.any(neg_daily_returns)
        
        if all_zeros == True or neg_daily_returns.shape[0] < 2:
            return 0, 0 
        else:
            sortino_ratio = (daily_returns.mean() - risk_free) / neg_daily_returns.std()
            annualized_sortino_ratio = ((daily_returns.mean() - risk_free) / neg_daily_returns.std()) * np.sqrt(252) # 240 / 252

        return sortino_ratio, annualized_sortino_ratio
    
    
    '''
    ' Calcola return, mdd, romad 
    ' dato un vettore di equity
    '''
    @staticmethod
    def get_return_mdd_romad_from_equity(equity_line):
        if len(equity_line) > 0:
            global_return = equity_line[-1] 

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
        

        return global_return, mdd, romad, i, j

    '''
    ' Calcolo return mdd, romad di una serie (close)
    ' per il buy & hold
    '
    ' RETURN: equity_line, global_return, mdd, romad, i, j 
    '''
    @staticmethod
    def get_return_mdd_romad_bh(close, multiplier, compact_results=False):

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


        if compact_results == False:
            return close, global_return, mdd, romad, i, j
        else: 
            return {
                'equity_line': close, 
                'return': global_return, 
                'mdd': mdd, 
                'romad': romad, 
                'i': i, 
                'j': j
            }
        
        
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

        y_pred = df['decision'].tolist()
        delta = df['delta_next_day'].tolist()

        if stop_loss > 0 and penalty > 0:
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
        hold_count, hold_guessed = 0, 0
        
        y_true = []

        if delta_to_use == 'delta_next_day':
            y_true = df['label_next_day'].tolist()
        else: 
            y_true = df['label_current_day'].tolist()

        #accuracy = accuracy_score(y_true, y_pred, normalize=True)
        accuracy = balanced_accuracy_score(y_true, y_pred)

        #print("Delta qui dentro [1]", delta)
        #input()
        for i, val in enumerate(y_pred):
            # Long
            if val == Measures.long_value:
                #print("Dentro if -  val:", val, " - delta[" + str(i) + "]", delta[i])
                long_count += 1
                if delta[i] >= 0:
                    long_guessed += 1
            # Short 
            elif val == Measures.short_value:
                short_count += 1
                if delta[i] < 0:
                    short_guessed += 1
            elif val == Measures.hold_value:
                #print("Dentro if -  val:", val, " - delta[" + str(i) + "]", delta[i])
                hold_count += 1
                if delta[i] < 0:
                    hold_guessed += 1
        #print("Delta qui dentro [2]", delta)
        #input()
        #hold_count = len(y_pred) - long_count - short_count

        # percentuale di long e shorts azzeccate
        long_precision = 0 if long_count == 0 else long_guessed / long_count
        short_precision = 0 if short_count == 0 else short_guessed / short_count

        #print("Hold guessed", hold_guessed)
        #print("Hold count", hold_count)
        #print("Delta", delta)
        hold_precision = 0 if hold_count == 0 else hold_guessed / hold_count

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
            "precision": hold_precision,
            "count": hold_count,
            "coverage": hold_coverage,
            "random_perc": random_val['hold'],
            "random_count": random_val['hold_count']
            }

        general = {
            "total_coverage": long['coverage'] + short['coverage'],
            "total_trade": long['count'] + short['count'], 
            "total_guessed_trade": long['guessed'] + short['guessed'],
            "total_operation": long['count'] + short['count'] + hold['count'], 
            "accuracy": accuracy
        }

        return long, short, hold, general

    '''
    '
    '''
    @staticmethod
    def get_precision_label(df, label_to_use):
        long_count, long_guessed = 0, 0
        short_count, short_guessed = 0, 0
        hold_count, hold_guessed = 0, 0

        y_pred = df['decision'].tolist()

        if label_to_use == 'label_next_day':
            y_true = df['label_next_day'].tolist()
        else: 
            y_true = df['label_current_day'].tolist()

        for i, val in enumerate(y_pred):
            # Long
            if val == Measures.long_value:
                long_count += 1
                if y_true[i] == val:
                    long_guessed += 1
            # Short 
            elif val == Measures.short_value:
                short_count += 1
                if y_true[i] == val:
                    short_guessed += 1
            elif val == Measures.hold_value:
                hold_count += 1
                if y_true[i] == val:
                    hold_guessed += 1
                    
        # percentuale di long e shorts azzeccate
        long_precision = 0 if long_count == 0 else long_guessed / long_count
        short_precision = 0 if short_count == 0 else short_guessed / short_count
        hold_precision = 0 if hold_count == 0 else hold_guessed / hold_count


        long = {
            "precision": long_precision, 
            "count": long_count,
            "guessed": long_guessed,
            "coverage": long_count / len(y_pred),
            "random_perc": y_true.count(Measures.long_value) / len(y_true),
            "random_count": y_true.count(Measures.long_value)
        }
        
        short = {
            "precision": short_precision, 
            "count": short_count,
            "guessed": short_guessed,
            "coverage": short_count / len(y_pred),
            "random_perc": y_true.count(Measures.short_value) / len(y_true),
            "random_count": y_true.count(Measures.short_value)
        }
        
        hold = {
            "precision": hold_precision,
            "count": hold_count,
            "guessed": hold_guessed,
            "coverage": hold_count / len(y_pred),
            "random_perc": y_true.count(Measures.hold_value) / len(y_true),
            "random_count": y_true.count(Measures.hold_value)
        }

        return long, short, hold


    '''
    ' Calcolo il cosidetto valore PoR
    ' Precision over Random (il metodo, per motivi di compatibilità, si chiama 
    ' ancora precision over coverage)
    ' Misura di quanto aumenta la precision di long o short
    ' rispetto alla % di coverage, che rappresenta anche 
    ' la linea del random per quella classe
    '''
    @staticmethod
    def get_precision_over_coverage(df, multiplier, stop_loss, penalty, delta_to_use='delta_next_day'):
        delta = df[delta_to_use].tolist()

        coverage = Measures.get_delta_coverage(delta=delta)
        long, short, hold, general = Measures.get_precision_count_coverage(df=df, multiplier=multiplier, stop_loss=stop_loss, penalty=penalty, delta_to_use=delta_to_use)

        if long['precision'] > 0:
            long_poc = np.divide(long['precision'], coverage['long'])
        else: 
            long_poc = -0.0001

        if short['precision'] > 0:
            short_poc = np.divide(short['precision'], coverage['short'])
        else: 
            short_poc = -0.0001

        if hold['precision'] > 0:
            hold_poc = np.divide(hold['precision'], coverage['short'])
        else: 
            hold_poc = -0.0001

        if long_poc != 0.0:
            long_poc = round((long_poc - 1 ) * 100, 3)
        
        if short_poc != 0.0:
            short_poc = round((short_poc - 1 ) * 100, 3)

        if hold_poc != 0.0:
            hold_poc = round((hold_poc - 1 ) * 100, 3)

        
        '''
        # Evito di sfanculare i plot
        if long_poc > 30.0: 
            long_poc = 30.0
        
        if short_poc > 30.0: 
            short_poc = 30.0

        if long_poc < -30.0: 
            long_poc = -30.0
        
        if short_poc < -30.0: 
            short_poc = -30.0


        
        print("long precision:", long['precision'])
        print("long coverage:", coverage['long'])
        print("long poc:", long_poc)
        
        print("short precision:", short['precision'])
        print("short coverage:", coverage['short'])
        print("short poc:", short_poc)
        input()
        '''

        return long_poc, short_poc, hold_poc


    '''
    ' Calcolo il cosidetto valore PoL
    ' Precision over Label 
    ' Misura di quanto aumenta la precision di long, short o idle
    ' rispetto alla label, che rappresenta anche 
    ' la linea del random per quella classe
    '''
    @staticmethod
    def get_precision_over_label(df, label_to_use='label_next_day'):
        long, short, hold = Measures.get_precision_label(df=df, label_to_use='label_next_day')
        long_poc = 0.0
        short_poc = 0.0
        hold_poc = 0.0


        # LONG
        if long['random_perc'] > 0:
            if long['precision'] > 0:
                long_poc = np.divide(long['precision'], long['random_perc'])
            else: 
                long_poc = -0.0001
            
        
        # SHORT
        if short['random_perc'] > 0:
            if short['precision'] > 0:
                short_poc = np.divide(short['precision'], short['random_perc'])
            else: 
                short_poc = -0.0001

        # HOLD
        if hold['random_perc'] > 0:
            if hold['precision'] > 0:
                hold_poc = np.divide(hold['precision'], hold['random_perc']) if hold['random_perc'] > 0  else 0
            else:
                hold_poc = -0.0001


        if long_poc != 0.0:
            long_poc = round((long_poc - 1 ) * 100, 3)
        
        if short_poc != 0.0:
            short_poc = round((short_poc - 1 ) * 100, 3)

        if hold_poc != 0.0:
            hold_poc = round((hold_poc - 1 ) * 100, 3)

        return long_poc, short_poc, hold_poc



    '''
    ' Dato un df di decisioni finali (post ensemble)
    ' calcolo la coverage di long, short e hold su base mensile 
    ' In questo modo si può valutare con un equity la coverage delle operazioni
    ' nel tempo
    '''
    @staticmethod
    def get_coverage_per_month(df): 

        date_df = df.copy()
        date_df = date_df[['date_time']]
        date_df['start_date'] = date_df['date_time'].tolist()
        date_df['date_time'] = pd.to_datetime(date_df['date_time'])
        date_df = date_df.groupby(pd.Grouper(key='date_time', freq='M'), sort=True).agg({'start_date': 'first'}).reset_index()
        date_df = date_df.rename(columns={"date_time": "end_date"})
        date_df['end_date'] = date_df['end_date'].astype(str)
        
        coverage_long_list = []
        coverage_short_list = []
        coverage_long_short_list = []
        coverage_hold_list = []
        

        for i in range(0, date_df.shape[0]):
            min_df = Market.get_df_by_data_range(df=df.copy(), start_date=date_df.iloc[i].start_date, end_date=date_df.iloc[i].end_date)

            y_pred = min_df['decision'].tolist()

            if len(y_pred) > 0:
                long_count = y_pred.count(2)
                short_count = y_pred.count(0)
                hold_count = y_pred.count(1)
                # percentuale di operazioni di long e shorts sul totale di operazioni fatte
                coverage_long_list.append( long_count / (len(y_pred)) )
                coverage_short_list.append( short_count / (len(y_pred)) )
                coverage_long_short_list.append( (long_count + short_count) / (len(y_pred)) )
                coverage_hold_list.append( hold_count / (len(y_pred)) )
            else: 
                coverage_long_list.append( 0 )
                coverage_short_list.append( 0 )
                coverage_long_short_list.append( 0 )
                coverage_hold_list.append( 0 )

        coverage_long_list = [i * 100 for i in coverage_long_list]
        coverage_short_list = [i * 100 for i in coverage_short_list]
        coverage_long_short_list = [i * 100 for i in coverage_long_short_list]
        coverage_hold_list = [i * 100 for i in coverage_hold_list]

        coverage = {
            "coverage_long": coverage_long_list, 
            "coverage_short": coverage_short_list,
            "coverage_long_short": coverage_long_short_list, 
            "coverage_hold": coverage_hold_list,
            "months_start": date_df['start_date'].tolist(),
            "months_end": date_df['end_date'].tolist()
        }

        return coverage

    '''
    '
    '''
    @staticmethod
    def get_avg_coverage_per_month(predictions_folder, walks_list, nets_list, epochs): 
        df = pd.DataFrame()
        
        for net_id, net in enumerate(nets_list):

            for walk_id, walk in enumerate(walks_list):
                df_net = pd.read_csv(predictions_folder + walk + '/' + net)
                df_net = df_net.drop(df_net.columns[0], axis=1)
                df = pd.concat([df, df_net])

            date_df = df.copy()
            date_df = date_df[['date_time']]
            date_df['start_date'] = date_df['date_time'].tolist()
            date_df['date_time'] = pd.to_datetime(date_df['date_time'])
            date_df = date_df.groupby(pd.Grouper(key='date_time', freq='M'), sort=True).agg({'start_date': 'first'}).reset_index()
            date_df = date_df.rename(columns={"date_time": "end_date"})
            date_df['end_date'] = date_df['end_date'].astype(str)
            
            coverage_long_df = pd.DataFrame(columns=df.columns, data={'date_time': date_df['start_date'].tolist()})
            coverage_short_df = pd.DataFrame(columns=df.columns, data={'date_time': date_df['start_date'].tolist()})
            coverage_long_short_df = pd.DataFrame(columns=df.columns, data={'date_time': date_df['start_date'].tolist()})
            coverage_hold_df = pd.DataFrame(columns=df.columns, data={'date_time': date_df['start_date'].tolist()})

            
            for epoch_id in range(1, epochs + 1): 
                startt = time.time()

                coverage_long_list = []
                coverage_short_list = []
                coverage_long_short_list = []
                coverage_hold_list = []

                for i in range(0, date_df.shape[0]):
                    min_df = Market.get_df_by_data_range(df=df.copy(), start_date=date_df.iloc[i].start_date, end_date=date_df.iloc[i].end_date)
                    

                    y_pred = min_df['epoch_' + str(epoch_id)].tolist()

                    if len(y_pred) > 0:
                        long_count = y_pred.count(2)
                        short_count = y_pred.count(0)
                        hold_count = y_pred.count(1)
                        # percentuale di operazioni di long e shorts sul totale di operazioni fatte
                        coverage_long_list.append( long_count / (len(y_pred)) )
                        coverage_short_list.append( short_count / (len(y_pred)) )
                        coverage_long_short_list.append( (long_count + short_count) / (len(y_pred)) )
                        coverage_hold_list.append( hold_count / (len(y_pred)) )
                    else: 
                        coverage_long_list.append( 0 )
                        coverage_short_list.append( 0 )
                        coverage_long_short_list.append( 0 )
                        coverage_hold_list.append( 0 )

                coverage_long_list = [i * 100 for i in coverage_long_list]
                coverage_short_list = [i * 100 for i in coverage_short_list]
                coverage_long_short_list = [i * 100 for i in coverage_long_short_list]
                coverage_hold_list = [i * 100 for i in coverage_hold_list]
                endt = time.time()
                print('Epoca n: ' + str(epoch_id) + ' conclusa in:', endt-startt)
                coverage_long_df['epoch_' + str(epoch_id)] = coverage_long_list
                coverage_short_df['epoch_' + str(epoch_id)] = coverage_short_list
                coverage_long_short_df['epoch_' + str(epoch_id )] = coverage_long_short_list
                coverage_hold_df['epoch_' + str(epoch_id)] = coverage_hold_list

            print(coverage_long_df)
            input()
            coverage_long_df = coverage_long_df.set_index('date_time')
            print(coverage_long_df.mean(axis=1))

            coverage = {
                "coverage_long": coverage_long_list, 
                "coverage_short": coverage_short_list,
                "coverage_long_short": coverage_long_short_list, 
                "coverage_hold": coverage_hold_list,
                "months_start": date_df['start_date'].tolist(),
                "months_end": date_df['end_date'].tolist()
            }

        return coverage