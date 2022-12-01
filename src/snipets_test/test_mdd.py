import pandas as pd
import numpy as np
from classes.Market import Market
import matplotlib.pyplot as plt

#########################################################################################

def mdd_simple(series):
    # calcolo la posizione i-esima
    i = np.argmax(np.maximum.accumulate(series) - series) # end of the period
    # calcolo la posizione j-esima
    j = np.argmax(xs[:i]) # start of period

    # mdd
    mdd = series[j] - series[i] 
    # romad
    romad = sum(series) / mdd 
    # return totale
    global_return = sum(series) 

    return mdd, romad, global_return, i, j, series[i], series[j]


#############################################################################

sp500 = Market(dataset='sp500')
one_d = sp500.group(freq='1d', nan=False)

xs = one_d['close'].tail(1000).values.flatten().tolist()

mdd, romad, global_return, i, j, i_val, j_val = mdd_simple(xs)

print(" ++++ Stackoverflow ++++ ")
print("MDD Pred: " + str(mdd))
print("Computed Return: " + str(global_return))
print("Romad: " + str(romad))
print("I: " + str(i))
print("J: " + str(j))
print("I val: " + str(i_val))
print("J: " + str(j_val))


plt.plot(xs)
plt.plot([i, j], [xs[i], xs[j]], 'o', color='Red', markersize=10)
plt.show()