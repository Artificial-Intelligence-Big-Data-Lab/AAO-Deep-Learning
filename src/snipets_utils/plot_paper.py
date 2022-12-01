import pandas as pd 
import numpy as np 
from sklearn import metrics
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

thrs = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

# BUY HOLD
BH_net_profit = [56600, 56600, 56600, 56600, 56600, 56600]
BH_romad = [3.847, 3.847, 3.847, 3.847, 3.847, 3.847]
BH_annualized_sharpe_ratio = [0.258, 0.258, 0.258, 0.258, 0.258, 0.258]
BH_sortino_ratio = [0.394, 0.394, 0.394, 0.394, 0.394, 0.394]

# RANDOM GUESSING
rd_net_profit = [-41912.5, -9575, -5775, -1275, 0, 0 ]
rd_romad = [-0.9914, -0.5961, -0.6047, -0.6107, 0, 0]
rd_annualized_sharpe_ratio = [-0.9305, -0.6612, -1.6652, -4.0674, 0, 0]
rd_sortino_ratio = [-0.3951, -0.265, -0.5860, -1.3311, 0, 0]

# CNN 1D
cnn1d_net_profit = [-69525, -52250, -40725, -13287.5, -6650, 1337.5]
cnn1d_romad = [-0.9048, -0.8326, -0.9279, -0.7496, -0.4634, 0.2003]
cnn1d_annualized_sharpe_ratio = [-0.9361, -0.9211, -1.1723, -0.7436, -0.8369, -0.5820]
cnn1d_sortino_ratio = [-0.4252, -0.4202, -0.4510, -0.2812, -0.3873, -0.2930]

# CNN 2D

cnn2d_net_profit = [66625, 82312.5, 62600, 46212.5, 17487.5, 1375]
cnn2d_romad = [8.289, 11.950, 8.517, 9.105, 3.97, 0.365]
cnn2d_annualized_sharpe_ratio = [1.196,  1.808, 1.518, 1.596, 0.452, -1.445]
cnn2d_sortino_ratio = [0.452, 0.828, 1.009, 1.504, 0.370, -0.524]

plt.figure(figsize=(12, 9))

# LEGGENDE DA METTERE DENTRO UN SUBPLOT PER ESSERE VISUALIZZATE
red_patch = mpatches.Patch(color='red', label='Buy Hold')
orange_patch = mpatches.Patch(color='orange', label='Random Guessing')
green_patch = mpatches.Patch(color='green', label='CNN 1D')
blue_patch = mpatches.Patch(color='blue', label='CNN 2D (our approach)')
plt.legend(handles=[red_patch, orange_patch, green_patch, blue_patch])

plt.subplot(2, 2, 1)
plt.plot(thrs, rd_net_profit, color="orange")
plt.plot(thrs, cnn1d_net_profit, color="green")
plt.plot(thrs, BH_net_profit, color="red")
plt.plot(thrs, cnn2d_net_profit, color="blue")
plt.xlabel('Threshold')
plt.ylabel('Net Profit')
plt.grid(color='black', linestyle='-', linewidth=0.5)

plt.subplot(2, 2, 2)
plt.plot(thrs, rd_romad, color="orange")
plt.plot(thrs, cnn1d_romad, color="green")
plt.plot(thrs, BH_romad, color="red")
plt.plot(thrs, cnn2d_romad, color="blue")
plt.xlabel('Threshold')
plt.ylabel('Romad')
plt.grid(color='black', linestyle='-', linewidth=0.5)


plt.subplot(2, 2, 3)
plt.plot(thrs, rd_annualized_sharpe_ratio, color="orange")
plt.plot(thrs, cnn1d_annualized_sharpe_ratio, color="green")
plt.plot(thrs, BH_annualized_sharpe_ratio, color="red")
plt.plot(thrs, cnn2d_annualized_sharpe_ratio, color="blue")
plt.xlabel('Threshold')
plt.ylabel('Annualized Sharpe Ratio')
plt.grid(color='black', linestyle='-', linewidth=0.5)


plt.subplot(2, 2, 4)
plt.plot(thrs, rd_sortino_ratio, color="orange")
plt.plot(thrs, cnn1d_sortino_ratio, color="green")
plt.plot(thrs, BH_sortino_ratio, color="red")
plt.plot(thrs, cnn2d_sortino_ratio, color="blue")
plt.xlabel('Threshold')
plt.ylabel('Sortino Ratio')
plt.grid(color='black', linestyle='-', linewidth=0.5)


plt.show()