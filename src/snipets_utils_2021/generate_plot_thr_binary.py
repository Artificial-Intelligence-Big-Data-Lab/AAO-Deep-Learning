import os
import sys
import cv2
import time
import pickle
import datetime
import functools
import matplotlib

import numpy as np
import pandas as pd
from datetime import timedelta
from classes.Utils import create_folder, df_date_merger_binary
import json
import platform
import matplotlib.pyplot as plt

values = {
    'random': {
        'precision': [41.97, 41.97, 41.97, 41.97, 41.97, 41.97, 41.97, 41.97, 41.97, 41.97, 41.97]
    },
    'bh': {
        'return_all': [48125, 48125, 48125, 48125, 48125, 48125, 48125, 48125, 48125, 48125, 48125, 48125],
        'mdd_all': [15150, 15150, 15150, 15150, 15150, 15150, 15150, 15150, 15150, 15150, 15150, 15150],
        'romad_all': [3.18, 3.18, 3.18, 3.18, 3.18, 3.18, 3.18, 3.18, 3.18, 3.18, 3.18, 3.18],
        'return_avg': [16383.33, 16383.33, 16383.33, 16383.33, 16383.33, 16383.33, 16383.33, 16383.33, 16383.33, 16383.33, 16383.33, 16383.33],
        'mdd_avg': [12350, 12350, 12350, 12350, 12350, 12350, 12350, 12350, 12350, 12350, 12350, 12350],
        'romad_avg': [1.67, 1.67, 1.67, 1.67, 1.67, 1.67, 1.67, 1.67, 1.67, 1.67, 1.67, 1.67]
    },
    'bh_2':{
        'return_all': [42000, 42000, 42000, 42000, 42000, 42000, 42000, 42000, 42000, 42000, 42000, 42000],
        'mdd_all': [12737.5, 12737.5, 12737.5, 12737.5, 12737.5, 12737.5, 12737.5, 12737.5, 12737.5, 12737.5, 12737.5, 12737.5],
        'romad_all': [3.3, 3.3, 3.3, 3.3, 3.3, 3.3, 3.3, 3.3, 3.3, 3.3, 3.3, 3.3],
        'return_avg': [14000, 14000, 14000, 14000, 14000, 14000, 14000, 14000, 14000, 14000, 14000, 14000],
        'mdd_avg': [10833.33, 10833.33, 10833.33, 10833.33, 10833.33, 10833.33, 10833.33, 10833.33, 10833.33, 10833.33, 10833.33, 10833.33],
        'romad_avg': [1.63, 1.63, 1.63, 1.63, 1.63, 1.63, 1.63, 1.63, 1.63, 1.63, 1.63, 1.63]
    },
    '0.4': {
        'precision': [42.3, 42.39, 42.71, 42.73, 42.47, 42.39, 41.93, 42.0, 41.82, 42.13, 41.92],
        'por': [0.79, 1.0, 1.76, 1.81, 1.19, 1.0, -0.1, 0.07, -0.36, 0.38, -0.12],
        'cvg': [68.53, 73.32, 80.18, 82.26, 86.14, 87.44, 89.44, 91.13, 95.14, 96.96, 99.36],
        'return': [6862.5, 2683.33, 5975.0, 8200.0, 7729.17, 3529.17, 337.5, -220.83, -266.67, -229.17, -133.33],
        'mdd': [7633.33, 6079.17, 6104.17, 4662.5, 4429.17, 4354.17, 3983.33, 5012.5, 3683.33, 3041.67, 1062.5],
        'romad': [1.48, 0.83, 1.22, 1.89, 2.7, 1.48, 0.24, 0.44, -0.04, 0.67, -0.18]
    },
    '0.3': {
        'precision': [43.95, 42.21, 43.11, 42.6, 42.18, 42.02, 42.32, 41.81, 41.78, 42.06, 41.73],
        'por': [4.72, 0.57, 2.72, 1.5, 0.5, 0.12, 0.83, -0.38, -0.45, 0.21, -0.57],
        'cvg': [59.26, 65.68, 74.55, 76.43, 82.39, 83.94, 86.99, 89.25, 94.11, 97.41, 99.35],
        'return': [12091.67, 6612.5, 8120.83, 8212.5, 3512.5, 1120.83, 1975.0, -2404.17, 1795.83, -1504.17, -975.0],
        'mdd': [6141.67, 8745.83, 6462.5, 6091.67, 6708.33, 5979.17, 5900.0, 5433.33, 3000.0, 3470.83, 1725.0],
        'romad': [2.12, 1.23, 2.33, 1.36, 0.56, 0.27, 1.12, -0.39, 0.79, -0.31, -0.56]
    },
    '0.2': {
        'precision': [41.18, 41.1, 42.51, 42.4, 42.66, 41.88, 42.05, 41.77, 41.62, 41.56, 41.88],
        'por': [-1.88, -2.07, 1.29, 1.02, 1.64, -0.21, 0.19, -0.48, -0.83, -0.98, -0.21],
        'cvg': [46.12, 50.2, 61.08, 65.87, 72.67, 75.33, 79.86, 83.94, 91.13, 95.47, 98.84],
        'return': [10950.0, 6050.0, 9345.83, 7925.0, 6516.67, 5866.67, 3283.33, 6104.17, -233.33, -558.33, -512.5],
        'mdd': [8441.67, 9916.67, 8200.0, 7258.33, 8745.83, 7479.17, 7270.83, 5179.17, 5279.17, 3012.5, 1662.5],
        'romad': [1.62, 0.73, 1.56, 1.21, 0.98, 0.91, 0.39, 1.42, -0.13, -0.11, -0.08]
    },
    '0.1': {
        'precision': [42.29, 40.91, 41.09, 40.54, 42.55, 41.65, 41.7, 42.18, 42.04, 42.02, 41.85],
        'por': [0.76, -2.53, -2.1, -3.41, 1.38, -0.76, -0.64, 0.5, 0.17, 0.12, -0.29],
        'cvg': [36.46, 41.26, 53.89, 57.39, 64.83, 69.44, 72.67, 79.6, 88.99, 95.66, 99.35],
        'return': [6895.83, 9541.67, 8183.33, 7091.67, 8983.33, 9054.17, 6295.83, 6583.33, 3266.67, 1904.17, -308.33],
        'mdd': [11033.33, 8845.83, 9500.0, 9470.83, 8533.33, 6866.67, 6216.67, 6229.17, 4833.33, 2400.0, 675.0],
        'romad': [0.97, 1.1, 1.16, 1.27, 1.38, 1.45, 1.03, 2.6, 1.81, 3.12, -0.48]
    },
    '-0.1': {
        'precision': [42.74, 39.29, 43.09, 41.8, 42.42, 41.64, 39.87, 41.76, 41.92, 41.94, 42.17],
        'por': [1.83, -6.39, 2.67, -0.41, 1.07, -0.79, -5.0, -0.5, -0.12, -0.07, 0.48],
        'cvg': [17.87, 21.18, 29.92, 33.49, 42.68, 45.53, 52.98, 62.7, 77.46, 91.06, 98.58],
        'return': [13466.67, 5991.67, 9879.17, 8462.5, 12620.83, 9679.17, 1495.83, 4412.5, 4716.67, 2562.5, 1695.83],
        'mdd': [9575.0, 11770.83, 9895.83, 9720.83, 8700.0, 9854.17, 11933.33, 8895.83, 6770.83, 4137.5, 729.17],
        'romad': [1.72, 1.07, 1.03, 0.99, 1.5, 0.98, 0.18, 0.51, 0.63, 0.98, 2.47]
    },
    '-0.2': {
        'precision': [40.0, 44.39, 43.68, 44.08, 43.67, 41.11, 42.44, 44.58, 43.15, 42.57, 42.31],
        'por': [-4.69, 5.77, 4.07, 5.03, 4.05, -2.05, 1.12, 6.22, 2.81, 1.43, 0.81],
        'cvg': [13.66, 16.12, 22.47, 26.75, 34.07, 38.86, 43.91, 52.98, 66.78, 82.64, 96.44],
        'return': [11683.33, 12825.0, 11212.5, 13287.5, 12995.83, 7216.67, 10720.83, 15783.33, 11491.67, 6887.5, 1933.33],
        'mdd': [10025.0, 9687.5, 8620.83, 9954.17, 9750.0, 9345.83, 8525.0, 6004.17, 5712.5, 4254.17, 1891.67],
        'romad': [1.82, 2.14, 1.51, 1.94, 1.49, 0.9, 1.4, 2.7, 2.46, 1.99, 1.97]
    },
    '-0.3':{
        'precision': [32.1, 40.38, 41.42, 39.2, 39.9, 41.29, 40.89, 40.46, 41.13, 41.55, 42.15],
        'por': [-23.52, -3.79, -1.31, -6.6, -4.93, -1.62, -2.57, -3.6, -2.0, -1.0, 0.43],
        'cvg': [8.29, 11.01, 15.93, 19.11, 26.55, 29.6, 35.1, 43.39, 60.17, 76.88, 93.91],
        'return': [4912.5, 10900.0, 10979.17, 7162.5, 4804.17, 5016.67, 9712.5, 2666.67, -1116.67, 2983.33, 3083.33],
        'mdd': [12754.17, 11133.33, 12104.17, 11233.33, 12779.17, 13437.5, 10454.17, 9433.33, 10850.0, 6170.83, 3429.17],
        'romad': [0.68, 1.18, 0.9, 0.97, 0.54, 0.45, 1.12, 0.42, 0.07, 0.88, 1.57]
    },
    '-0.4': {
        'precision': [44.14, 41.34, 41.14, 43.89, 42.53, 39.12, 43.55, 40.33, 42.74, 44.04, 41.84],
        'por': [5.17, -1.5, -1.98, 4.57, 1.33, -6.79, 3.76, -3.91, 1.83, 4.93, -0.31],
        'cvg': [6.15, 7.96, 11.72, 13.73, 19.88, 22.66, 27.26, 33.74, 51.75, 68.01, 86.92],
        'return': [12545.83, 11412.5, 14029.17, 15641.67, 15404.17, 5862.5, 12875.0, 7862.5, 12770.83, 16279.17, 2762.5],
        'mdd': [11758.33, 12158.33, 10454.17, 10275.0, 9479.17, 11895.83, 10000.0, 11316.67, 8850.0, 5145.83, 3870.83],
        'romad': [1.68, 1.4, 1.58, 1.95, 1.78, 0.85, 1.63, 0.75, 1.46, 4.38, 0.99]
    },
    '-0.5': {
        'precision': [38.93, 42.55, 41.91, 38.59, 39.52, 41.41, 41.51, 39.36, 40.82, 41.85, 42.82],
        'por': [-7.24, 1.38, -0.14, -8.05, -5.84, -1.33, -1.1, -6.22, -2.74, -0.29, 2.03],
        'cvg': [5.63, 6.86, 9.45, 11.2, 16.58, 19.3, 23.06, 28.43, 43.33, 60.88, 81.61],
        'return': [12095.83, 13962.5, 13158.33, 10620.83, 10383.33, 9858.33, 12625.0, 8191.67, 10441.67, 6379.17, 7545.83],
        'mdd': [11108.33, 10716.67, 9795.83, 11762.5, 10695.83, 11075.0, 11258.33, 10425.0, 10083.33, 7170.83, 4225.0],
        'romad': [1.77, 1.83, 1.69, 1.48, 1.25, 1.04, 1.59, 0.81, 1.02, 0.89, 2.24]
    },
    '-0.7': {
        'precision': [27.78, 35.63, 31.88, 44.0, 37.81, 43.24, 41.19, 40.28, 42.28, 43.28, 42.07],
        'por': [-33.81, -15.11, -24.04, 4.84, -9.91, 3.03, -1.86, -4.03, 0.74, 3.12, 0.24],
        'cvg': [3.63, 4.27, 6.41, 6.6, 9.58, 11.07, 13.27, 16.84, 28.69, 42.03, 64.12],
        'return': [9754.17, 11916.67, 9779.17, 11287.5, 12045.83, 12908.33, 10650.0, 8945.83, 13700.0, 12400.0, 4487.5],
        'mdd': [11675.0, 11041.67, 12412.5, 11779.17, 12170.83, 10591.67, 11704.17, 13541.67, 8016.67, 6350.0, 6066.67],
        'romad': [1.29, 1.78, 1.21, 1.66, 1.37, 1.49, 1.3, 0.99, 1.83, 1.98, 0.83]
    },
    '-0.9': {
        'precision': [21.8, 29.09, 35.48, 30.91, 36.87, 39.02, 40.13, 39.24, 40.19, 42.39, 41.26],
        'por': [-48.06, -30.69, -15.46, -26.35, -12.15, -7.03, -4.38, -6.5, -4.24, 1.0, -1.69],
        'cvg': [2.53, 3.43, 4.73, 5.25, 6.87, 7.38, 8.74, 11.07, 16.64, 28.3, 47.67],
        'return': [10733.33, 12075.0, 11950.0, 11304.17, 13183.33, 10295.83, 12204.17, 8612.5, 7670.83, 10104.17, 504.17],
        'mdd': [11804.17, 10887.5, 10095.83, 11983.33, 9562.5, 11300.0, 11025.0, 11204.17, 11670.83, 9062.5, 9741.67],
        'romad': [1.52, 1.76, 1.66, 1.34, 2.05, 1.42, 1.44, 1.65, 1.47, 1.31, 0.23],
    },
    '-1.1': {
        'precision': [17.62, 34.76, 40.94, 39.03, 48.13, 42.44, 40.18, 41.52, 43.31, 41.63, 41.4],
        'por': [-58.02, -17.18, -2.45, -7.01, 14.68, 1.12, -4.26, -1.07, 3.19, -0.81, -1.36],
        'cvg': [1.29, 1.81, 2.66, 3.43, 4.86, 5.44, 6.02, 8.81, 13.28, 19.82, 35.62],
        'return': [11645.83, 13504.17, 14070.83, 12045.83, 16179.17, 13483.33, 12408.33, 9541.67, 9716.67, 7450.0, 7529.17],
        'mdd': [10933.33, 10304.17, 9591.67, 11166.67, 9229.17, 10337.5, 10408.33, 11516.67, 8616.67, 9304.17, 7304.17],
        'romad': [1.71, 1.9, 1.69, 1.41, 1.95, 1.41, 1.22, 1.11, 1.43, 0.96, 1.4]
    }
}



''' POR '''
fig, ax = plt.subplots(figsize=(22,9))

ax.plot(values['0.4']['cvg'], np.zeros(11), color="black", linestyle='--')
ax.plot(values['0.3']['cvg'], np.zeros(11), color="black", linestyle='--')
ax.plot(values['0.2']['cvg'], np.zeros(11), color="black", linestyle='--')
ax.plot(values['0.1']['cvg'], np.zeros(11), color="black", linestyle='--')
ax.plot(values['-0.1']['cvg'], np.zeros(11), color="black", linestyle='--')
ax.plot(values['-0.2']['cvg'], np.zeros(11), color="black", linestyle='--')
ax.plot(values['-0.3']['cvg'], np.zeros(11), color="black", linestyle='--')
ax.plot(values['-0.4']['cvg'], np.zeros(11), color="black", linestyle='--')
ax.plot(values['-0.5']['cvg'], np.zeros(11), color="black", linestyle='--')
ax.plot(values['-0.7']['cvg'], np.zeros(11), color="black", linestyle='--')
ax.plot(values['-0.9']['cvg'], np.zeros(11), color="black", linestyle='--')
ax.plot(values['-1.1']['cvg'], np.zeros(11), color="black", linestyle='--')

'''
# MAGGIORI DI 0 
ax.plot(values['0.4']['cvg'], values['0.4']['por'], label="PoR Thr 0.4")
for i,j in zip(values['0.4']['cvg'], values['0.4']['por']):
    ax.annotate(str(j),xy=(i,j))

ax.plot(values['0.3']['cvg'], values['0.3']['por'], label="PoR Thr 0.3")
for i,j in zip(values['0.3']['cvg'], values['0.3']['por']):
    ax.annotate(str(j),xy=(i,j))

ax.plot(values['0.2']['cvg'], values['0.2']['por'], label="PoR Thr 0.2")
for i,j in zip(values['0.2']['cvg'], values['0.2']['por']):
    ax.annotate(str(j),xy=(i,j))

ax.plot(values['0.1']['cvg'], values['0.1']['por'], label="PoR Thr 0.1")
for i,j in zip(values['0.1']['cvg'], values['0.1']['por']):
    ax.annotate(str(j),xy=(i,j))
'''

# MINORI DI 0 
'''
ax.plot(values['-0.1']['cvg'], values['-0.1']['por'], label="PoR Thr -0.1")
for i,j in zip(values['-0.1']['cvg'], values['-0.1']['por']):
    ax.annotate(str(j),xy=(i,j))

ax.plot(values['-0.2']['cvg'], values['-0.2']['por'], label="PoR Thr -0.2")
for i,j in zip(values['-0.2']['cvg'], values['-0.2']['por']):
    ax.annotate(str(j),xy=(i,j))

ax.plot(values['-0.3']['cvg'], values['-0.3']['por'], label="PoR Thr -0.3")
for i,j in zip(values['-0.3']['cvg'], values['-0.3']['por']):
    ax.annotate(str(j),xy=(i,j))

ax.plot(values['-0.4']['cvg'], values['-0.4']['por'], label="PoR Thr -0.4")
for i,j in zip(values['-0.4']['cvg'], values['-0.4']['por']):
    ax.annotate(str(j),xy=(i,j))

'''
ax.plot(values['-0.5']['cvg'], values['-0.5']['por'], label="PoR Thr -0.5")
for i,j in zip(values['-0.5']['cvg'], values['-0.5']['por']):
    ax.annotate(str(j),xy=(i,j))

ax.plot(values['-0.7']['cvg'], values['-0.7']['por'], label="PoR Thr -0.7")
for i,j in zip(values['-0.7']['cvg'], values['-0.7']['por']):
    ax.annotate(str(j),xy=(i,j))

ax.plot(values['-0.9']['cvg'], values['-0.9']['por'], label="PoR Thr -0.9")
for i,j in zip(values['-0.9']['cvg'], values['-0.9']['por']):
    ax.annotate(str(j),xy=(i,j))

ax.plot(values['-1.1']['cvg'], values['-1.1']['por'], label="PoR Thr -1.1")
for i,j in zip(values['-1.1']['cvg'], values['-1.1']['por']):
    ax.annotate(str(j),xy=(i,j))

ax.set(xlabel='CVG', ylabel='PoR', title='')
ax.grid()
plt.legend(loc='best')

path = 'C:/Users/Utente/Desktop/Dataset Json Classifier Vix/charts_multiple_thrs/'
create_folder(path)
plt.savefig(path + 'POR_chart_thr_-0.5_-1.1.png')


''' precision ''
fig, ax = plt.subplots(figsize=(22,9))

ax.plot(cvg_01, random_precision, color="black", linestyle='--')
ax.plot(cvg_02, random_precision, color="black", linestyle='--')
ax.plot(cvg_03, random_precision, color="black", linestyle='--')
ax.plot(cvg_04, random_precision, color="black", linestyle='--')
ax.plot(cvg_05, random_precision, color="black", linestyle='--')
ax.plot(cvg_07, random_precision, color="black", linestyle='--')
ax.plot(cvg_09, random_precision, color="black", linestyle='--')
ax.plot(cvg_1_1, random_precision, color="black", linestyle='--')

# MAGGIORI DI 0
ax.plot(cvg_p04, delta_precision_p04, label="Precision Thr 0.4")
for i,j in zip(cvg_p04, delta_precision_p04):
    ax.annotate(str(j) ,xy=(i,j))

ax.plot(cvg_p03, delta_precision_p03, label="Precision Thr 0.3")
for i,j in zip(cvg_p03, delta_precision_p03):
    ax.annotate(str(j) ,xy=(i,j))

ax.plot(cvg_p02, delta_precision_p02, label="Precision Thr 0.2")
for i,j in zip(cvg_p02, delta_precision_p02):
    ax.annotate(str(j) ,xy=(i,j))

ax.plot(cvg_p01, delta_precision_p01, label="Precision Thr 0.1")
for i,j in zip(cvg_p01, delta_precision_p01):
    ax.annotate(str(j) ,xy=(i,j))

ax.plot(cvg_p0, delta_precision_p0, label="Precision Thr 0")
for i,j in zip(cvg_p0, delta_precision_p0):
    ax.annotate(str(j) ,xy=(i,j))
''
# MINORI DI 0
ax.plot(cvg_01, delta_precision_01, label="Precision Thr -0.1")
for i,j in zip(cvg_01, delta_precision_01):
    ax.annotate(str(j) ,xy=(i,j))

ax.plot(cvg_02, delta_precision_02, label="Precision Thr -0.2")
for i,j in zip(cvg_02, delta_precision_02):
    ax.annotate(str(j) ,xy=(i,j))

ax.plot(cvg_03, delta_precision_03, label="Precision Thr -0.3")
for i,j in zip(cvg_03, delta_precision_03):
    ax.annotate(str(j) ,xy=(i,j))

ax.plot(cvg_04, delta_precision_04, label="Precision Thr -0.4")
for i,j in zip(cvg_04, delta_precision_04):
    ax.annotate(str(j) ,xy=(i,j))


ax.plot(cvg_05, delta_precision_05, label="Precision Thr -0.5")
for i,j in zip(cvg_05, delta_precision_05):
    ax.annotate(str(j) ,xy=(i,j))

ax.plot(cvg_07, delta_precision_07, label="Precision Thr -0.7")
for i,j in zip(cvg_07, delta_precision_07):
    ax.annotate(str(j) ,xy=(i,j))

ax.plot(cvg_09, delta_precision_09, label="Precision Thr -0.9")
for i,j in zip(cvg_09, delta_precision_09):
    ax.annotate(str(j) ,xy=(i,j))

ax.plot(cvg_1_1, delta_precision_1_1, label="Precision Thr -1.1")
for i,j in zip(cvg_1_1, delta_precision_1_1):
    ax.annotate(str(j) ,xy=(i,j))

ax.set(xlabel='CVG', ylabel='Precision', title='')
ax.grid()
plt.legend(loc='best')

path = 'C:/Users/Utente/Desktop/charts_multiple_thrs/'
create_folder(path)
plt.savefig(path + 'PRECISION_chart_thr_positive.png')
'''