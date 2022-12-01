import numpy as np

# PESI LONG
loss_weight = w_array = np.ones((3,3))
loss_weight[2, 2] = 0
loss_weight[1, 2] = 7
loss_weight[0, 2] = 9

loss_weight[2, 1] = 1
loss_weight[1, 1] = 0
loss_weight[0, 1] = 1

loss_weight[2, 0] = 1
loss_weight[1, 0] = 1
loss_weight[0, 0] = 0

# PESI SHORT 
loss_weight = w_array = np.ones((3,3))
loss_weight[2, 2] = 0
loss_weight[1, 2] = 1
loss_weight[0, 2] = 1

loss_weight[2, 1] = 1
loss_weight[1, 1] = 0
loss_weight[0, 1] = 1

loss_weight[2, 0] = 9
loss_weight[1, 0] = 7
loss_weight[0, 0] = 0


# PESI BILANCIATI
loss_weight = w_array = np.ones((3,3))
loss_weight[2, 2] = 0
loss_weight[1, 2] = 4
loss_weight[0, 2] = 6

loss_weight[2, 1] = 1
loss_weight[1, 1] = 0
loss_weight[0, 1] = 1

loss_weight[2, 0] = 6
loss_weight[1, 0] = 4
loss_weight[0, 0] = 0