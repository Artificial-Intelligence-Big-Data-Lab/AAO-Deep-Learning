# visualize feature maps output from each block in the vgg model
from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.models import Model
from matplotlib import pyplot
from numpy import expand_dims
from vgg16.SmallerVGGNet import SmallerVGGNet
from keras.models import load_model
from classes.CustomLoss import w_categorical_crossentropy
import numpy as np
import functools
# load the model
#model = VGG16()

loss_weight = w_array = np.ones((3,3))
loss_weight[2, 2] = 0
loss_weight[1, 2] = 2
loss_weight[0, 2] = 6

loss_weight[2, 1] = 2
loss_weight[1, 1] = 0
loss_weight[0, 1] = 2

loss_weight[2, 0] = 10
loss_weight[1, 0] = 4
loss_weight[0, 0] = 0
ncce = functools.partial(w_categorical_crossentropy, weights=loss_weight)
ncce.__name__ ='w_categorical_crossentropy'

#model = SmallerVGGNet.build_parallel_2d(height=20, width=20, depth=3, init_var=0)
#model = load_model('C:/Users/Utente/Desktop/img-demo/walk_0_net_0.model', custom_objects={ 'w_categorical_crossentropy': ncce})
#model = load_model('C:/Users/Utente/Desktop/img-demo/walk_0_net_10.model', custom_objects={ 'w_categorical_crossentropy': ncce})
#model = load_model('C:/Users/Utente/Desktop/img-demo/walk_0_net_19.model', custom_objects={ 'w_categorical_crossentropy': ncce})
model = load_model('C:/Users/Utente/Desktop/img-demo/walk_0_net_71.model', custom_objects={ 'w_categorical_crossentropy': ncce})
model.summary()

#######################################
# FILTERS
#######################################

# retrieve weights from the second hidden layer
filters, biases = model.layers[2].get_weights()
# normalize filter values to 0-1 so we can visualize them
f_min, f_max = filters.min(), filters.max()
filters = (filters - f_min) / (f_max - f_min)
# plot first few filters
n_filters, ix = 6, 1
for i in range(n_filters):
	# get the filter
	f = filters[:, :, :, i]
	# plot each channel separately
	for j in range(3):
		# specify subplot and turn of axis
		ax = pyplot.subplot(n_filters, 3, ix)
		ax.set_xticks([])
		ax.set_yticks([])
		# plot filter channel in grayscale
		pyplot.imshow(f[:, :, j], cmap='gray')
		ix += 1
# show the figure
#pyplot.show()

#######################################
#  FEATURE MAP 
#######################################
 

# # redefine model to output right after the first hidden layer
ixs = [0, 1, 2, 3] # tutti i livelli della small cnn andrea seba
outputs = [model.layers[i].output for i in ixs]
model = Model(inputs=model.inputs, outputs=outputs)

# load the image with the required shape
#img = load_img('C:/Users/Utente/Desktop/img-demo/ff7-01.jpg', target_size=(40, 40))
img = load_img('C:/Users/Utente/Desktop/img-demo/gaf-demo.png', target_size=(40, 40))

# convert the image to an array
img = img_to_array(img)
# expand dimensions so that it represents a single 'sample'
img = expand_dims(img, axis=0)
# prepare the image (e.g. scale pixel values for the vgg)
img = preprocess_input(img)
# get feature map for first hidden layer
feature_maps = model.predict(img)

# plot the output from each block

for fmap_id, fmap in enumerate(feature_maps):
    if fmap_id < 2:
        square = 5
    else:
        square = 8

    # plot all 64 maps in an 8x8 squares    
    ix = 1

    for _ in range(square):
        for _ in range(square):
            # specify subplot and turn of axis
            ax = pyplot.subplot(square, square, ix)
            ax.set_xticks([])
            ax.set_yticks([])
            # plot filter channel in grayscale
            pyplot.imshow(fmap[0, :, :, ix-1], cmap='gray')
            #pyplot.imshow(fmap[0, :, :, ix-1], cmap='rainbow')
            ix += 1
    # show the figure
    pyplot.show()