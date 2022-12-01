import numpy as np 
from numpy import expand_dims
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import ImageDataGenerator
from matplotlib import pyplot
# load the image
img = load_img('C:/Users/Utente/Documents/GitHub/PhD-Market-Nets/images/sp500_cet_perc_200/5min_227/gadf/delta_current_day_percentage/2020-07-27.png')
img2 = load_img('C:/Users/Utente/Documents/GitHub/PhD-Market-Nets/images/sp500_cet_perc_200/5min_227/gadf/delta_current_day_percentage/2020-07-28.png')

data = np.zeros(shape=(2, 227, 227, 3))
# convert to numpy array
data[0] = img_to_array(img)
data[1] = img_to_array(img)

print(data.shape)
# expand dimension to one sample
#samples = expand_dims(data, 0)

# create image data augmentation generator
datagen = ImageDataGenerator(
    rotation_range=30,
	zoom_range=0.15,
	width_shift_range=0.2,
	height_shift_range=0.2,
	shear_range=0.15,
	horizontal_flip=True,
	fill_mode="nearest"
    )

# prepare iterator
it = datagen.flow(x=data, y=[1, 0], batch_size=2)


counter = 0
# generate samples and plot
for i in it.next():
    print(i[0].shape)
    
    
    # define subplot
    pyplot.subplot(330 + 1 + counter)

    #generate batch of images
    #batch = it.next()
    # convert to unsigned integers for viewing
    image = i[0].astype('uint8')
    # plot raw pixel data
    pyplot.imshow(image)
    counter += 1
    # show the figure
pyplot.show()
