import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import sys
import cv2
import time
import pickle
import datetime
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from keras.models import load_model
from vgg16.SmallerVGGNet import SmallerVGGNet
from sklearn.preprocessing import LabelBinarizer
from keras.optimizers import Adam, RMSprop, SGD, Nadam
from keras import backend as K
from keras.callbacks import ModelCheckpoint
from sklearn.model_selection import train_test_split

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID";
os.environ["CUDA_VISIBLE_DEVICES"]="1";

datasets_path_original = '../datasets/PetImages/'
dataset_path_resized = '../datasets/pet_images_resized/'

output_path = '../experiments/cat_dog_small_net/'

classes = ['dog', 'cat'] # dog = 0, cat = 1

first_index = 0
middle_index = 12000
last_index = (12499)

__number_of_nets = 20
__epochs = 100
__init_lr = 0.001
__bs = 32

def images_filename_reader(path):
        filename_list = []

        for file in os.listdir(path):
            filename_list.append(file)

        return filename_list

'''
'
'''
def resize_images(): 

    for i in range(first_index, last_index):
        print("Leggo immagini con indice " + str(i))

        dog_img = datasets_path_original + classes[0] + '/' + str(i) + '.jpg'
        cat_img = datasets_path_original + classes[1] + '/' + str(i) + '.jpg'
        
        try:
            img =  cv2.imread(dog_img)
            img_resized = cv2.resize(img,(40, 40))
            cv2.imwrite(dataset_path_resized + classes[0] + '/' + str(i) + '.jpg', img_resized)
        except Exception as e:
            print("Ignorata immagine con indice " + str(i) + " per i cani")

        try:
            img =  cv2.imread(cat_img)
            img_resized = cv2.resize(img,(40, 40))
            cv2.imwrite(dataset_path_resized + classes[1] + '/' + str(i) + '.jpg', img_resized)

        except Exception as e:
            print("Ignorata immagine con indice " + str(i) + " per i gatti")



def get_train(): 
    dog_path = dataset_path_resized + classes[0] + '/'
    cat_path = dataset_path_resized + classes[1] + '/'

    dog_list = images_filename_reader(dog_path)
    cat_list = images_filename_reader(cat_path)

    dog_size_list = len(dog_list)
    cat_size_list = len(cat_list)
    tot_size = dog_size_list + cat_size_list

    if dog_size_list > cat_size_list:
        max_range = dog_size_list
    else: 
        max_range = cat_size_list

    
    x = np.zeros(shape=(tot_size, 40, 40, 3), dtype=np.uint8)
    # creo il vettore di label prendendole direttamente dal dataframe
    y = np.zeros(shape=(tot_size), dtype=np.uint8)

    img_index_append = 0
    ignored_count = 0

    for i in range(0, max_range):
        #print("Leggo immagini con indice " + str(i))
        dog_img = dataset_path_resized + classes[0] + '/' + str(i) + '.jpg'
        cat_img = dataset_path_resized + classes[1] + '/' + str(i) + '.jpg'
        
        try:
            img =  cv2.imread(dog_img)
            x[img_index_append] =  img
            y[img_index_append] =  0

            img_index_append += 1
        except Exception as e:
            ignored_count += 1
            print("Ignorata immagine " + dataset_path_resized + classes[0] + '/' + str(i) + '.jpg')

        try:
            img =  cv2.imread(cat_img)
            x[img_index_append] =  img
            y[img_index_append] =  1

            img_index_append += 1
        except Exception as e:
            ignored_count += 1
            print("Ignorata immagine " + dataset_path_resized + classes[1] + '/' + str(i) + '.jpg')

    x = x[:-ignored_count, :]
    y = y[:-ignored_count]
     
    return x, y
    
'''
'
'''
def train(x_train, y_train, x_val, y_val, index_net):

        # Carico le immagini
        X_train = x_train.astype('float32') / 255
        X_val = x_val.astype('float32') / 255

        # binarizing labels
        lb = LabelBinarizer()
        y_train = lb.fit_transform(y_train)
        y_val = lb.fit_transform(y_val)

        # [INFO] compiling model...
        model = SmallerVGGNet.build(height=40, width=40, depth=3, classes=len(lb.classes_), init_var=index_net)

        opt = Adam(lr=__init_lr, decay=(__init_lr/ __epochs))
        model.compile(loss="sparse_categorical_crossentropy", optimizer=opt, metrics=["accuracy"])

        H = model.fit(X_train, y_train, batch_size=__bs, validation_data=(x_val, y_val), epochs=__epochs, verbose=1)
       
        return model, H


def train_small(x_train, y_train, x_val, y_val):

    # Carico le immagini
    X_train = x_train.astype('float32') / 255
    X_val = x_val.astype('float32') / 255

    # binarizing labels
    lb = LabelBinarizer()
    y_train = lb.fit_transform(y_train)
    y_val = lb.fit_transform(y_val)

    # [INFO] compiling model...
    model = SmallerVGGNet.build_small(height=40, width=40, depth=3)

    # compile model
    opt = SGD(lr=0.001, momentum=0.9)
    model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])

    H = model.fit(X_train, y_train, batch_size=__bs, validation_data=(x_val, y_val), epochs=__epochs, verbose=1)
    
    return model, H

'''
'
'''
def save_models(model, H, index_net):
    full_models_path = output_path + 'models/'

    # Se non esiste la cartella, la creo
    if not os.path.isdir(full_models_path):
        os.makedirs(full_models_path)

    # Salvo il modello
    model.save(full_models_path + 'net_' + str(index_net) + '.model')
    
'''
' Salvo i plot di accuracy e loss di training e validation
' Li salvo in una sottocartella per ogni walk, in modo da avere 
' dentro la stessa cartella tutti i risultati delle reti per walk
'''
def save_plots(H, index_net):

    full_plots_path = output_path + 'accuracy_loss_plots/'

    # Se non esiste la cartella, la creo
    if not os.path.isdir(full_plots_path):
        os.makedirs(full_plots_path)

    plt.figure(figsize=(15,12))
    # plot the training loss and accuracy
    plt.style.use("ggplot")
    plt.subplot(2, 1, 1)
    plt.plot(np.arange(0, __epochs), H.history["acc"], label="train_acc")
    plt.plot(np.arange(0, __epochs), H.history["val_acc"], label="val_acc")
    #plt.title("Training Accuracy")
    plt.xlabel("Epoch #")
    plt.ylabel("Accuracy")
    plt.legend(loc="upper left")

    plt.subplot(2, 1, 2)
    plt.plot(np.arange(0, __epochs), H.history["loss"], label="train_loss")
    plt.plot(np.arange(0, __epochs), H.history["val_loss"], label="val_loss")
    #plt.title("Training Loss")
    plt.xlabel("Epoch #")
    plt.ylabel("Loss")
    plt.legend(loc="upper left")

    plt.savefig(full_plots_path + 'net_' + str(index_net) + '.png')

    plt.close('all')


#print("Ridimensiono le immagini")
#resize_images()
print("Carico le immagini in x y")
x, y = get_train()

print("Splitto il datasets...")
x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.20, random_state=42)

print("Lancio il training")

model, H = train_small(x_train=x_train, y_train=y_train, x_val=x_val, y_val=y_val)
save_plots(H=H, index_net=1)
save_models(model=model, H=H, index_net=1)

'''
for index_net in range(0, 6): 
    model, H = train(x_train=x_train, y_train=y_train, x_val=x_val, y_val=y_val, index_net=index_net)

    # Salvo il modello ed i grafici della rete
    save_plots(H=H, index_net=index_net)
    save_models(model=model, H=H, index_net=index_net)
    
    K.clear_session()
'''
