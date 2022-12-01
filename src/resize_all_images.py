from PIL import Image
import os, sys
from classes.Utils import create_folder
from tqdm import tqdm

#input_folder = "C:/Users/Andrea/Pictures/lexicons/for_cnn_model_filter_500_500_grayscale/"
#output_folder = "C:/Users/Andrea/Pictures/lexicons/for_cnn_model_filter_256_256_grayscale/"

input_folder = "C:/Users/Andrea/Pictures/lexicons/for_cnn_model_filter_500_500_opacity/"
output_folder = "C:/Users/Andrea/Pictures/lexicons/for_cnn_model_filter_256_256_opacity/"

create_folder(output_folder)
dirs = os.listdir(input_folder)

def resize():
    for item in tqdm(dirs):
        if os.path.isfile(input_folder + item):
            im = Image.open(input_folder + item)
            f, e = os.path.splitext(input_folder + item)
            imResize = im.resize((256,256), Image.ANTIALIAS)
            imResize.save(output_folder + item)

resize()