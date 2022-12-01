import os, glob, re
from PIL import Image

def atoi(text):
        return int(text) if text.isdigit() else text

def natural_keys(text):
    return [ atoi(c) for c in re.split(r'(\d+)', text) ]


experiments = [ "001bis - ExpSalva_5_3_15gennaio",
                "002bis - ExpSalva_5_3_15gennaio",
                "004 - pesiloss3 - 16gennaio",
                "005 - pesiloss5 - 16gennaio",
                "006 - 17gennaio",
                "007 - 17gennaio",
                "008 - 18gennaio",
                "009 - 18gennaio - replica exp 004 fullbatch",
                "010 - 21gennaio",
                "012 - prova loss - primo",
                "013 - prova loss - secondo",
                "015 - 28gennaio - ripetizione 012",
                "016 - Class weight pro-short",
                "017 - Class weight pro-short - loss pesante",
            ]

for experiment_name in experiments: 
    #base_path = "C:/Users/Utente/Documents/GitHub/PhD-Market-Nets/experiments/"
    base_path = "/media/unica/HDD 9TB Raid0 - 1/experiments/"

    input_path = base_path + experiment_name + "/accuracy_loss_plots" + "/"
    output_path = base_path + experiment_name + "/thumb_accuracy_loss_plots" + "/"

    folder_list = os.listdir(input_path)
    folder_list.sort(key=natural_keys)

    for folder in folder_list:
        print("Running ", experiment_name, "folder ", folder)
        files = os.listdir(input_path + folder)

        for file in files:     
            image = Image.open(input_path + folder + '/' + file)
            image.thumbnail((250, 150), Image.ANTIALIAS)

            final_output_path = output_path + folder + '/'
            if not os.path.isdir(final_output_path):
                os.makedirs(final_output_path)

            image.save(final_output_path + file, 'png')
