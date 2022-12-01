import os
import sys
import time
import glob
import argparse
from PIL import Image
from functools import reduce
from classes.Utils import create_folder

'''
' @author Andrea Corriga
' Lo scopo di questa classe è quello di poter
' unire in una sola immagine diverse immagini relative allo stesso
' segnale (open, close, delta, etc) appartenente a risoluzione diverse
' Dovrebbe essere usata come ultimo step prima di passare all'utilizzo delle
' reti neurali. 
'''

class ImagesMerger:

    # Cartella base da cui leggere le immagini
    __input_path = "../images/"

    # Lo uso per indicare la cartella principale da cui prendere le immagini. In genere è il dataset tipo 'sp500'
    input_folders = []

    # Lista delle varie risoluzioni che voglio unire. Es (1hour, 1day, etc)
    resolutions = []

    # Lista dei vari segnali che voglio prendere (open, close, delta, etc)
    signals = []

    # Matrice contenente le posizioni delle immagini
    positions = []

    # La tipologia di immagine, GADF o GASF
    type = ""

    # Dimensione finale dell'immagine
    img_size = []

    # Dove andrò a salvare le immagini
    output_path = ""

    '''
    ' Valido i parametri passati al metodo run
    ' Mi assicuro che i parametri obbligatori non siano nulli
    ' ed i parametri che devono essere liste vengano effettivamente 
    ' passati come liste
    '''
    def __validate(self, input_folders=None, resolutions=None, signals=None, positions=None, type='gadf', img_size=None, output_path=""):

        # Non può essere nullo
        if input_folders == None:
            sys.exit("ImagesMerger.__validate: input_folders cannot be null")

        # Dev'essere una lista semplice ['', '', '']
        if not isinstance(resolutions, list):
            sys.exit("ImagesMerger.__validate: resolutions must be list")

        # Dev'essere una lista semplice ['', '', '']
        if not isinstance(signals, list):
            sys.exit("ImagesMerger.__validate: signals must be list")

        # Dev'essere una lista di tuple [(0, 0), (1, 1)]
        if not isinstance(positions, list):
            sys.exit("ImagesMerger.__validate: positions must be list")

        if type != 'gadf' and type != 'gasf':
            sys.exit('ImagesMerger.__validate: type must be gadf or gasf')

        # Queste due liste devono avere la stessa lunghezza
        #if len(resolutions) != len(positions):
        #    sys.exit("ImagesMerger.__validate: resolutions and position parameters must have equal lenght. resolutions len: " + str(len(resolutions)) + ", positions len: " + str(len(positions)))

        # img size dev'essere una tupla [20, 20]
        if not isinstance(img_size, list):
            sys.exit("ImagesMerger.__validate: img_size must be list")
        if len(img_size) != 2:
            sys.exit("ImagesMerger.__validate: img_size must be 2 elements list")

        # Non può essere nullo
        if output_path == "":
            sys.exit("ImagesMerger.__validate: output_path cannot be null")

        self.input_folders = input_folders
        self.resolutions = resolutions
        self.signals = signals
        self.positions = positions
        self.type = type
        self.img_size = img_size
        self.output_path = '../images/merge/' + output_path + '/' + type + '/'


    '''
    ' Funzione di appoggio, leggo tutte le
    ' immagini dentro una cartella. La uso per 
    ' prendere i nomi delle immagini che unirò successivamente
    '''
    def __filename_reader(self, file_path):
        filename_list = []

        for file in os.listdir(file_path):
            filename_list.append(file)

        return filename_list

    '''
    ' Siccome questa classe è molto dinamica, 
    ' a seconda di quanti segnali e risoluzioni si scelgono
    ' il numero delle cartelle di output varia. Prima di unire
    ' tutte le immagini, creo preventivamente le cartelle 
    ' per evitare errori nello script. 
    '''
    def __make_output_dir(self):
        # Per ogni segnale
        for signal in self.signals:
            # Se non esiste la cartella, la creo
            if not os.path.isdir(self.output_path + signal):
                os.makedirs(self.output_path + signal)

    '''
    ' Leggo una immagine prendendo come parametro il percorso
    ' esatto 
    '''
    def __get_image(self, full_path):
        img = Image.open(full_path)
        return img

    '''
    ' Unisco le immagini in un unica immagine.
    ' I parametri passati sono fondamentali per capire da dove leggere le immagini originali, quali
    ' segnali utilizzare e come sistemare i vari pezzi del "puzzle" nella nuova immagine. 
    ' Per capire come utilizzare i parametri vedere la doc oppure i commenti alle variabili iniziali 
    '''
    def run(self, input_folders=None, resolutions=None, signals=None, positions=None, type="gadf", img_size=None, output_path=""):
        self.__validate(input_folders=input_folders, resolutions=resolutions, signals=signals, positions=positions, type=type, img_size=img_size, output_path=output_path)
        self.__make_output_dir()

        # https://www.geeksforgeeks.org/python-find-common-elements-in-list-of-lists/
        # Devo trovare solo i file comuni a tutte le risoluzioni, visto che potrebbe
        # esserci un numero diverso di immagini
        list_filenames = []

        # Creo quindi una lista di liste con tutti i filename
        for i, resolution in enumerate(self.resolutions):
            list_filenames.append( self.__filename_reader('../images/' + self.input_folders + '/' + self.resolutions[i] + '/' + self.type + '/' + self.signals[0] + '/') )

        # Genero una lista con solo gli elementi comuni a tutte le liste presenti in list_filenames
        filenames = list(reduce(lambda i, j: i & j, (set(x) for x in list_filenames)))

        # Per ogni file trovato
        for filename in filenames:

            # Per ogni segnale, unisco le immagini di differenti risoluzioni
            for signal in self.signals:

                # Creo una nuova immagine vuota con le dimensioni specificate dal parametro size
                new_img = Image.new('RGB', self.img_size)

                # Da qui l'importanza di avere positions e resolutions della stessa dimensione
                # Uso un solo ciclo per svolgere operazioni su entrambi. Ogni risoluzione
                # Deve avere una posizione specifica nella nuova immagine
                for i in range(len(self.positions)):
                    # La nuova immagine sara' 60x60

                    # Calcolo a runtime il percorso completo di una immagine
                    full_path = '../images/' + self.input_folders + '/' + self.resolutions[i] + '/' + self.type + '/' + signal + '/'

                    # Leggo l'immagine e la incollo in un quadrante specifico della nuova immagine
                    # L'immagine la leggo e passo come parametro inline
                    new_img.paste(self.__get_image(full_path + filename), self.positions[i])

                # Salvo l'immagine
                new_img.save(self.output_path + signal + '/' + filename)

    '''
    ' Unisco le immagini in un unica immagine.
    ' I parametri passati sono fondamentali per capire da dove leggere le immagini originali, quali
    ' segnali utilizzare e come sistemare i vari pezzi del "puzzle" nella nuova immagine. 
    ' Per capire come utilizzare i parametri vedere la doc oppure i commenti alle variabili iniziali 
    '''
    def run_multivariate(self, input_folders=None, resolutions=None, signals=None, positions=None, type="gadf", img_size=None, output_path=""):
        self.__validate(input_folders=input_folders, resolutions=resolutions, signals=signals, positions=positions, type=type, img_size=img_size, output_path=output_path)
        create_folder(self.output_path + 'multivariate/')

        # https://www.geeksforgeeks.org/python-find-common-elements-in-list-of-lists/
        # Devo trovare solo i file comuni a tutte le risoluzioni, visto che potrebbe
        # esserci un numero diverso di immagini
        list_filenames = []

        # Creo quindi una lista di liste con tutti i filename
        for i, resolution in enumerate(self.resolutions):
            list_filenames.append( self.__filename_reader('../images/' + self.input_folders + '/' + self.resolutions[i] + '/' + self.type + '/' + self.signals[0] + '/') )

        # Genero una lista con solo gli elementi comuni a tutte le liste presenti in list_filenames
        filenames = list(reduce(lambda i, j: i & j, (set(x) for x in list_filenames)))

        # Per ogni file trovato
        for filename in filenames:
            # Creo una nuova immagine vuota con le dimensioni specificate dal parametro size
            new_img = Image.new('RGB', self.img_size)

            # Per ogni segnale, unisco le immagini di differenti risoluzioni
            for id_signal, signal in enumerate(self.signals):
                for i in range(len(self.positions[id_signal])):
                

                # Da qui l'importanza di avere positions e resolutions della stessa dimensione
                # Uso un solo ciclo per svolgere operazioni su entrambi. Ogni risoluzione
                # Deve avere una posizione specifica nella nuova immagine
                
                    # La nuova immagine sara' 60x60

                    # Calcolo a runtime il percorso completo di una immagine
                    full_path = '../images/' + self.input_folders + '/' + self.resolutions[i] + '/' + self.type + '/' + signal + '/'

                    # Leggo l'immagine e la incollo in un quadrante specifico della nuova immagine
                    # L'immagine la leggo e passo come parametro inline
                    new_img.paste(self.__get_image(full_path + filename), self.positions[id_signal][i])

                # Salvo l'immagine
                new_img.save(self.output_path + 'multivariate/' + filename)

    '''
    '
    '''
    def run_multivariate_multisignal(self, input_folders=None, resolutions=None, signals=None, positions=None, type="gadf", img_size=None, output_path=""):
        self.__validate(input_folders=input_folders, resolutions=resolutions, signals=signals, positions=positions, type=type, img_size=img_size, output_path=output_path)
        create_folder(self.output_path + 'multivariate/')

        # https://www.geeksforgeeks.org/python-find-common-elements-in-list-of-lists/
        # Devo trovare solo i file comuni a tutte le risoluzioni, visto che potrebbe
        # esserci un numero diverso di immagini
        list_filenames = []

        for i, input_folder in enumerate(input_folders):
            list_filenames.append( self.__filename_reader('../images/' + input_folder + '/' + resolutions[i] + '/' + type + '/' + signals[i] + '/') )

        # Genero una lista con solo gli elementi comuni a tutte le liste presenti in list_filenames
        filenames = list(reduce(lambda i, j: i & j, (set(x) for x in list_filenames)))

        # Per ogni file trovato
        for filename in filenames:
            # Creo una nuova immagine vuota con le dimensioni specificate dal parametro size
            new_img = Image.new('RGB', self.img_size)

            # Per ogni segnale, unisco le immagini di differenti risoluzioni
            for i, signal in enumerate(signals):
                            
                # Calcolo a runtime il percorso completo di una immagine
                full_path = '../images/' + input_folders[i] + '/' + resolutions[i] + '/' + type + '/' + signal + '/'
                # Leggo l'immagine e la incollo in un quadrante specifico della nuova immagine
                # L'immagine la leggo e passo come parametro inline
                new_img.paste(self.__get_image(full_path + filename), positions[i])

            # Salvo l'immagine
            new_img.save(self.output_path + 'multivariate/' + filename)
