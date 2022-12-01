# VggHandler()

Questa classe serve per poter utilizzare con facilità le reti neurali predisposte all'interno del Framework. 
Per poter utilizzare questa classe, è necessario aver utilizzato in precedenza la classe `Gaf` e `ImagesMerger` per generare correttamente le immagini da usare come samples per effettuare il training del modello.

Nella sezione esempi viene fornito un tutorial unico su come generare le immagini ed infine utilizzare la rete neurale.  

Costruttore della classe vuoto. 

**Nota importante**

E' possibile, se si utilizza una workstation con più GPU, selezionare la GPU da utilizzare per fare il training
Per fare ciò, basterà inserire nell'header del file che si vuole eseguire questo snippet: 

```python
import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID";
os.environ["CUDA_VISIBLE_DEVICES"]="1";
```


# net_config

Imposta gli iperparametri della rete e permette di personalizzare il salvataggio del modello. 
Il parametro `save_model_history` se settato a `true` permette di salvare ogni tot epoche (specificato col parametro `save_model_history`) un modello intermedio.

**Parametri**:

- **epochs**: _int_
- **number_of_nets**: _int_ 
- **init_lr**: _float_
- **bs**: _int_
- **save_pkl**: _boolean_ 
- **save_model_history**: _boolean_ 
- **model_history_period**: _int_


**Esempio**: 

blabla

```python
import os 
from classes.VggHandler import VggHandler

vgg = VggHandler()

vgg.net_config(epochs=200, number_of_nets=20, save_pkl=False, save_model_history=True, model_history_period=40)
```

# run_initialize()

**Parametri**:

- **predictions_dataset**: _string_
- **predictions_images_folder**: _string_ 
- **input_images_folders**: _list_ 
- **input_datasets**: _list_ 
- **training_set**: _list_
- **validation_set**: _list_
- **test_set**: _list_ 
- **input_shape**: _triple_ `(40, 40, 3)`
- **output_folder**: _string_

```python
import os 
from classes.VggHandler import VggHandler

vgg = VggHandler()

vgg.net_config(epochs=200, number_of_nets=20, save_pkl=False, save_model_history=True, model_history_period=40)

training_set = [['2000-02-01', '2013-12-31']]
validation_set = [ ['2014-01-01', '2014-12-31']]
test_set = [['2015-01-01', '2016-05-31']]

input_images_folders = ['merge_sp500/gadf/delta/']
input_datasets = ['sp500']

predictions_dataset = 'sp500'
predictions_images_folder = 'merge_sp500/gadf/delta/'

vgg.run_initialize( predictions_dataset=predictions_dataset,
                    predictions_images_folder=predictions_images_folder,

                    input_images_folders=input_images_folders,
                    input_datasets=input_datasets,

                    training_set=training_set,
                    validation_set=validation_set,
                    test_set=test_set,
                    
                    input_shape=(40,40,3),
                    output_folder='multi_companies_big_net_1')

```

# run()

```python
import os 
from classes.VggHandler import VggHandler

vgg = VggHandler()

vgg.net_config(epochs=200, number_of_nets=20, save_pkl=False, save_model_history=True, model_history_period=40)

input_images_folders = ['merge_sp500/gadf/delta/' ]
input_datasets = ['sp500']

predictions_dataset = 'sp500'
predictions_images_folder = 'merge_sp500/gadf/delta/'

vgg.run_initialize( predictions_dataset=predictions_dataset,
                    predictions_images_folder=predictions_images_folder,

                    input_images_folders=input_images_folders,
                    input_datasets=input_datasets,

                    training_set=training_set,
                    validation_set=validation_set,
                    test_set=test_set,
                    
                    input_shape=(40,40,3),
                    output_folder='multi_companies_big_net_1')

vgg.run()

```

un_small()
```

# run_again()

# get_predictions()

```python
vgg.get_predictions(set_type='validation')
vgg.get_predictions(set_type='test')
```

# run_again()

**Parametri**:

- **model_input_folder**: _string_


# get_predictions_foreach_epoch()

```python
vgg.get_predictions(set_type='validation')
vgg.get_predictions(set_type='test')
```