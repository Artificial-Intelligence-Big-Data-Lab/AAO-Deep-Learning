from classes.Market import Market
from PIL import Image
from tqdm import tqdm

#img_positive = Image.open(r'C:/Users/Utente/Documents/GitHub/AIBD-new-event-detection/lexicons-to-img/img/for_cnn_binary_test/positive_1_opacity.png')

#img_negative = Image.open(r'C:/Users/Utente/Documents/GitHub/AIBD-new-event-detection/lexicons-to-img/img/for_cnn_binary_test/negative_0.9_opacity.png')
#img_negative = Image.open(r'C:/Users/Utente/Documents/GitHub/AIBD-new-event-detection/lexicons-to-img/img/for_cnn_binary_test/negative_0.7_opacity.png')
#img_negative = Image.open(r'C:/Users/Utente/Documents/GitHub/AIBD-new-event-detection/lexicons-to-img/img/for_cnn_binary_test/negative_0.5_opacity.png')
#img_negative = Image.open(r'C:/Users/Utente/Documents/GitHub/AIBD-new-event-detection/lexicons-to-img/img/for_cnn_binary_test/negative_0.1_opacity.png')
#img_negative = Image.open(r'C:/Users/Utente/Documents/GitHub/AIBD-new-event-detection/lexicons-to-img/img/for_cnn_binary_test/negative_0_opacity.png')


#img_positive = Image.open(r'C:/Users/Utente/Documents/GitHub/AIBD-new-event-detection/lexicons-to-img/img/for_cnn_binary_test/positive.png')
#img_negative = Image.open(r'C:/Users/Utente/Documents/GitHub/AIBD-new-event-detection/lexicons-to-img/img/for_cnn_binary_test/negative.png')

img_positive = Image.open(r'C:/Users/Utente/Documents/GitHub/AIBD-new-event-detection/lexicons-to-img/img/for_cnn_binary_test/positive_2_common.png')
img_negative = Image.open(r'C:/Users/Utente/Documents/GitHub/AIBD-new-event-detection/lexicons-to-img/img/for_cnn_binary_test/negative_2_common.png')

sp500 = Market(dataset='sp500_cet')

sp500 = sp500.get_binary_labels(freq='1d', thr=-0.5)

sp500 = sp500.reset_index() 

sp500['date_time'] = sp500['date_time'].astype(str)

for index, row in tqdm(sp500.iterrows(), total=sp500.shape[0]):

    

    if row['label_next_day'] == 1: 
        #print(row['date_time'], '-', 1)
        img_positive.save(r'C:/Users/Utente/Documents/GitHub/AIBD-new-event-detection/lexicons-to-img/img/for_cnn_binary_test/dataset_3_dots_1_common/' + row['date_time'] + '.png')
    
    if row['label_next_day'] == 0: 
        #print(row['date_time'], '-', 0)
        img_negative.save(r'C:/Users/Utente/Documents/GitHub/AIBD-new-event-detection/lexicons-to-img/img/for_cnn_binary_test/dataset_3_dots_1_common/' + row['date_time'] + '.png')
