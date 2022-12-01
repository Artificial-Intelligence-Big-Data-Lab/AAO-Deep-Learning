import cv2

def similarity(template, img):
    res_1 = cv2.matchTemplate(templ=template, image=img, method=cv2.TM_SQDIFF)
    res_2 = cv2.matchTemplate(templ=template, image=img, method=cv2.TM_SQDIFF_NORMED)
    res_3 = cv2.matchTemplate(templ=template, image=img, method=cv2.TM_CCORR)
    res_4 = cv2.matchTemplate(templ=template, image=img, method=cv2.TM_CCORR_NORMED)
    res_5 = cv2.matchTemplate(templ=template, image=img, method=cv2.TM_CCOEFF)
    res_6 = cv2.matchTemplate(templ=template, image=img, method=cv2.TM_CCOEFF_NORMED)

    
    print("TM_SQDIFF:",  res_1[0][0] )
    print("TM_SQDIFF_NORMED:", (1 - res_2)[0][0] )
    
    print("TM_CCORR:",  res_3[0][0] )
    print("TM_CCORR_NORMED:", (res_4)[0][0] )
    
    print("TM_CCOEFF:",  res_5[0][0] )
    print("TM_CCOEFF_NORMED:",  (res_6)[0][0] )

basepath = 'C:/Users/Utente/Documents/GitHub/PhD-Market-Nets/images/sp500_cet/1day/gadf/delta_current_day/'

template = cv2.imread(basepath + '2000-03-07.png')
image = cv2.imread(basepath + '2000-03-08.png') # simile
image_2 = cv2.imread(basepath + '2000-04-14.png') # meno simile


cat = cv2.imread('C:/Users/Utente/Pictures/gattino.jpg')
cat = cv2.resize(cat, (40, 40))

cat_2 = cv2.imread('C:/Users/Utente/Pictures/gattino.png')
cat_2 = cv2.resize(cat_2, (40, 40))

print("Prima immagine (ad occhio più simile):")
similarity(template=template, img=image)

print("\nSeconda immagine  (ad occhio meno simile):")
similarity(template=template, img=image_2)


print("\nTeza immagine  (stessa img):")
similarity(template=template, img=template)

print("\nQuarta immagine  (gattino a caso):")
similarity(template=template, img=cat)

print("\nGattino vs Gattino:")
similarity(template=cat_2, img=cat)