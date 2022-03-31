import pydicom
import numpy as np
import os, sys
import matplotlib.pyplot as plt
import pandas as pd 
from tqdm import tqdm
from pydicom import dcmread
import cv2
#%%
# Función que permite normalizar las imágnes entre 0 y 1
def normalize(im):
    im_out = np.copy((im)).astype(np.float32)
    im_out = im_out - im_out.min()
    im_out = im_out / im_out.max()
    im_out[np.isnan(im_out)] = 0
    return im_out

# Genera un Tensor con el set de imágenes TC. Retorna un Array con el tensor de imágenes normalizdas, sus clases y los respectivos nombres de la imagen 
# El parametro de entrada (url) es la ubicación donde estan contenidas las imágenes
# n_clases = cantidad de carpetas a evaluar
# Downsamplig = 0 -> NO Downsamplig
# Downsamplig = 1 -> Imagen = 256x256
# Downsamplid = 2 -> Imagen = 128x128
def dataset(url, n_clases, downsampling):
    os.chdir(url)
    url_1 = os.getcwd()
    clases = os.listdir() #se obtienen las nombres de las carpetas y/o clases 
    clases.sort() #se reordenan de forma alfabetica
   # n_clases = len(clases) #se obtiene el numero de clases
    Set_img = [] #array de imágenes
    Set_clases = [] #array de clases
    Set_name_images =[] #array de nombre de las imágenes

    for h in tqdm (range(n_clases)):
        if clases[h] != '.DS_Store':
            url_2 = url_1 + '/' + clases[h] #se define la ruta por cada clase
            os.chdir(str(url_2))
            image_names = os.listdir()
            image_names.sort() 
            n_image_names = len(image_names)
            for i in range(n_image_names):
                if image_names[i] != '.DS_Store':
                    url_3 = url_2 + '/' + image_names[i] #se define la ruta de cada imagén
                    im = dcmread(url_3) #leer imagen en formato .dcm
                    im = im.pixel_array #convertir imagen .dcm a Array
                    im = normalize(im) #se normaliza la imagen
                    if downsampling == 1:
                      im = cv2.pyrDown(im) #Downsamplig de 512x512 -> 256x256
                    if downsampling == 2:
                      im = cv2.pyrDown(im) #Downsamplig de 512x512 -> 256x256
                      im = cv2.pyrDown(im) #Downsamplig de 256x256 -> 128x128
                    Set_img.append(im) #se concatenan las imagnes y etiquetas
                    Set_clases.append(clases[h])
                    Set_name_images.append(image_names[i])

    Set_img = np.array(Set_img) #se convierten los arreglos en array
    Set_clases = np.array(Set_clases) 
    Set_name_images = np.array(Set_name_images)
    return Set_img, Set_clases, Set_name_images
# %%
"""url = '/Users/stevenmartine/Documents/Datasets/Breast_Katherym/Computed tomography/Selection'
Dataset_img, Data_clases, Data_names = Dataset_TC(url)
#%%
n_img = 2750
plt.imshow(Dataset_img[n_img,:,:], cmap='gray')
print(Data_clases[n_img])
print(Data_names[n_img])

# %%
Dataset_recorte = Dataset_img[:,120:450,:]
plt.imshow(Dataset_recorte[n_img,:,:], cmap='gray')"""

# %%
