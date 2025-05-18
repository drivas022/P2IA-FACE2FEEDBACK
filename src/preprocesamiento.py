import cv2
import numpy as np
import os

def cargar_y_preprocesar_imagen(ruta_img, tamaño=(48, 48)):
    imagen = cv2.imread(ruta_img, cv2.IMREAD_GRAYSCALE)
    imagen = cv2.resize(imagen, tamaño)
    imagen = imagen.astype('float32') / 255.0
    imagen = np.expand_dims(imagen, axis=-1)
    return imagen
