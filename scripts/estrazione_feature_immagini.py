from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Model
from tensorflow.keras.applications.vgg16 import preprocess_input
import numpy as np
from tensorflow.keras.preprocessing import image
import os

from pathlib import Path
import pandas as pd

PROJECT_ROOT=Path(__file__).resolve().parents[2]
DATA=PROJECT_ROOT/"dataset"/"NDB-UFES An oral cancer and leukoplakia dataset composed of histopathological images and patient data"
PERCORSO_IMMAGINI=DATA/"images"

# Carica il modello senza il top (ultimo strato di classificazione)
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))



path = PERCORSO_IMMAGINI
img_list = []

for filename in os.listdir(path):
    if filename.endswith(".jpg") or filename.endswith(".png"):
        #Carichiamo l'immagine con la dimensione 224x224 in img
        img = image.load_img(os.path.join(path, filename), target_size=(224, 224))
        #Trasforma l'img in array
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = preprocess_input(img_array)
        img_list.append(img_array)
        features = base_model.predict(img_array)
        print(features.shape)

        features_flat = features.reshape(features.shape[0], -1)
        print(features_flat.shape)

# Converte la lista in array numpy
img_array = np.vstack(img_list)


