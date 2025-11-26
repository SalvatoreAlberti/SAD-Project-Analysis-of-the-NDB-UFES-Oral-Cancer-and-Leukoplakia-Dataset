from pathlib import Path
import pandas as pd
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Model
from tensorflow.keras.applications.vgg16 import preprocess_input
import numpy as np
from tensorflow.keras.preprocessing import image
import os

PROJECT_ROOT=Path(__file__).resolve().parents[2]
DATA=PROJECT_ROOT/"dataset"/"NDB-UFES An oral cancer and leukoplakia dataset composed of histopathological images and patient data"
DATASET_ORIGINALE=DATA/"ndb-ufes.csv"
DATASET_MODIFICATO=PROJECT_ROOT/"dataset"/"dataset_modificato.csv"
PERCORSO_IMMAGINI=DATA/"images"

def main():
    df=pd.read_csv(DATASET_ORIGINALE)

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
            df.at[i, "feature_Immagine"] = features_flat[0]  # i = indice della riga corrispondente all’immagine

    
    df.to_csv(DATASET_MODIFICATO)
if(__name__)=="__main__":
    main()