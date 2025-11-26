from pathlib import Path
import pandas as pd
from keras.applications import VGG16
from keras.models import Model
from keras.applications.vgg16 import preprocess_input
from keras.preprocessing import image
import numpy as np
import os
from sklearn.decomposition import PCA

# Percorsi
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA = PROJECT_ROOT / "dataset" / "NDB-UFES An oral cancer and leukoplakia dataset composed of histopathological images and patient data"
DATASET_ORIGINALE = DATA / "ndb-ufes.csv"
DATASET_MODIFICATO = PROJECT_ROOT / "dataset" / "dataset_modificato.csv"
PERCORSO_IMMAGINI = DATA / "images"

def main():

   # Leggi dataset originale
    df = pd.read_csv(DATASET_ORIGINALE)

    # Controlla che ci sia la colonna 'path'
    if "path" not in df.columns:
        print("Errore: il DataFrame deve contenere una colonna 'path' con i nomi dei file immagine.")
        return

    # Carica il modello VGG16 senza top
    base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

    features_all = []
    filenames = []

    # Cicla su tutte le immagini
    for filename in os.listdir(PERCORSO_IMMAGINI):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            img_path = os.path.join(PERCORSO_IMMAGINI, filename)
            img = image.load_img(img_path, target_size=(224, 224))
            img_array = image.img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0)
            img_array = preprocess_input(img_array)

            # Estrai feature
            features = base_model.predict(img_array)
            features_flat = features.reshape(features.shape[0], -1)  # Flatten
            features_all.append(features_flat[0])
            filenames.append(filename)

    # Converti in array NumPy
    features_all = np.array(features_all)

    # Applica PCA per ridurre a 100 componenti
    pca = PCA(n_components=100)
    features_reduced = pca.fit_transform(features_all)

    # Crea DataFrame delle feature ridotte
    df_features = pd.DataFrame(features_reduced, columns=[f"feat_{i}" for i in range(100)])
    df_features["path"] = filenames

    # Unisci le feature al dataset originale tramite la colonna 'path'
    df_modificato = df.merge(df_features, on="path", how="left")

    # Salva il dataset modificato
    df_modificato.to_csv(DATASET_MODIFICATO, index=False)
    print(f"Dataset modificato salvato in: {DATASET_MODIFICATO}")

if __name__ == "__main__":
    main()
