from pathlib import Path
import pandas as pd
from keras.applications import VGG16
from keras.models import Model
from keras.applications.vgg16 import preprocess_input
from keras.preprocessing import image
import numpy as np
import os
from sklearn.decomposition import PCA
from estrazione_feature_immagini import calcola_feature_pca_da_patch

# Percorsi
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA = PROJECT_ROOT / "dataset" / "NDB-UFES An oral cancer and leukoplakia dataset composed of histopathological images and patient data"
DATASET_ORIGINALE = DATA / "ndb-ufes.csv"
DATASET_MODIFICATO = PROJECT_ROOT / "dataset" / "dataset_modificato.csv"
PERCORSO_IMMAGINI = DATA / "images"
PATCH_INDEX= PROJECT_ROOT/"dataset"/"patches.csv"
def main():

   # Leggi dataset originale
    df = pd.read_csv(DATASET_ORIGINALE)
    age_numeric = pd.to_numeric(df["age_group"], errors="coerce")

# === 4. Definisco la mappatura e sostituisco direttamente age_group ===
    mapping = {
    0: "Young",
    1: "Middle",
    2: "Elderly"
}

    df["age_group"] = age_numeric.map(mapping)

# === 5. Controllo eventuali valori non mappati ===
    valori_non_mappati = df.loc[df["age_group"].isna(), "age_group"]

    if "path" not in df.columns:
        raise ValueError("Nel dataset originale deve esserci una colonna 'path' con il path dell'immagine.")

    df["orig_image_path"] = (PERCORSO_IMMAGINI / df["path"]).astype(str)

    # 4) CALCOLA LE FEATURE PCA A PARTIRE DALLE PATCH
    df_pca = calcola_feature_pca_da_patch(
        patch_index_path=PATCH_INDEX,
        n_components=90,   # puoi cambiare questo numero se vuoi
    )

    # 5) MERGE: aggiungi le colonne pc1..pcN al dataset
    df_finale = df.merge(df_pca, on="orig_image_path", how="left")

    # 6) Salva il dataset modificato (ora con age_group sistemata + feature PCA)
    df_finale.to_csv(DATASET_MODIFICATO, index=False)
    print(f"Dataset modificato con feature PCA salvato in: {DATASET_MODIFICATO}")




    

if __name__ == "__main__":
    main()
