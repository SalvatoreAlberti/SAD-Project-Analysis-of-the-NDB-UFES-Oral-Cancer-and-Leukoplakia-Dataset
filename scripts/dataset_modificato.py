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



    # Salva il dataset modificato
    df.to_csv(DATASET_MODIFICATO, index=False)
    print(f"Dataset modificato salvato in: {DATASET_MODIFICATO}")

if __name__ == "__main__":
    main()
