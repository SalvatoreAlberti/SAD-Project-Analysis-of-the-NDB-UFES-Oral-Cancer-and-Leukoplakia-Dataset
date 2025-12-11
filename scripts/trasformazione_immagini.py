from skimage import io, color
from pathlib import Path
import pandas as pd
import cv2
import os

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA = PROJECT_ROOT / "dataset" / "NDB-UFES An oral cancer and leukoplakia dataset composed of histopathological images and patient data"
DATASET_ORIGINALE = DATA / "ndb-ufes.csv"
PERCORSO_IMMAGINI = DATA / "images"
IMMAGINI_TRASFORMATE = PROJECT_ROOT / "dataset" / "immagini_trasformate"
os.makedirs(IMMAGINI_TRASFORMATE, exist_ok=True)

# 1️⃣ Leggo il dataset originale
df = pd.read_csv(DATASET_ORIGINALE)

# 2️⃣ Controllo che esista la colonna 'path'
if "path" not in df.columns:
    raise ValueError("Nel dataset originale manca la colonna 'path' con il nome del file immagine.")

for i, row in df.iterrows():
    # ⚠️ PRIMO ERRORE: qui devi usare la colonna 'path', non df[row]
    # Se nel CSV hai qualcosa tipo "images/xxx.png" o solo "xxx.png":
    relative_path = row["path"]              # stringa dal CSV
    img_path = PERCORSO_IMMAGINI / relative_path

    # Controllo opzionale che il file esista
    if not img_path.is_file():
        print(f"[SKIP] File non trovato: {img_path}")
        continue

    img = io.imread(str(img_path))
    hed = color.rgb2hed(img)

    hematoxylin = hed[:, :, 0]

    # Inversione per rendere i nuclei scuri
    hematoxylin_inv = -hematoxylin

    # Normalizzazione per salvarlo come immagine 8-bit
    hematoxylin_norm = cv2.normalize(
        hematoxylin_inv,
        None,
        alpha=0, beta=255,
        norm_type=cv2.NORM_MINMAX,
        dtype=cv2.CV_8U
    )

    # ⚠️ SECONDO ERRORE: cv2.imwrite vuole un *file*, non una cartella
    # Creo un nome file di output, ad esempio aggiungendo il suffisso '_hematoxylin'
    original_name = Path(relative_path).name           # es: "immagine1.png"
    stem = Path(original_name).stem                    # "immagine1"
    out_name = f"{stem}.png"               # "immagine1_hematoxylin.png"
    out_path = IMMAGINI_TRASFORMATE / out_name

    cv2.imwrite(str(out_path), hematoxylin_norm)
    print(f"[OK] Salvata: {out_path}")
