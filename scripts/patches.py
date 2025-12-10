import os
import cv2
import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA = PROJECT_ROOT / "dataset" / "NDB-UFES An oral cancer and leukoplakia dataset composed of histopathological images and patient data"
DATASET_ORIGINALE = DATA / "ndb-ufes.csv"
DATASET_MODIFICATO = PROJECT_ROOT / "dataset" / "dataset_modificato.csv"
PERCORSO_IMMAGINI = DATA / "images"
DATASET_CARTELLA= PROJECT_ROOT/"dataset"

CARTELLA_PATCH = DATASET_CARTELLA/"patches4"                    # cartella dove salveremo le patch

PATCH_SIZE = 256   # lato patch (in pixel), cambia se vuoi (es. 512)
STRIDE = 256       # passo griglia; = PATCH_SIZE → patch non sovrapposte

os.makedirs(CARTELLA_PATCH, exist_ok=True)

# === CARICO IL CSV DELLE IMMAGINI ORIGINALI ===

df_img = pd.read_csv(DATASET_ORIGINALE)

if "path" not in df_img.columns:
    raise ValueError(f"La colonna 'path' non è presente in {DATASET_ORIGINALE}")

# === LISTA PER LE RIGHE DEL NUOVO CSV (MINIMALE) ===

patch_rows = []
patch_counter = 0

# === LOOP SU TUTTE LE IMMAGINI ORIGINALI ===

for idx, row in df_img.iterrows():
    rel_path = row["path"]          # es. "img_001.png"
    orig_image_path = os.path.join(PERCORSO_IMMAGINI, rel_path)

    if not os.path.exists(orig_image_path):
        print(f"[ATTENZIONE] File non trovato: {orig_image_path}")
        continue

    img = cv2.imread(orig_image_path)
    if img is None:
        print(f"[ATTENZIONE] Impossibile leggere l'immagine: {orig_image_path}")
        continue

    h, w, _ = img.shape
    print(f"Elaboro {orig_image_path} - dimensioni: {w}x{h}")

    base_name = os.path.splitext(os.path.basename(orig_image_path))[0]

    # Calcolo metà altezza e metà larghezza
    h_mid = h // 2
    w_mid = w // 2

    # Definisco i 4 quadranti (y1, y2, x1, x2, etichetta)
    quadrants = [
        (0,      h_mid, 0,      w_mid,  "top_left"),
        (0,      h_mid, w_mid,  w,      "top_right"),
        (h_mid,  h,     0,      w_mid,  "bottom_left"),
        (h_mid,  h,     w_mid,  w,      "bottom_right"),
    ]

    for (y1, y2, x1, x2, label) in quadrants:
        patch = img[y1:y2, x1:x2]

        patch_name = f"{base_name}_{label}.png"
        patch_path = os.path.join(CARTELLA_PATCH, patch_name)

        cv2.imwrite(patch_path, patch)
        patch_counter += 1

        patch_rows.append({
            "patch_path": patch_path,
            "orig_image_path": orig_image_path
        })


print(f"Numero totale di patch salvate: {patch_counter}")

# === SALVO IL CSV MINIMALE ===

df_patches_min = pd.DataFrame(patch_rows)
df_patches_min.to_csv(os.path.join(DATASET_CARTELLA,"patches_4.csv"), index=False)
print("Salvato patch_index_minimo.csv con forma:", df_patches_min.shape)
