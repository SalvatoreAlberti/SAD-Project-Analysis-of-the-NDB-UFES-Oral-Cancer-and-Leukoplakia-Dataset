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

CARTELLA_PATCH = DATASET_CARTELLA/"patches"                    # cartella dove salveremo le patch

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

    # Scorro l'immagine con una griglia PATCH_SIZE x PATCH_SIZE
    for y in range(0, h - PATCH_SIZE + 1, STRIDE):
        for x in range(0, w - PATCH_SIZE + 1, STRIDE):
            patch = img[y:y+PATCH_SIZE, x:x+PATCH_SIZE]

            # QUI NON SCARTIAMO NIENTE: ogni patch viene salvata
            patch_name = f"{base_name}_y{y}_x{x}.png"
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
df_patches_min.to_csv(os.path.join(DATASET_CARTELLA,"patches.csv"), index=False)
print("Salvato patch_index_minimo.csv con forma:", df_patches_min.shape)
