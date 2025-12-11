import os
import csv
from pathlib import Path
from inference_sdk import InferenceHTTPClient

# ============================
# CONFIGURAZIONI
# ============================

API_URL = "https://serverless.roboflow.com"
API_KEY = "fu4TuA4WUwHtEumvOagj"
WORKSPACE = "salvatore-zfksp"
WORKFLOW_ID = "custom-workflow"

BASE_DIR = Path(__file__).resolve().parent.parent  # es: Progetto_SAD/

DATASET_DIR = BASE_DIR.parent / "dataset"

IMAGES_FOLDER = DATASET_DIR / "immagini_trasformate"
CSV_ORIGINALE = DATASET_DIR / "NDB-UFES An oral cancer and leukoplakia dataset composed of histopathological images and patient data/ndb-ufes.csv"
CSV_MODIFICATO = BASE_DIR.parent / "dataset_modificato.csv"

# ============================
# 1. ESEGUI IL MODELLO SU TUTTE LE IMMAGINI
# ============================

client = InferenceHTTPClient(
    api_url=API_URL,
    api_key=API_KEY
)

# Elenco dei file immagine nella cartella
image_files = [
    f for f in os.listdir(IMAGES_FOLDER)
    if f.lower().endswith((".jpg", ".jpeg", ".png"))
]

# Dizionario: nome file → numero nuclei
risultati_modello = {}

for img_name in image_files:
    img_path = os.path.join(IMAGES_FOLDER, img_name)

    # Esegui workflow su una singola immagine
    result = client.run_workflow(
        workspace_name=WORKSPACE,
        workflow_id=WORKFLOW_ID,
        images={"image": img_path},
        use_cache=False
    )

    # Estrazione predizioni dal formato di output del workflow
    predictions = []
    if isinstance(result, list) and len(result) > 0:
        if "predictions" in result[0]:
            if "predictions" in result[0]["predictions"]:
                predictions = result[0]["predictions"]["predictions"]

    # Conta nuclei
    count = len(predictions)
    risultati_modello[img_name] = count

    print(f"{img_name}: {count} nuclei trovati")

# ============================
# 2. CREA IL NUOVO CSV MODIFICATO
# ============================

with open(CSV_ORIGINALE, newline="", encoding="utf-8") as infile, \
     open(CSV_MODIFICATO, "w", newline="", encoding="utf-8") as outfile:

    reader = csv.DictReader(infile)

    # Aggiungi la nuova colonna
    fieldnames = reader.fieldnames + ["numero_nuclei"]
    writer = csv.DictWriter(outfile, fieldnames=fieldnames)

    writer.writeheader()

    for row in reader:
        # Estrai SOLO il nome file dalla colonna path
        nome_file = os.path.basename(row["path"])

        # Trova il numero di nuclei corrispondente
        row["numero_nuclei"] = risultati_modello.get(nome_file, None)

        writer.writerow(row)

print("✔️ Nuovo file creato:", CSV_MODIFICATO)
