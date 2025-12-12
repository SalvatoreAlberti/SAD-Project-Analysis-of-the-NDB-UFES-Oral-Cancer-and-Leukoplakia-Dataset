from pathlib import Path
import pandas as pd

# Percorsi
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA = PROJECT_ROOT / "dataset" / "NDB-UFES An oral cancer and leukoplakia dataset composed of histopathological images and patient data"
DATASET_MODIFICATO = PROJECT_ROOT / "dataset" / "dataset_modificato.csv"

def main():
    # 1) Leggi dataset
    df = pd.read_csv(DATASET_MODIFICATO)

    # 2) Fix age_group (non svuotare se è già testo)
    mapping = {0: "Young", 1: "Middle", 2: "Elderly"}
    age_num = pd.to_numeric(df["age_group"], errors="coerce")
    df["age_group"] = age_num.map(mapping).fillna(df["age_group"])

    # 3) Tipi corretti
    df = df.reset_index(drop=True)
    df["patient_id"] = df["patient_id"].astype(str).str.strip()
    df["numero_nuclei"] = pd.to_numeric(df["numero_nuclei"], errors="raise")

    # 4) Rimuovi duplicati consecutivi: tieni riga con numero_nuclei max per blocco
    keep_idx = []
    i = 0

    while i < len(df):
        pid = df.loc[i, "patient_id"]

        j = i + 1
        while j < len(df) and df.loc[j, "patient_id"] == pid:
            j += 1

        block = df.iloc[i:j]
        best_idx = block["numero_nuclei"].idxmax()  # se pari merito tiene la prima

        keep_idx.append(best_idx)
        i = j

    df = df.loc[keep_idx].sort_index().reset_index(drop=True)

    # 5) Salva
    df.to_csv(DATASET_MODIFICATO, index=False)
    print(f"Dataset modificato salvato in: {DATASET_MODIFICATO}")

if __name__ == "__main__":
    main()
