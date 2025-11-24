from pathlib import Path
import pandas as pd

PROJECT_ROOT=Path(__file__).resolve().parents[2]
DATA=PROJECT_ROOT/"dataset"/"NDB-UFES An oral cancer and leukoplakia dataset composed of histopathological images and patient data"
DATASET_ORIGINALE=DATA/"ndb-ufes.csv"
DATASET_MODIFICATO=PROJECT_ROOT/"dataset"/"dataset_modificato.csv"

def main():
    df=pd.read_csv(DATASET_ORIGINALE)

    #modifiche...
    

    df.to_csv(DATASET_MODIFICATO)

if(__name__)=="__main__":
    main()