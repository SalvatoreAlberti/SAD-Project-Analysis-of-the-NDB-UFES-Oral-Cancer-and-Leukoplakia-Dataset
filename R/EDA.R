library(here)
# Risalgo di un livello dalla cartella R/
PROJECT_ROOT <- here::here("..")

# Percorso della cartella dataset
DATA <- file.path(PROJECT_ROOT, 
                  "dataset",
                  "NDB-UFES An oral cancer and leukoplakia dataset composed of histopathological images and patient data")

# Percorso completo del CSV
DATASET_ORIGINALE <- file.path(DATA, "ndb-ufes.csv")

# Caricamento dataset
df <- read.csv(DATASET_ORIGINALE, header = TRUE, sep = ",")

