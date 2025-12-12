


PROJECT_ROOT <- normalizePath("../../../", winslash = "/")
DATA <- file.path(PROJECT_ROOT, "dataset") 
DATASET_ORIGINALE <- file.path(DATA, "dataset_modificato.csv")

df <- read.csv(DATASET_ORIGINALE, header = TRUE, sep = ",")

# 1️⃣ scelgo la colonna da codificare
col_da_onehot <- "localization"   # metti qui il nome che ti interessa

# (opzionale ma consigliato) mi assicuro che sia factor
df[[col_da_onehot]] <- as.factor(df[[col_da_onehot]])

# 2️⃣ faccio il one-hot SOLO su quella colonna
dummies <- model.matrix(~ get(col_da_onehot) - 1, data = df)

# rinomino le colonne in modo carino (togliendo "get(col_da_onehot)" dal nome)
colnames(dummies) <- sub("get\\(col_da_onehot\\)", col_da_onehot, colnames(dummies))

# 3️⃣ tolgo la colonna originale dal df
df_senza_col <- df[ , setdiff(names(df), col_da_onehot)]

# 4️⃣ ricompongo il data frame: tutto uguale + colonne one-hot
df_onehot <- cbind(df_senza_col, dummies)
df_onehot
