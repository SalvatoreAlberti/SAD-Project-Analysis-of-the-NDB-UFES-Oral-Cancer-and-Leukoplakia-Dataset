library(tidyverse)


PROJECT_ROOT <- normalizePath("../../../", winslash = "/")
# Percorso cartella dataset
DATA <- file.path(PROJECT_ROOT,
                  "dataset") 


# Percorso completo del CSV
DATASET_ORIGINALE <- file.path(DATA, "dataset_modificato.csv")

X <- as.matrix(df[, pc_cols])
dim(X)   # n_immagini x 90
# controlla che siano pc1, pc2, ...
df_pc <- df %>%
  select(all_of(pc_cols)) %>%
  mutate(across(everything(), ~ as.numeric(.)))
pc_classes <- sapply(df[, pc_cols], class)
pc_classes[pc_classes == "character"]

wss <- c()
k_range <- 2:10

for (k in k_range) {
  km <- kmeans(X, centers = k, nstart = 25)
  wss <- c(wss, km$tot.withinss)
}

plot(k_range, wss, type = "b",
     xlab = "Numero di cluster k",
     ylab = "Tot within-cluster sum of squares",
     main = "Metodo del gomito (PCA features)")


