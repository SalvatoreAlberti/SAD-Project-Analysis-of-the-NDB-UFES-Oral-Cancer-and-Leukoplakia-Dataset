library(scales)
library(cluster)

PROJECT_ROOT <- normalizePath("../../../", winslash = "/")
DATA <- file.path(PROJECT_ROOT, "dataset") 
DATASET_ORIGINALE <- file.path(DATA, "dataset_modificato.csv")

df <- read.csv(DATASET_ORIGINALE, header = TRUE, sep = ",")

# colonna da codificare
col_da_onehot <- "localization"   # metti qui il nome che ti interessa

# mi assicuro che sia factor
df[[col_da_onehot]] <- as.factor(df[[col_da_onehot]])

#  faccio il one-hot 
dummies <- model.matrix(~ get(col_da_onehot) - 1, data = df)

# rinomino le colonne 
colnames(dummies) <- sub("get\\(col_da_onehot\\)", col_da_onehot, colnames(dummies))

#⃣ tolgo la colonna originale dal df
df_senza_col <- df[ , setdiff(names(df), col_da_onehot)]

#⃣ ricompongo il data frame: tutto uguale + colonna one-hot
df_onehot <- cbind(df_senza_col, dummies)
df_onehot
colonne_clustering<-c("larger_size","numero_nuclei","localizationBuccal mucosa", "localizationFloor of mouth","localizationGingiva","localizationLip","localizationPalate","localizationTongue")
df_clustering<-df_onehot[,colonne_clustering]
d<-daisy(df_clustering,metric="gower", weights = c(1,1,0.3,0.3,0.3,0.3,0.3,0.3))
hc_complete <- hclust(d, method = "complete")
plot(hc_complete , cex = 0.6)
rect.hclust(hc_complete, k = 3, border = 2:4)   # evidenzia 3 cluster
cl <- cutree(hc_complete, k = 3)
cl
table(cl)
sil <- silhouette(cl, d)   # calcola silhouette per ogni osservazione
plot(sil)                  # grafico silhouette
mean(sil[, 3])    