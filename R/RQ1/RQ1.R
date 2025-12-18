library(scales)
library(cluster)
PROJECT_ROOT <- normalizePath("../../../", winslash = "/")
DATA <- file.path(PROJECT_ROOT, "dataset") 
DATASET_ORIGINALE <- file.path(DATA, "dataset_modificato.csv")

df <- read.csv(DATASET_ORIGINALE, header = TRUE, sep = ";")

# colonna da codificare
#col_da_onehot <- "localization"   # metti qui il nome che ti interessa

# mi assicuro che sia factor
#df[[col_da_onehot]] <- as.factor(df[[col_da_onehot]])

#  faccio il one-hot 
#dummies <- model.matrix(~ get(col_da_onehot) - 1, data = df)

# rinomino le colonne 
#colnames(dummies) <- sub("get\\(col_da_onehot\\)", col_da_onehot, colnames(dummies))

#⃣ tolgo la colonna originale dal df
#df_senza_col <- df[ , setdiff(names(df), col_da_onehot)]

#⃣ ricompongo il data frame: tutto uguale + colonna one-hot
#df_onehot <- cbind(df_senza_col, dummies)
#df_onehot
colonne_clustering<-c("larger_size","numero_nuclei")
df_clustering<-df[,colonne_clustering]
df_clustering$larger_size   <- as.numeric(df_clustering$larger_size)
df_clustering$numero_nuclei <- as.numeric(df_clustering$numero_nuclei)
#df_clustering$localization<-as.factor(df_clustering$localization)
colonne_scalate<-scale(df_clustering)
d<-daisy(colonne_scalate,metric="euclidean")
d
hc_complete <- hclust(d, method = "ward.D2")
plot(hc_complete , cex = 0.6)
rect.hclust(hc_complete, k = 2, border = 2:4)   # evidenzia 3 cluster
cl <- cutree(hc_complete, k = 2)


# attacco il cluster alle righe (include anche localization)
df_out <- data.frame(
  larger_size   = df_clustering$larger_size,
  numero_nuclei = df_clustering$numero_nuclei,
  cluster       = cl
)

# quante righe per cluster
print(table(df_out$cluster))

# stampa le righe di ogni cluster (con localization)
for (k in sort(unique(df_out$cluster))) {
  cat("\n====================\n")
  cat("CLUSTER", k, "- n =", sum(df_out$cluster == k), "\n")
  print(df_out[df_out$cluster == k, c("larger_size","numero_nuclei")])
}


cl
table(cl)
sil <- silhouette(cl, d)   # calcola silhouette per ogni osservazione
plot(sil)                  # grafico silhouette
mean(sil[, 3])    