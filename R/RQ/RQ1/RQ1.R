# RQ1 - Clustering gerarchico: single linkage + distanza Jaccard

library(cluster)
library(scales)

# Caricamento dati
PROJECT_ROOT <- normalizePath("../../../../", winslash = "/")
DATA <- file.path(PROJECT_ROOT, "dataset") 
DATASET_ORIGINALE <- file.path(DATA, "dataset_modificato.csv")

df <- read.csv(DATASET_ORIGINALE, header = TRUE, sep = ";")

# Selezione variabili morfologiche
colonne_clustering <- c("larger_size", "numero_nuclei")
df_clustering <- df[, colonne_clustering]

df_clustering$larger_size   <- as.numeric(df_clustering$larger_size)
df_clustering$numero_nuclei <- as.numeric(df_clustering$numero_nuclei)

# Rimozione NA
complete_idx <- complete.cases(df_clustering)
df_clustering <- df_clustering[complete_idx, ]
df_complete   <- df[complete_idx, ]

# BINARIZZAZIONE (necessaria per Jaccard)
df_binary <- data.frame(
  larger_size   = ifelse(df_clustering$larger_size   > median(df_clustering$larger_size), 1, 0),
  numero_nuclei = ifelse(df_clustering$numero_nuclei > median(df_clustering$numero_nuclei), 1, 0)
)

# MATRICE DI DISTANZA (JACCARD)
dist_jaccard <- daisy(df_binary, metric = "gower")  

# CLUSTERING GERARCHICO (Legame Completo)
hc <- hclust(dist_jaccard, method = "complete")

# Dendrogramma
plot(
  hc,
  labels = FALSE,
  hang = -1,
  main = "Clustering gerarchico (Complete linkage, distanza Jaccard)",
  xlab = "Osservazioni",
  ylab = "Distanza"
)

# SCELTA DEL NUMERO DI CLUSTER (Silhouette)
silhouette_analysis <- function(k){
  clusters <- cutree(hc, k = k)
  sil <- silhouette(clusters, dist_jaccard)
  mean(sil[, 3])
}

k_values <- 2:10
silhouette_scores <- sapply(k_values, silhouette_analysis)

plot(
  k_values,
  silhouette_scores,
  type = "b",
  pch = 19,
  xlab = "Numero di cluster",
  ylab = "Silhouette media",
  main = "Metodo della silhouette"
)

optimal_k <- k_values[which.max(silhouette_scores)]
cat("Numero ottimale di cluster:", optimal_k, "\n")

# CLUSTERING FINALE
df_complete$cluster <- factor(cutree(hc, k = optimal_k))

# TABELLA DI CONTINGENZA: CLUSTER × DIAGNOSI
tab_cluster_diagnosi <- table(df_complete$cluster, df_complete$diagnosis)
print(tab_cluster_diagnosi)

# Percentuali per cluster
prop.table(tab_cluster_diagnosi, margin = 1) * 100

# Percentuali per diagnosi
prop.table(tab_cluster_diagnosi, margin = 2) * 100

# CENTROIDI
centroids <- aggregate(
  cbind(larger_size, numero_nuclei) ~ cluster,
  data = df_complete,
  mean
)

# GRAFICO FINALE
plot(
  df_complete$larger_size,
  df_complete$numero_nuclei,
  col = df_complete$cluster,
  pch = 19,
  xlab = "Dimensione della lesione (cm)",
  ylab = "Numero di nuclei",
  main = "Fenotipi morfologici (Clustering gerarchico)"
)

points(
  centroids$larger_size,
  centroids$numero_nuclei,
  pch = 4,
  cex = 2,
  lwd = 3
)

legend(
  "topright",
  legend = c(paste("Cluster", levels(df_complete$cluster)), "Centroide"),
  col = c(1:length(levels(df_complete$cluster)), "black"),
  pch = c(rep(19, length(levels(df_complete$cluster))), 4)
)


# WCSS / BCSS / Calinski-Harabasz
X <- scale(df_clustering)  # standardizzazione come nel tuo kmeans
clusters <- as.integer(df_complete$cluster)

k <- length(unique(clusters))
n <- nrow(X)

# Centroide globale
global_centroid <- colMeans(X)

# Centroidi per cluster (euclidei)
centroids_euclid <- aggregate(X, by = list(cluster = clusters), FUN = mean)
# La prima colonna è "cluster", le altre sono le medie
centroids_mat <- as.matrix(centroids_euclid[, -1, drop = FALSE])

# WCSS: somma delle distanze quadratiche dei punti dal centroide del proprio cluster
wcss <- 0
for (i in 1:k) {
  idx <- which(clusters == i)
  Xi <- X[idx, , drop = FALSE]
  ci <- centroids_mat[i, ]
  wcss <- wcss + sum(rowSums((Xi - matrix(ci, nrow = nrow(Xi), ncol = ncol(Xi), byrow = TRUE))^2))
}

# BCSS: tra-cluster, rispetto al centroide globale
bcss <- 0
for (i in 1:k) {
  idx <- which(clusters == i)
  ni <- length(idx)
  ci <- centroids_mat[i, ]
  bcss <- bcss + ni * sum((ci - global_centroid)^2)
}

cat("Centroide Globale:", global_centroid, "\n")
cat("WCSS:", wcss, "\n")
cat("BCSS:", bcss, "\n")

# Indice di Calinski
ch_index <- (bcss / (k - 1)) / (wcss / (n - k))
cat("Indice di Calinski–Harabasz:", ch_index, "\n")


# SALVATAGGIO DATASET CON CLUSTER

# Percorso di salvataggio
DATASET_CLUSTER <- file.path(DATA, "dataset_con_cluster.csv")

# Salvataggio del dataset completo con cluster
write.csv(
  df_complete,
  file = DATASET_CLUSTER,
  row.names = FALSE
)

cat("Dataset con cluster salvato in:", DATASET_CLUSTER, "\n")