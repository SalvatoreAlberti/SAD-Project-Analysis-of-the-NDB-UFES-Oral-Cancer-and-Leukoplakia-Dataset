# ================================
# 0. LIBRERIE E CARICAMENTO DATI
# ================================

library(dplyr)
library(MASS)   # per regressione logistica ordinale (polr)

PROJECT_ROOT <- normalizePath("../../../../", winslash = "/")
DATA <- file.path(PROJECT_ROOT, "dataset")

DATASET_CLUSTER <- file.path(DATA, "dataset_modificato.csv")

df <- read.csv(DATASET_CLUSTER, header = TRUE, sep = ";")

# Variabile cluster
df$cluster <- factor(df$cluster)

# ================================
# 1. DEFINIZIONE OUTCOME ORDINALE
# ================================

# Costruzione del livello di rischio clinico (progressione)
df$risk_level <- factor(
  df$diagnosis,
  levels = c(
    "Leukoplakia without dysplasia",
    "Leukoplakia with dysplasia",
    "OSCC"
  ),
  ordered = TRUE
)

# Controlli
str(df$risk_level)
table(df$risk_level)

# ================================
# 2. TABELLA DI CONTINGENZA
# ================================

# Cluster × rischio clinico
tab_risk <- table(df$cluster, df$risk_level)
print(tab_risk)

# Percentuali per cluster (righe)
prop_risk <- prop.table(tab_risk, margin = 1) * 100
round(prop_risk, 1)

# ================================
# 3. VERIFICA DELLE IPOTESI (χ²)
# ================================

# Ipotesi nulla H0:
# Il rischio clinico è indipendente dal cluster morfologico

# Ipotesi alternativa H1:
# Il rischio clinico dipende dal cluster morfologico

alpha <- 0.05

# Test χ² di indipendenza
chi_test <- chisq.test(tab_risk, correct = FALSE)
print(chi_test)

# Decisione statistica
if (chi_test$p.value < alpha) {
  cat("\nDecisione: RIFIUTO H0 (p-value <", alpha, ")\n")
} else {
  cat("\nDecisione: NON RIFIUTO H0 (p-value >=", alpha, ")\n")
}


# ================================
# 4. RESIDUI STANDARDIZZATI
# ================================

# Identificazione delle celle che contribuiscono al χ²
std_residuals <- round(chi_test$stdres, 2)
print(std_residuals)

# Interpretazione guida:
# residuo > +2  → sovra-rappresentazione
# residuo < −2  → sotto-rappresentazione

