library(dplyr)

# ================================
# CARICAMENTO DATASET CON CLUSTER
# ================================

PROJECT_ROOT <- normalizePath("../../../../", winslash = "/")
DATA <- file.path(PROJECT_ROOT, "dataset")

DATASET_CLUSTER <- file.path(DATA, "dataset_con_cluster.csv")

df <- read.csv(DATASET_CLUSTER, header = TRUE, sep = ",")

# ================================
# 0. Impostazioni iniziali
# ================================

# Variabile chiave (fenotipo morfologico)
cluster_var <- "cluster"

# Variabili cliniche e demografiche (RQ2)
rq2_vars <- c(
  "tobacco_use",
  "alcohol_consumption",
  "sun_exposure",
  "diagnosis",
  "localization",
  "gender",
  "age_group",
  "skin_color"
)

# Converto in factor
df[[cluster_var]] <- factor(df[[cluster_var]])
df[rq2_vars] <- lapply(df[rq2_vars], factor)

# Ordine per age_group (se presente)
if ("age_group" %in% names(df)) {
  df$age_group <- factor(
    df$age_group,
    levels = c("Young", "Middle", "Elderly"),
    ordered = TRUE
  )
}

# ================================
# 1. Funzioni statistiche
# ================================

chi_square_safe <- function(x, y) {
  tab <- table(x, y)
  
  if (nrow(tab) < 2 || ncol(tab) < 2 || sum(tab) == 0) {
    return(list(p_value = NA, chi2 = NA, df = NA, expected_min = NA))
  }
  
  test <- suppressWarnings(chisq.test(tab, correct = FALSE))
  
  list(
    p_value = as.numeric(test$p.value),
    chi2 = as.numeric(test$statistic),
    df = as.numeric(test$parameter),
    expected_min = min(test$expected)
  )
}

cramers_v <- function(x, y) {
  tab <- table(x, y)
  
  if (nrow(tab) < 2 || ncol(tab) < 2 || sum(tab) == 0) return(NA)
  
  chi <- suppressWarnings(chisq.test(tab, correct = FALSE))
  
  chi2 <- as.numeric(chi$statistic)
  n <- sum(tab)
  r <- nrow(tab)
  k <- ncol(tab)
  
  sqrt(chi2 / (n * (min(r, k) - 1)))
}

# ================================
# 2. ANALISI RQ2: CLUSTER × VARIABILI
# ================================

results_rq2 <- data.frame(
  variable = character(),
  p_value = numeric(),
  chi2 = numeric(),
  df = numeric(),
  expected_min = numeric(),
  cramers_v = numeric(),
  strength = character(),
  stringsAsFactors = FALSE
)

for (v in rq2_vars) {
  
  tmp <- df[, c(cluster_var, v)]
  tmp <- tmp[complete.cases(tmp), ]
  
  if (nrow(tmp) == 0 || length(unique(tmp[[v]])) < 2) next
  
  res <- chi_square_safe(tmp[[cluster_var]], tmp[[v]])
  v_cr <- cramers_v(tmp[[cluster_var]], tmp[[v]])
  
  strength <- NA
  if (!is.na(v_cr)) {
    strength <- cut(
      v_cr,
      breaks = c(-Inf, 0.1, 0.3, 0.5, Inf),
      labels = c("assente/debole", "debole", "moderata", "forte")
    )
    strength <- as.character(strength)
  }
  
  results_rq2 <- rbind(
    results_rq2,
    data.frame(
      variable = v,
      p_value = res$p_value,
      chi2 = res$chi2,
      df = res$df,
      expected_min = res$expected_min,
      cramers_v = v_cr,
      strength = strength,
      stringsAsFactors = FALSE
    )
  )
}

# ================================
# 3. Correzione per confronti multipli
# ================================

results_rq2$p_adj_fdr <- p.adjust(results_rq2$p_value, method = "fdr")

# Ordino per significatività
results_rq2 <- results_rq2[order(results_rq2$p_adj_fdr), ]

cat("\n===== RQ2: Associazione tra fenotipi morfologici e fattori clinico-demografici =====\n")
print(results_rq2)
