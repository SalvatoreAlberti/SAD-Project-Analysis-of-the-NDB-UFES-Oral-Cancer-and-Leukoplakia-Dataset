library(dplyr)

# ================================
# CARICAMENTO DEL DATASET
# ================================

# Risalgo di due livelli dalla cartella R
PROJECT_ROOT <- normalizePath("../../../", winslash = "/")

# Percorso cartella dataset
DATA <- file.path(PROJECT_ROOT, "dataset")

# Percorso completo del CSV
DATASET_ORIGINALE <- file.path(DATA, "dataset_modificato.csv")

# Caricamento dataset
df <- read.csv(DATASET_ORIGINALE, header = TRUE, sep = ",")

# ================================
# 0. Impostazioni iniziali
# ================================

# Colonne ID da escludere
id_vars <- c("public_id", "lesion_id", "patient_id", "path")

# Variabili numeriche "vere"
numeric_true <- c("larger_size")  # aggiungi qui se ne hai altre

# Tutte le variabili categoriali (tutto tranne ID e numeriche vere)
categorical_vars <- setdiff(names(df), c(id_vars, numeric_true))

# Trasformo in factor tutte le variabili categoriali
df[categorical_vars] <- lapply(df[categorical_vars], factor)

# Se age_group è già stringa nel CSV, qui imposto solo l'ORDINE
if ("age_group" %in% names(df)) {
  df$age_group <- factor(
    df$age_group,
    levels = c("Young", "Middle", "Elderly"),  # adatta ai tuoi livelli reali
    ordered = TRUE
  )
}

cat_vars <- categorical_vars  # alias

# ================================
# 1. Funzione per χ² "sicuro"
# ================================

chi_square_safe <- function(x, y) {
  tab <- table(x, y)
  
  # Se tabella troppo piccola o degenerata → NA
  if (nrow(tab) < 2 || ncol(tab) < 2 || sum(tab) == 0) {
    return(list(
      p_value = NA_real_,
      chi2 = NA_real_,
      df = NA_real_,
      expected_min = NA_real_
    ))
  }
  
  # Applica χ² senza correzione di Yates
  test <- suppressWarnings(chisq.test(tab, correct = FALSE))
  
  return(list(
    p_value = as.numeric(test$p.value),
    chi2 = as.numeric(test$statistic),
    df = as.numeric(test$parameter),
    expected_min = min(test$expected)
  ))
}

# ================================
# 2. Funzione per Cramér's V
# ================================

cramers_v <- function(x, y) {
  tab <- table(x, y)
  
  # Se tabella troppo piccola → NA
  if (nrow(tab) < 2 || ncol(tab) < 2 || sum(tab) == 0) {
    return(NA_real_)
  }
  
  suppressWarnings({
    chi <- chisq.test(tab, correct = FALSE)
  })
  
  chi2 <- as.numeric(chi$statistic)
  n    <- sum(tab)
  r    <- nrow(tab)
  k    <- ncol(tab)
  
  v <- sqrt(chi2 / (n * (min(r, k) - 1)))
  return(v)
}

# ================================
# 3. χ² + Cramér's V per tutte le coppie di categoriche
# ================================

results <- data.frame(
  var1 = character(),
  var2 = character(),
  p_value = numeric(),
  chi2 = numeric(),
  df = numeric(),
  expected_min = numeric(),
  cramers_v = numeric(),
  strength = character(),
  stringsAsFactors = FALSE
)

for (i in seq_along(cat_vars)) {
  for (j in seq_along(cat_vars)) {
    if (i < j) {
      x <- df[[cat_vars[i]]]
      y <- df[[cat_vars[j]]]
      
      res_chi <- chi_square_safe(x, y)
      v       <- cramers_v(x, y)
      
      # interpreto forza di associazione da Cramér's V
      strength <- NA_character_
      if (!is.na(v)) {
        strength <- cut(
          v,
          breaks = c(-Inf, 0.1, 0.3, 0.5, Inf),
          labels = c("assente/debole", "debole", "moderata", "forte")
        )
        strength <- as.character(strength)
      }
      
      results <- rbind(results, data.frame(
        var1         = cat_vars[i],
        var2         = cat_vars[j],
        p_value      = res_chi$p_value,
        chi2         = res_chi$chi2,
        df           = res_chi$df,
        expected_min = res_chi$expected_min,
        cramers_v    = v,
        strength     = strength,
        stringsAsFactors = FALSE
      ))
    }
  }
}

# Ordino per p-value crescente
results_sorted <- results[order(results$p_value), ]

cat("\n===== RISULTATI χ² + Cramér's V =====\n")
print(results_sorted)


# ================================
# 4. Kruskal–Wallis: larger_size vs tutte le categoriche
# ================================

numeric_var <- "larger_size"

kruskal_results <- data.frame(
  var = character(),
  p_value = numeric(),
  statistic = numeric(),
  stringsAsFactors = FALSE
)

for (v in cat_vars) {
  # costruisco un data.frame temporaneo pulito
  tmp <- data.frame(
    y = df[[numeric_var]],
    g = df[[v]]
  )
  
  # rimuovo NA
  tmp <- tmp[!is.na(tmp$y) & !is.na(tmp$g), ]
  
  # se c'è meno di 2 gruppi, salto
  if (nrow(tmp) == 0 || length(unique(tmp$g)) < 2) next
  
  test <- kruskal.test(y ~ g, data = tmp)
  
  kruskal_results <- rbind(
    kruskal_results,
    data.frame(
      var = v,
      p_value = as.numeric(test$p.value),
      statistic = as.numeric(test$statistic),
      stringsAsFactors = FALSE
    )
  )
}

# Ordino per p-value
kruskal_results <- kruskal_results[order(kruskal_results$p_value), ]

cat("\n===== RISULTATI KRUSKAL–WALLIS (larger_size vs variabili categoriali) =====\n")
print(kruskal_results)



