# ================================
# CARICAMENTO DATASET
# ================================
PROJECT_ROOT <- normalizePath("../../../", winslash = "/")
DATA <- file.path(PROJECT_ROOT, "dataset")
DATASET_ORIGINALE <- file.path(DATA, "dataset_modificato.csv")

df <- read.csv(DATASET_ORIGINALE, header = TRUE, sep = ";", stringsAsFactors = FALSE)
ncol(df)
names(df)
library(dplyr)
library(VIM)

# ================================
# FUNZIONI
# ================================
to_na_not_informed <- function(x) {
  x <- trimws(as.character(x))
  x[tolower(x) == "not informed"] <- NA
  x
}

knn_impute_kdiff <- function(df_in,
                             k_skin = 5,
                             k_alc_tob = 5,
                             k_sun = 9,
                             seed = 123) {
  
  # colonne target (solo se presenti)
  v_skin <- intersect("skin_color", names(df_in))
  v_alc  <- intersect("alcohol_consumption", names(df_in))
  v_tob  <- intersect("tobacco_use", names(df_in))
  v_sun  <- intersect("sun_exposure", names(df_in))
  
  # pulizia "Not informed" -> NA + factor
  vars_clean <- c(v_skin, v_alc, v_tob, v_sun)
  
  df2 <- df_in %>%
    mutate(across(all_of(vars_clean), to_na_not_informed)) %>%
    mutate(across(all_of(vars_clean), as.factor))
  
  # predittori per distanza
  dist_base <- c("age_group","gender","localization","diagnosis","dysplasia_severity",
                 "TaskII","TaskIII","TaskIV","numero_nuclei","larger_size")
  dist_base <- intersect(dist_base, names(df2))   # tengo solo quelli presenti
  
  set.seed(seed)
  
  # -------------------------
  # STEP 1: imputo skin_color (k = k_skin)
  # -------------------------
  if ("skin_color" %in% names(df2)) {
    df_skin_work <- df2[, unique(c("skin_color", dist_base)), drop = FALSE]
    
    df_skin_imp <- VIM::kNN(
      df_skin_work,
      variable = "skin_color",
      dist_var = dist_base,
      k = k_skin,
      imp_var = FALSE
    )
    
    df2[, names(df_skin_imp)] <- df_skin_imp
  }
  
  # distanza "main" include skin_color (ora imputata)
  dist_main <- intersect(c(dist_base, "skin_color"), names(df2))
  
  # -------------------------
  # STEP 2: imputo alcohol + tobacco (k = k_alc_tob)
  # -------------------------
  vars_alc_tob <- intersect(c("alcohol_consumption","tobacco_use"), names(df2))
  
  if (length(vars_alc_tob) > 0) {
    df_at_work <- df2[, unique(c(vars_alc_tob, dist_main)), drop = FALSE]
    
    df_at_imp <- VIM::kNN(
      df_at_work,
      variable = vars_alc_tob,
      dist_var = dist_main,
      k = k_alc_tob,
      imp_var = FALSE
    )
    
    df2[, names(df_at_imp)] <- df_at_imp
  }
  
  # -------------------------
  # STEP 3: imputo SOLO sun_exposure (k = k_sun)
  # -------------------------
  if ("sun_exposure" %in% names(df2)) {
    
    dist_sun <- intersect(
      c("skin_color", "age_group", "localization", "gender",
        "alcohol_consumption", "tobacco_use","numero_nuclei"),
      names(df2)
    )
    
    df_sun_work <- df2[, unique(c("sun_exposure", dist_sun)), drop = FALSE]
    
    df_sun_imp <- VIM::kNN(
      df_sun_work,
      variable = "sun_exposure",
      dist_var = dist_sun,
      k = k_sun,
      imp_var = FALSE
    )
    
    df2[, names(df_sun_imp)] <- df_sun_imp
  }
  
  
  return(df2)
}

# ================================
# ESECUZIONE
# ================================
df_imp_knn <- knn_impute_kdiff(df, k_skin = 5, k_alc_tob = 5, k_sun = 9, seed = 123)

# Controllo NA residui
check_cols <- intersect(c("skin_color","alcohol_consumption","tobacco_use","sun_exposure"), names(df_imp_knn))
print(sapply(df_imp_knn[check_cols], function(x) sum(is.na(x))))

# Stampa: tutte le colonne (tibble/console-friendly)
print(df_imp_knn)

# (opzionale) apri in visualizzazione tabellare (RStudio)
# View(df_imp_knn)

# --- STAMPA SOLO DEI VALORI IMPUTATI (per le colonne interessate) ---

target_cols <- intersect(c("skin_color","alcohol_consumption","tobacco_use","sun_exposure"), names(df_imp_knn))

# 1) Ricreo il dataset "pulito" (Not informed -> NA) SOLO per capire quali righe erano mancanti prima
df_before <- df %>%
  mutate(across(all_of(target_cols), to_na_not_informed)) %>%
  mutate(across(all_of(target_cols), as.factor))

# 2) Indici delle righe che avevano NA in almeno una delle colonne interessate
idx_imp <- Reduce(`|`, lapply(target_cols, function(v) is.na(df_before[[v]])))

# 3) Stampo solo quelle righe e solo le colonne interessate (più ID se esistono)
id_cols <- intersect(c("public_id","lesion_id","patient_id","path"), names(df_imp_knn))
cols_show <- c(id_cols, target_cols)
if ("dysplasia_severity" %in% names(df_imp_knn)) {
  df_imp_knn$dysplasia_severity <- trimws(as.character(df_imp_knn$dysplasia_severity))
  df_imp_knn$dysplasia_severity[df_imp_knn$dysplasia_severity == "" | is.na(df_imp_knn$dysplasia_severity)] <- "No Leukoplakia"
  df_imp_knn$dysplasia_severity <- as.factor(df_imp_knn$dysplasia_severity)
}

print(df_imp_knn)
write.csv2(df_imp_knn, DATASET_ORIGINALE, row.names = FALSE)

