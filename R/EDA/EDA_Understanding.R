library(dplyr)

#CARICAMENTO DEL DATASET
  # Risalgo di due livelli dalla cartella R
  PROJECT_ROOT <- normalizePath("../../../", winslash = "/")
  
  # Percorso cartella dataset
  DATA <- file.path(PROJECT_ROOT,
                    "dataset", 
                    "NDB-UFES An oral cancer and leukoplakia dataset composed of histopathological images and patient data")
                    
  
  # Percorso completo del CSV
  DATASET_ORIGINALE <- file.path(DATA, "ndb-ufes.csv")
  
  # Caricamento dataset
  df <- read.csv(DATASET_ORIGINALE, header = TRUE, sep = ",")

#UNDERSTANDING

  #Visualizzo la struttura del dataset
  cat("\nStruttura del dataset\n")
  str(df)
  
  #Conto il numero di righe e colonne del dataset
  cat("\nNumero righe e colonne del dataset\n")
  print(dim (df))
  
  #Visualizzo l'intestazione del dataset
  cat("\nIntestazione del dataset:\n")
  print(colnames(df))  # Mostra i nomi delle colonne

  cat("\nControllo per ogni feature quanti valori NA ci sono, quanto valori NON nulli, quanti valori contengono stringhe vuote, qaunti valori contengono caratteri NON alfanumerici :\n\n")
  #Controllo i valori nulli per ogni colonna del dataset
  check_na <- sapply(df, function(x) sum(is.na(x)))
  
  # Visualizzo il risultato
  null_summary <- data.frame(Feature = names(check_na), Null_Count = check_na)
  
  # Conto il numero di valori per ogni colonna del dataset
  value_counts <- sapply(df, function(x) sum(!is.na(x)))
  
  # Creo un data frame per visualizzare il risultato
  value_count_summary <- data.frame(Feature = names(value_counts), Value_Count = value_counts)
  
  # Controllo per ogni feature se ci sono stringhe vuote
  empty_string_counts <- sapply(df, function(x) sum(x == ""))
  
  # Creo un data frame per visualizzare i conteggi delle stringhe vuote
  empty_string_summary <- data.frame(Feature = names(empty_string_counts), Empty_String_Count = empty_string_counts)
  
  # Conto i valori che contengono caratteri non alfanumerici (includendo i caratteri speciali)
  non_alphanumeric_counts <- sapply(df, function(x) {
    if (is.character(x)) {
      sum(grepl("[^[:alnum:][:space:]À-ÿ]", x))  # Includo anche i caratteri speciali
    } else {
      0  # Se non è una colonna carattere, conta 0
    }
  })
  
  # Creo un data frame per visualizzare i conteggi dei caratteri non alfanumerici
  non_alphanumeric_summary <- data.frame(Feature = names(non_alphanumeric_counts), Non_Alphanumeric_Count = non_alphanumeric_counts)
  
  # Combino i data frame dei valori nulli, dei conteggi di valori, delle stringhe vuote e dei caratteri non alfanumerici
  combined_summary <- merge(null_summary, value_count_summary, by = "Feature")
  combined_summary <- merge(combined_summary, empty_string_summary, by = "Feature")
  combined_summary <- merge(combined_summary, non_alphanumeric_summary, by = "Feature")
  
  # Visualizzo il risultato finale
  print(combined_summary)
 