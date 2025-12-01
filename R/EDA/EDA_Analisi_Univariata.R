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

#Analisi Univariata

#Analisi variabili numeriche

#Calcolo di variabili statistiche: Media, mediana, min e max, quartili
#deviazione standard
  variabili_numeriche <- c ("larger_size", "age_group")
  
  for(variabile in variabili_numeriche){
    cat("\n-----", variabile, "-----\n")
  
    print(summary(df[[variabile]]))  #calcolo di min, max, quartili e mediana
    cat("Media:", mean(df[[variabile]], na.rm = TRUE), "\n")
    cat("Deviazione standard:", sd(df[[variabile]], na.rm= TRUE), "\n\n")
  }
  
  # Istogramma + densità per ogni variabile numerica
  for(variabile in variabili_numeriche){
    
    # Istogramma
    hist(df[[variabile]],
         main = paste("Istogramma di", variabile),
         xlab = variabile,
         col = "lightblue",
         border = "white")
    
    # Densità
    plot(density(df[[variabile]], na.rm = TRUE),
         main = paste("Densità di", variabile),
         xlab = variabile)
  }
  
  
  #BoxPlot per identificare gli outlier
    for(variabile in variabili_numeriche){
      boxplot(df[[variabile]],
              main = paste("Boxplot di", variabile),
              horizontal = TRUE)
    }
  
  #Visualizzo gli outlier numericamente
    for(variabile in variabili_numeriche){
      Q1 <- quantile(df[[variabile]], 0.25, na.rm = TRUE)
      Q3 <- quantile(df[[variabile]], 0.75, na.rm = TRUE)
      IQR <- Q3-Q1
      
      outliers <- df[[variabile]][ 
        (df[[variabile]] < Q1 - 1.5*IQR) |
          (df[[variabile]] > Q3 + 1.5*IQR)
      ]
      cat("Outlier per", variabile, ":\n")
      print(outliers)
      cat("\n")
    }
  
  #Analisi variabili categoriali
  variabili_categoriali <- c(
    "path", "localization", "tobacco_use", "alcohol_consumption",
    "sun_exposure", "gender", "skin_color", "diagnosis",
    "dysplasia_severity", "TaskII", "TaskIII", "TaskIV"
  )
  
  for(variabile in variabili_categoriali){
    cat("\n-----", variabile, "-----\n")
    
    print(table(df[[variabile]]))               #Frequenze Assolute
    print(prop.table(table(df[[variabile]])))   #Frequenze relative
  }
  
  #Barplot per visualizzare la distribuzione
  for (variabile in variabili_categoriali) {
    barplot(table(df[[variabile]]),
            main = paste("Barplot di", variabile),
            las = 2, col = "skyblue")
  }
  
  #Gestione delle stringhe vuote in "dysplasia_severity"
  df$dysplasia_severity[df$dysplasia_severity == ""] <- NA
  
  
