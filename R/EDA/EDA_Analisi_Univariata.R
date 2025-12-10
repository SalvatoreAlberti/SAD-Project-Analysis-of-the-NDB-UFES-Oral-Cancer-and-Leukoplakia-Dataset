library(dplyr)

#CARICAMENTO DEL DATASET
# Risalgo di due livelli dalla cartella R
PROJECT_ROOT <- normalizePath("../../../", winslash = "/")

# Percorso cartella dataset
DATA <- file.path(PROJECT_ROOT,
                  "dataset")


# Percorso completo del CSV
DATASET_ORIGINALE <- file.path(DATA, "dataset_modificato.csv")

# Caricamento dataset
df <- read.csv(DATASET_ORIGINALE, header = TRUE, sep = ",")

#Analisi Univariata

#Analisi variabili numeriche

#Calcolo di variabili statistiche: Media, mediana, min e max, quartili
#deviazione standard
  variabili_numeriche <- c ("larger_size")
  #ordino la variabile age_group
  df$age_group <- factor(
    df$age_group,
    levels = c("Young", "Middle", "Elderly"),  # ordine che vuoi
    ordered = TRUE
  )
  
  for(variabile in variabili_numeriche){
    cat("\n-----", variabile, "-----\n")
  
    print(summary(df[[variabile]]*10))  #calcolo di min, max, quartili e mediana
    cat("Deviazione standard:", sd(df[[variabile]]*10, na.rm= TRUE), "\n\n")
  }
  
  # Istogramma + densità per ogni variabile numerica
  for(variabile in variabili_numeriche){
    
    # Istogramma
    hist(df[[variabile]]*10,
         main = paste("Istogramma di", variabile),
         xlab = variabile,
         col = "lightblue",
         border = "white")
    
    # Densità
    plot(density(df[[variabile]]*10, na.rm = TRUE),
         main = paste("Densità di", variabile),
         xlab = variabile)
  }
  
  
  #BoxPlot per identificare gli outlier
    for(variabile in variabili_numeriche){
      boxplot(df[[variabile]]*10,
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
      ]*10
      cat("Outlier per", variabile, ":\n")
      print(outliers)
      cat("\n")
    }
  
  #Analisi variabili categoriali
  variabili_categoriali <- c("localization", "tobacco_use", "alcohol_consumption",
    "sun_exposure", "gender", "skin_color", "age_group", "diagnosis",
    "dysplasia_severity", "TaskII", "TaskIII", "TaskIV"
  )
  
  for(variabile in variabili_categoriali){
    cat("\n-----", variabile, "-----\n")
    
    cat("frequenze assolute: \n")
    print(table(df[[variabile]])) #Frequenze Assolute
    cat("frequenze relative: \n")
    print(prop.table(table(df[[variabile]]))*100)   #Frequenze relative
  }
  
  #Barplot per visualizzare la distribuzione
  op <- par(mar = c(9, 4, 4, 2), xpd = NA)
  
  for (variabile in variabili_categoriali) {
    freq  <- table(df[[variabile]])
    max_y <- max(freq) * 1.4
    
    # etichette mandate a capo se troppo lunghe
    lab_wrapped <- sapply(names(freq), function(x)
      paste(strwrap(x, width = 7), collapse = "\n")
    )
    
    # barplot SENZA etichette sull’asse x
    bp <- barplot(freq,
                  main = paste("Barplot di", variabile),
                  col  = "skyblue",
                  ylim = c(0, max_y),
                  xaxt = "n",   # niente nomi sull'asse x
                  las  = 1)     # solo l’asse y orizzontale
    
    # aggiungo le etichette più in basso
    text(x      = bp,
         y      = -max_y * 0.12,   # sposta più o meno in basso (puoi regolare 0.12)
         labels = lab_wrapped,
         cex    = 0.8,
         adj    = c(0.5, 1))
  }
  
  #ricostruisco i barplot e le tabelle per le variabili che contengono not informed rimuovendolo
  variabili_categoriali_not_informed<-c("tobacco_use", "alcohol_consumption","sun_exposure","skin_color")
  for(variabile in variabili_categoriali_not_informed){
    cat("\n-----", variabile, "senza Not informed-----\n")
    table<-table(df[[variabile]])
    table_modificata<-table[names(table)!="Not informed"]
    cat("frequenze assolute: \n")
    print(table_modificata) #Frequenze Assolute
    cat("frequenze relative: \n")
    print(prop.table(table_modificata)*100)   #Frequenze relative
  }
  for (variabile in variabili_categoriali_not_informed) { 
    freq<-table(df[[variabile]])
    freq_senza_NI <- freq[names(freq) != "Not informed"]
    max_y <- max(freq_senza_NI) * 1.4
    
    barplot(freq_senza_NI,
            main = paste("Barplot di", variabile, "senza Not informed"),
            las  = 1,
            col  = "skyblue",
            ylim = c(0, max_y))
  }
  
  
  
  
  df$dysplasia_severity[df$dysplasia_severity == ""] <- NA
  #ricostruisco barplot e analisi per dysplasia_severity senza considerare i valori NA
  cat("\n-----dysplasia_severity senza NA-----\n")
  table<-table(df[["dysplasia_severity"]])
  table_modificata<-table[!is.na(names(table))]
  cat("frequenze assolute: \n")
  print(table_modificata) #Frequenze Assolute
  cat("frequenze relative: \n")
  print(prop.table(table_modificata)*100)   #Frequenze relative
  
  max_y <- max(table_modificata) * 1.4
  
  barplot(table_modificata,
          main = paste("Barplot di dysplasia_severity senza NA"),
          las  = 1,
          col  = "skyblue",
          ylim = c(0, max_y))
