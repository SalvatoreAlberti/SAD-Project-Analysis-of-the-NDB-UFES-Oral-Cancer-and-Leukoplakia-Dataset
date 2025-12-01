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

#ANLISI BIVARIATA 
target <- "diagnosis" 
variabili_numeriche <- c("larger_size", "age_group")
variabili_categoriali <- c("tobacco_use", "alcohol_consumption", "sun_exposure",
                           "gender", "skin_color", "localization","dysplasia_severity",
                           "TaskII", "TaskIII", "TaskIV") 

#Analisi Bivariata - Variabili numeriche -> Diagnosi 
for(variabile in variabili_numeriche){
  cat("\n-----Analisi Bivariata-----\n") 
  
  #Boxplot 
  boxplot(df[[variabile]] ~ df[[target]],
          main = paste("Boxplot di", variabile, "per", target),
          xlab = target, ylab = variabile, col = "lightblue")
  
  #Densità per ogni classi di diagnosis 
  valori_target <- unique(df[[target]]) 
  plot(NULL, xlim = range(df[[variabile]], na.rm = TRUE), ylim = c(0,1), 
       main = paste("Densità di", variabile, "per", target),
       xlab = variabile, ylab = "Densità") 
  
  colori <- c("red", "blue", "green", "purple")
  i <- 1 
  
  for (val in valori_target) {
    dens <- density(df[[variabile]][df[[target]] == val], na.rm = TRUE)
    lines(dens, col = colori[i], lwd = 2) 
    legend("topright", legend = valori_target,
           col = colori[1:length(valori_target)],lwd = 2) 
    i <- i + 1 
  } 
  
  # Istogrammi sovrapposti 
  colori_transp <- c(rgb(1,0,0,0.5), rgb(0,0,1,0.5), rgb(0,1,0,0.5))
  hist(df[[variabile]][df[[target]] == valori_target[1]],
       col = colori_transp[1],
       main = paste("Istogrammi di", variabile, "per", target),
       xlab = variabile, freq = FALSE, border = "white") 
  
  for (i in 2:length(valori_target)) { 
    hist(df[[variabile]][df[[target]] == valori_target[i]], 
         col = colori_transp[i], add = TRUE, 
         freq = FALSE, border = "white") 
  } 
  
  # Confronto delle medie 
  cat("Media per gruppi:\n") 
  tapply(df[[variabile]], df[[target]], mean, na.rm = TRUE)
  
  # Test statistico: ANOVA 
  modello <- aov(df[[variabile]] ~ df[[target]])
  print(summary(modello)) 
  
}


#Analisi Bivariata - Variabili categoriali -> Diagnosi 
for(variabile in variabili_categoriali){
  cat("\n-----Analisi Bivariata-----\n") 
  
  tab <- table(df[[variabile]], df[[target]])
  print(tab)
  
  #percentuali
  cat("\nPercentuali per riga:\n")
  print(prop.table(tab, 1))
  
  cat("\nPercentuali per colonna:\n")
  print(prop.table(tab, 2))
  
  #BarPlot
  barplot(tab, beside = TRUE,
          main = paste("Distribuzione di", variabile, "per", target),
          xlab = target, ylab = "Frequenze",
          col = rainbow((nrow(tab))))
  legend("topright", legend = rownames(tab), fill = rainbow(nrow(tab)))
  
  # Test chi-quadrato
  if (all(tab > 0)) {
    cat("\nTest chi-quadrato:\n")
    print(chisq.test(tab))
  } else {
    cat("\nChi-quadrato NON applicabile (celle con zero)\n")
  }
  
}