library(dplyr)

# ================================
# CARICAMENTO DATASET CON CLUSTER
# ================================

PROJECT_ROOT <- normalizePath("../../../../", winslash = "/")
DATA <- file.path(PROJECT_ROOT, "dataset")
DATASET_CLUSTER <- file.path(DATA, "dataset_modificato.csv")

df <- read.csv(DATASET_CLUSTER, header = TRUE, sep = ";", stringsAsFactors = FALSE)

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

#diagnosis, c'è associazione per fisher
tab <- table(df$cluster, df$diagnosis)
tab
test<-chisq.test(tab)
test$expected
fisher.test(tab, simulate.p.value = TRUE, B = 20000)

#tobacco_use, non c'è associazione per fisher
tab1<-table(df$cluster, df$tobacco_use)
tab1
test1<-chisq.test(tab1)
test1$expected
fisher.test(tab1, simulate.p.value = TRUE, B = 20000)
#alcohol_consumption, non c'è associazione per fisher
tab2<-table(df$cluster,df$alcohol_consumption)
tab2
test2<-chisq.test(tab2)
test2
test2$expected
fisher.test(tab2, simulate.p.value = TRUE, B = 20000)
#skin_color, non c'è associazione per fisher
tab3<-table(df$cluster,df$skin_color)
tab3
test3<-chisq.test(tab3)
test3$expected
fisher.test(tab3, simulate.p.value = TRUE, B = 20000)
#localization, non c'è associazione per fisher
tab4<-table(df$cluster,df$localization)
tab4
test4<-chisq.test(tab4)
test4$expected
fisher.test(tab4, simulate.p.value = TRUE, B = 20000)

tab5<-table(df$cluster,df$sun_exposure)
tab5
test5<-chisq.test(tab5)
test5$expected
fisher.test(tab5, simulate.p.value = TRUE, B = 20000)
#gender, non c'è associazione per chi quadro
tab6<-table(df$cluster,df$gender)
tab6
test6<-chisq.test(tab6)
test6
test6$expected
#age_group, non c'è associazione per fisher
tab7<-table(df$cluster,df$age_group)
tab7
test7<-chisq.test(tab7)
test7$expected
fisher.test(tab7, simulate.p.value = TRUE, B = 20000)
#dysplasia_severity, non c'è associazione per fisher
tab8<-table(df$cluster, df$dysplasia_severity)
tab8
test8<-chisq.test(tab8)
test8$expected
fisher.test(tab8, simulate.p.value = TRUE, B = 20000)

#forza legame diagnosis
chi2 <- test$statistic
n <- sum(tab)
r <- nrow(tab)
c <- ncol(tab)

cramers_v <- sqrt(chi2 / (n * (min(r, c) - 1)))
cramers_v
