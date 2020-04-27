setwd("~/Desktop/binding")
library(tidyverse)

# Load all data for akt1
all_data <- read.csv("akt-1/data/all_data_akt1_unique.csv") %>% select(-X)

# Keep rdkits
unique_rdkit <- read_csv("akt-1/data/all_data_rdkit_unique.csv") %>% select(-X1)

# Load similarities for unique rdkits
similarities <- read_csv("akt-1/data/ecfp_sims.csv") %>% select(-X1)

# Calculate desired ratios
binary_ratio <- sum(all_data$Binary)/nrow(all_data)*100
not_sure_ratio <- sum(all_data$not_sure)/nrow(all_data)*100
assay_type_ratio <- nrow(filter(all_data,assay_type == "non_binding"))/nrow(all_data)*100

ratio <- c(binary_ratio,assay_type_ratio,not_sure_ratio)


###### Try to divide data at 8 categories from tree

data1 <- all_data %>% filter(Binary == 1 & assay_type == "binding" & not_sure == 1)
data2 <- all_data %>% filter(Binary == 1 & assay_type == "binding" & not_sure == 0)
data3 <- all_data %>% filter(Binary == 1 & assay_type == "non_binding" & not_sure == 1)
data4 <- all_data %>% filter(Binary == 1 & assay_type == "non_binding" & not_sure == 0)
data5 <- all_data %>% filter(Binary == 0 & assay_type == "binding" & not_sure == 1)
data6 <- all_data %>% filter(Binary == 0 & assay_type == "binding" & not_sure == 0)
data7 <- all_data %>% filter(Binary == 0 & assay_type == "non_binding" & not_sure == 1)
data8 <- all_data %>% filter(Binary == 0 & assay_type == "non_binding" & not_sure == 0)

d1 <- as.character(data1$rdkit)
d2 <- as.character(data2$rdkit)
d3 <- as.character(data3$rdkit)
d4 <- as.character(data4$rdkit)
d5 <- as.character(data5$rdkit)
d6 <- as.character(data6$rdkit)
d7 <- as.character(data7$rdkit)
d8 <- as.character(data8$rdkit)

cross_validation <- function(dataframe,d1,d2,d3,d4,d5,d6,d7,d8,allq1smiles,ecfp_sims,no_folds,n_drugs,max_sim,min_sim,ratio,dir){
  ecfp_sims <- as.matrix(ecfp_sims)
  diag(ecfp_sims) <- 0
  colmax <- apply(ecfp_sims,2,max)
  indcandidates <- which(colmax < max_sim & colmax >= min_sim)
  allq1smiles <- as.character(allq1smiles$rdkit)
  names(colmax) <- allq1smiles
  candidates <- as.character(allq1smiles[indcandidates])
  d1_cand <- d1[which(d1 %in% candidates)]
  d2_cand <- d2[which(d2 %in% candidates)]
  d3_cand <- d3[which(d3 %in% candidates)]
  d4_cand <- d4[which(d4 %in% candidates)]
  d5_cand <- d5[which(d5 %in% candidates)]
  d6_cand <- d6[which(d6 %in% candidates)]
  d7_cand <- d7[which(d7 %in% candidates)]
  d8_cand <- d8[which(d8 %in% candidates)]
  
  cold_val <- function(all_data,d1,d2,d3,d4,d5,d6,d7,d8,ncold,ratio,mtry){
    library(tidyverse)
    
    
    best_opt <- c(5,5,5)
    for (i in 1:mtry) {
     
      drugs_1 <- data.frame(sample(d1,ncold[1],replace = F))
      colnames(drugs_1) <- "rdkit"
      drugs_2 <- data.frame(sample(d2,ncold[2],replace = F))
      colnames(drugs_2) <- "rdkit"
      drugs_3 <- data.frame(sample(d3,ncold[3],replace = F))
      colnames(drugs_3) <- "rdkit"
      drugs_4 <- data.frame(sample(d4,ncold[4],replace = F))
      colnames(drugs_4) <- "rdkit"
      drugs_5 <- data.frame(sample(d5,ncold[5],replace = F))
      colnames(drugs_5) <- "rdkit"
      drugs_6 <- data.frame(sample(d6,ncold[6],replace = F))
      colnames(drugs_6) <- "rdkit"
      drugs_7 <- data.frame(sample(d7,ncold[7],replace = F))
      colnames(drugs_7) <- "rdkit"
      drugs_8 <- data.frame(sample(d8,ncold[8],replace = F))
      colnames(drugs_8) <- "rdkit"
      
      drugs_sampled <- bind_rows(drugs_1,drugs_2,drugs_3,drugs_4,drugs_5,drugs_6,drugs_7,drugs_8)
      drugs_sampled <- drugs_sampled$rdkit
      indices_all <- unique(which(all_data$rdkit %in% drugs_sampled))
      
      
      bin_new <- sum(all_data[indices_all,]$Binary)/nrow(all_data[indices_all,])*100
      as_new <- nrow(filter(all_data[indices_all,],assay_type == "non_binding"))/nrow(all_data[indices_all,])*100
      ns_new <- sum(all_data[indices_all,]$not_sure)/nrow(all_data[indices_all,])*100
      new_ratio <- c(bin_new,as_new,ns_new)
      
      opt <- abs(new_ratio-ratio)
      opt1 <- opt[1]
      opt2 <- opt[2]
      opt3 <- opt[3]
      if ((opt1 < best_opt[1]) & (opt2 < best_opt[2]) & (opt3 < best_opt[3]) ) {
        ratio <- new_ratio
        best_opt <- opt
        best_sample <- drugs_sampled
        print(best_opt)
        print(i)
      }
    }
    return(best_sample)
  }
  for (i in 7:no_folds) {
    alldata <- dataframe
    cold <- cold_val(all_data = alldata,d1 = as.character(d1_cand),d2 = as.character(d2_cand),
                     d3 = as.character(d3_cand),d4 = as.character(d4_cand),
                     d5 = as.character(d5_cand),d6 = as.character(d6_cand),
                     d7 = as.character(d7_cand),d8 = as.character(d8_cand),
                     ncold = n_drugs,ratio = ratio,mtry = 20000)
    
    alldata$iscold <- alldata$rdkit %in% cold
    
    val_data <- alldata %>% filter(iscold == T)
    train_data <- anti_join(alldata,val_data)
    dir.create(paste0(dir,"/fold_",i))
    trainsmiles <- unique(c(train_data$rdkit))
    write.csv(cold,paste0(dir,"/fold_",i,"/valsmiles_",i,".csv"))
    write.csv(trainsmiles,paste0(dir,"/fold_",i,"/trainsmiles_",i,".csv"))
    indkeep <- unique(c(which(val_data$rdkit %in% cold)))
    val_data <- val_data[indkeep,]
    val_data_cold <- val_data %>% filter((iscold == T ))
    val_data <- val_data %>% filter(!(iscold == T))
    write.csv(train_data,paste0(dir,"/fold_",i,"/train_",i,".csv"))
    write.csv(val_data,paste0(dir,"/fold_",i,"/val_",i,".csv"))
    write.csv(val_data_cold,paste0(dir,"/fold_",i,"/val_cold_",i,".csv"))
    
    # make new candidates
    print(length(which(cold %in% trainsmiles)))
    d1_cand <- d1_cand[-which(d1_cand %in% cold)]
    d2_cand <- d2_cand[-which(d2_cand %in% cold)]
    d3_cand <- d3_cand[-which(d3_cand %in% cold)]
    d4_cand <- d4_cand[-which(d4_cand %in% cold)]
    d5_cand <- d5_cand[-which(d5_cand %in% cold)]
    d6_cand <- d6_cand[-which(d6_cand %in% cold)]
    d7_cand <- d7_cand[-which(d7_cand %in% cold)]
    d8_cand <- d8_cand[-which(d8_cand %in% cold)]
    
  }
}

dir <- "akt-1/data_crossval"
no_folds <- 7 #200 each for 5 fold, 
n_drugs <- c(12,16,0,0,17,25,0,0) #c(32,50,0,0,51,67,0,0) for max=0.9 min=0.2
max_sim <- 0.70
min_sim <- 0.40

cross_validation(dataframe = all_data,d1 = d1,d2 = d2,d3 = d3,d4 = d4,d5 = d5,d6 = d6,
                 d7 = d7,d8 = d8,allq1smiles = unique_rdkit,ecfp_sims = similarities,
                 no_folds = no_folds ,n_drugs = n_drugs ,max_sim = max_sim,
                 min_sim = min_sim,ratio = ratio,dir = dir)

ecfp_sims <- as.matrix(similarities)
diag(ecfp_sims) <- 0
colmax <- apply(ecfp_sims,2,max)
indcandidates <- which(colmax < max_sim & colmax >= min_sim)
allq1smiles <- as.character(unique_rdkit$rdkit)
names(colmax) <- allq1smiles
candidates <- as.character(allq1smiles[indcandidates])
d1_cand <- d1[which(d1 %in% candidates)]
d2_cand <- d2[which(d2 %in% candidates)]
d3_cand <- d3[which(d3 %in% candidates)]
d4_cand <- d4[which(d4 %in% candidates)]
d5_cand <- d5[which(d5 %in% candidates)]
d6_cand <- d6[which(d6 %in% candidates)]
d7_cand <- d7[which(d7 %in% candidates)]
d8_cand <- d8[which(d8 %in% candidates)]

sum(n_drugs)

