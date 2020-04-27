setwd("~/Desktop/binding")
library(tidyverse)

restricted_dataset <- function(similarities, sim_threshold_low, sim_threshold_up, unique_rdkit, ndrugs) {
  library(tidyverse)
  # Load similarities for unique rdkits
  
  diag(similarities) <- -1
  similarities2 <-similarities >= sim_threshold_low 
  similarities2 <- as.matrix(similarities2)
  similarities2 <- similarities2 + 0
  
  rownames(similarities2) <- as.character(unique_rdkit$rdkit)
  colnames(similarities2) <- as.character(unique_rdkit$rdkit)
  sums <- rowSums(similarities2)
  names(sums) <- as.character(unique_rdkit$rdkit)
  sums <- sort(sums,decreasing = F)
  sums <- as.data.frame(sums)
  sums <- sums %>% rownames_to_column("drug")
  sums <- sums %>% group_by(sums) %>% mutate(count = n_distinct(drug))%>% ungroup
  un_sums <- unique(sums$sums)
  val_candidates <- sums$drug[sums$sums==0]
  init <- length(val_candidates)
  similarities2 <- similarities2[,-which(colnames(similarities2) %in% val_candidates)]
  for (i in ((init+1):ndrugs)) {
    drugs_inloop <- as.character(sums$drug[i])
    similarities2 <- similarities2[,which(!(colnames(similarities2) %in% drugs_inloop))]
    filt_row_sims <- similarities2[drugs_inloop,]
    similarities2 <- similarities2[,which(!((filt_row_sims)>=1))]
    #drugs_inloop <- drugs_inloop[-which(drugs_inloop %in% colnames(similarities2))]
    val_candidates <- c(val_candidates,drugs_inloop)
    print(length(unique(val_candidates)))
    print(ncol(similarities2))
  }
  train_drugs <- unique(as.character(colnames(similarities2)))
  return(list(train_drugs,val_candidates))
}
compute_ratio <- function(df){
  binary_ratio <- sum(df$Binary)/nrow(df)*100
  not_sure_ratio <- sum(df$not_sure)/nrow(df)*100
  assay_type_ratio <- nrow(filter(df,assay_type == "non_binding"))/nrow(df)*100
  return(c(binary_ratio,not_sure_ratio,assay_type_ratio))
}
  
similarities <- read_csv("akt-1//data/ecfp_sims.csv") %>% select(-X1)
unique_rdkit <- read_csv("akt-1//data/all_data_rdkit_unique.csv") %>% select(-X1)
result <- restricted_dataset(similarities = similarities,sim_threshold_low = 0.4,sim_threshold_up = 0.7,unique_rdkit = unique_rdkit, ndrugs = 400)

train_drugs <- result[[1]]
val_candidates <- result[[2]]


all_data <- read.csv("akt-1/data/all_data_akt1_unique.csv") %>% select(-X)


train_space <- all_data[which(all_data$rdkit %in% train_drugs),]
val_space <- all_data[which(all_data$rdkit %in% val_candidates),]



train_space_ratio <- compute_ratio(train_space)
val_space_ratio <-  compute_ratio(val_space)

fix_ratio <- function(vs,ts_ratio){
  vs_new <- vs %>% filter(Binary == 1)
  vs_zero <- vs %>% filter(Binary == 0)
  vs_new_ratio <- compute_ratio(vs_new)
  i <- 1
  while ({abs(ts_ratio[1]-vs_new_ratio[1]) > 0.2}){
    if ( i <= length(vs_zero$Binary)){
      vs_new <- bind_rows(vs_new,vs_zero[i,])
      i <- i + 1
      vs_new_ratio <- compute_ratio(vs_new)
      print(vs_new_ratio[1])
    }
  }
  return(vs_new)
}

vs_new <- fix_ratio(val_space,train_space_ratio)
vs_new_ratio <- compute_ratio(vs_new)

write_csv(train_space,"akt-1/data_crossval/fold_6/train_6.csv")
write_csv(vs_new,"akt-1/data_crossval/fold_6/val_cold_6.csv")


