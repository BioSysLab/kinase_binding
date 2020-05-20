library(tidyverse)

train <- read.csv("C:/Users/user/Documents/kinase_binding/learning/data/p38/split_aveb/fold_0/train_0.csv")

trainsmiles <- as.character(train$rdkit)
#write.csv(trainsmiles,"C:/Users/user/Documents/kinase_binding/learning/data/p38/split_aveb/fold_0/trainsmiles.csv")

sims <- read.csv("C:/Users/user/Documents/kinase_binding/learning/data/p38/split_aveb/fold_0/trainsims.csv")
sims <- sims[,-1]
rownames(sims) <- trainsmiles
colnames(sims) <- trainsmiles

dist <- 1 - sims
dist_mat <- as.matrix(dist)
n_max <- 5
dist_neg_thresh <- 0.5
dist_pos_thresh <- 0.85
prop_hard <- 0.5
result <- NULL
for (i in 1:nrow(train)) {
  # label of the anchor
  label <- train$Binary[i]
  # index of drugs with same label as anchor
  id_pos <- which(train$Binary == label)
  # index of negatives
  id_neg <- which(train$Binary != label)
  

  dist_pos <- dist_mat[i,id_pos]
  hist(dist_pos)
  dist_neg <- dist_mat[i,id_neg]
  hist(dist_neg)
  
  #hard triplets margined
  
  neg_candidates <- dist_neg[which(dist_neg <= dist_neg_thresh)]
  pos_candidates <- dist_pos[which(dist_pos >= dist_pos_thresh)]
  k <- 0.05
  while (length(neg_candidates)==0) {
    neg_candidates <- dist_neg[which(dist_neg <= (dist_neg_thresh+k))]
    k <- k+0.05
  }
  # all neg and all pos
  
  neg_all <- names(dist_neg)
  pos_all <- names(dist_pos)
  
  # maximum allowed hard triplets per drug
  
  if (length(neg_candidates)>n_max) {
    neg <- sample(names(neg_candidates),n_max)
  } else {
    neg <- names(neg_candidates)
  }
  
  pos <- sample(names(pos_candidates),length(neg))
  triplets_hard <- data.frame(matrix(rep("empty",length(neg)),nrow = length(neg),ncol = 4))
  colnames(triplets_hard) <- c("A","P","N","margin")
  triplets_hard$A <- train$rdkit[i]
  triplets_hard$P <- pos
  triplets_hard$N <- neg
  triplets_hard$margin <- paste0("margin = ",dist_pos_thresh - dist_neg_thresh + k - 0.05)
  
  neg_left <- neg_all[-which(neg_all %in% neg)]
  pos_left <- pos_all[-which(pos_all %in% pos)]
  
  # random triplets
  nrandom <- round((1/prop_hard)-1) * nrow(triplets_hard)
  neg_random <- sample(neg_left,nrandom)
  pos_random <- sample(pos_left,nrandom)
  
  triplets_random <- data.frame(matrix(rep("empty",length(nrandom)),nrow = nrandom,ncol = 4))
  colnames(triplets_random) <- c("A","P","N","margin")
  triplets_random$A <- train$rdkit[i]
  triplets_random$P <- pos_random
  triplets_random$N <- neg_random
  triplets_random$margin <- "random"
  
  triplets <- bind_rows(triplets_hard,triplets_random)
  result[[i]] <- triplets
}

triplets <- result[[1]]
for (i in 2:length(result)) {
  df <- result[[i]]
  triplets <- bind_rows(triplets,df)
}

write.csv(triplets,"C:/Users/user/Documents/kinase_binding/learning/data/p38/split_aveb/fold_0/triplets_train_0.csv")
