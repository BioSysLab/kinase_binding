library(Rtsne)
library(tidyverse)

embs <- read.csv("C:/Users/user/Documents/kinase_binding/learning/data/p38/split_aveb/fold_0/embeddings_0.csv")
embs <- embs[,-1]
val <- read.csv("C:/Users/user/Documents/kinase_binding/learning/data/p38/split_aveb/fold_0/val_0.csv")
train <- read.csv("C:/Users/user/Documents/kinase_binding/learning/data/p38/split_aveb/fold_0/train_0.csv")

tsne_emb <- Rtsne(scale(embs), dims = 2, perplexity=5, verbose=TRUE, max_iter = 1000,initial_dims = 20,check_duplicates = F)
df_emb_test <- data.frame(V1 = tsne_emb$Y[,1], V2 =tsne_emb$Y[,2], label = as.factor(val$Binary))
gtsne_test <- ggplot(df_emb_test, aes(V1, V2))+
  geom_point(aes(color = label),show.legend = T) + scale_color_discrete()
gtsne

train_embs <- read.csv("C:/Users/user/Documents/kinase_binding/learning/data/p38/split_aveb/fold_0/embeddings_train_0.csv")
train_embs <- train_embs[,-1]

tsne_emb_train <- Rtsne(scale(train_embs), dims = 2, perplexity=100, verbose=TRUE, max_iter = 1000,initial_dims = 50,check_duplicates = F)
df_emb <- data.frame(V1 = tsne_emb_train$Y[,1], V2 =tsne_emb_train$Y[,2], label = as.factor(train$Binary))
gtsne <- ggplot(df_emb, aes(V1, V2))+
  geom_point(aes(color = label),show.legend = T) + scale_color_discrete()
gtsne

require("class")

model1<- knn(train=(train_embs), test=(embs), cl=as.factor(train$Binary), k=300)
tab <- table(model1,as.factor(val$Binary))

accuracy <- function(x){sum(diag(x)/(sum(rowSums(x)))) * 100}
accuracy(tab)


rownames(embs) <- as.character(val$rdkit)
rownames(train_embs) <- as.character(train$rdkit)

pos <- length(which(as.character(train$Binary)=="1"))
pos_id <- which(as.character(train$Binary)=="1")
neg <- length(which(as.character(train$Binary)=="0"))
neg_id <- which(as.character(train$Binary)=="0")

pred_labels <- rep("empty",nrow(val))
for (j in 1:nrow(val)) {
  dist_pos <- NULL
  for (i in 1:pos) {
    dist_pos[i] <- dist(rbind(embs[j,],train_embs[pos_id[i],],method = "euclidian"))[1]
  }
  dist_neg <- NULL
  for (i in 1:neg) {
    dist_neg[i] <- dist(rbind(embs[j,],train_embs[neg_id[i],],method = "euclidian"))[1]
  }
  test <- t.test(dist_pos,dist_neg)
  if (test$p.value<0.01) {
    print(j)
    if (mean(dist_pos)>mean(dist_neg)) {
      pred_labels[j] <- "0"
    } else {
      pred_labels[j] <- "1"
    }
  }
}


pred_labels <- as.data.frame(pred_labels)
pred_labels$true <- as.character(val$Binary)

df <- pred_labels %>% filter(pred_labels != "empty")
length(which(df$pred_labels == df$true))/nrow(df)

test <- rbind(train_embs,embs[5,])

tsne_test <- Rtsne(scale(test), dims = 2, perplexity=50, verbose=TRUE, max_iter = 1000,initial_dims = 50,check_duplicates = F)
df_test <- data.frame(V1 = tsne_test$Y[,1], V2 =tsne_test$Y[,2], label = as.factor(c(train$Binary,2)))
gtest <- ggplot(df_test, aes(V1, V2))+
  geom_point(aes(color = label),show.legend = T) + scale_color_discrete()
gtest




all <- bind_rows(train,val)
all_emb <- bind_rows(train_embs,embs)

all_tsne <- Rtsne(scale(all_emb), dims = 2, perplexity=50, verbose=TRUE, max_iter = 1000,initial_dims = 50,check_duplicates = F)
df_all <- data.frame(V1 = all_tsne$Y[,1], V2 =all_tsne$Y[,2], label = as.factor(all$Binary))
gtest <- ggplot(df_test, aes(V1, V2))+
  geom_point(aes(color = label),show.legend = T) + scale_color_discrete()
gtest

a <- cbind(df_all$V1,df_all$V2)
id <- which(all$rdkit %in% val$rdkit)
model1<- knn(a[-id,], test=a[id,], cl=as.factor(all$Binary[-id]), k=30,use.all = T)
tab <- table(model1,as.factor(val$Binary))
tab
accuracy <- function(x){sum(diag(x)/(sum(rowSums(x)))) * 100}
accuracy(tab)
library(caret)
fitControl <- trainControl(method = "repeatedcv",
                           number = 10,
                           repeats = 10,
                           ## Estimate class probabilities
                           classProbs = TRUE,
                           ## Evaluate performance using 
                           ## the following function
                           summaryFunction = twoClassSummary)

train_embs$labels <- as.character(train$Binary)
train_embs <- train_embs %>% mutate(labels=if_else(labels == "1","positive","negative"))
alexo_tree = train(labels ~ ., 
                   data=train_embs, 
                   method="rf", 
                   trControl = fitControl,metric = "ROC",tuneLength = 10)
alexo_tree

extractPrediction(alexo_tree,
                  testX = embs, testY = NULL,
                  unkX = NULL,
                  verbose = FALSE)

a <- predict(alexo_tree,embs)
a <- as.data.frame(a)
a$true <- val$Binary

a <- a %>% mutate(true = if_else(true == 1,"positive","negative"))

length(which(a$a==a$true))/nrow(a)
