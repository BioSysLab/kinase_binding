setwd("~/Desktop/binding")
library(tidyverse)
library(webchem)
#Stage 1 

#Load jak1 data with IC50

jak1 <- read.csv("Proteins/JAK-1/jak-1_unique_IC50.csv")
jak1 <- jak1 %>% filter(!is.na(standard_value)) %>% filter(assay_description != "")

#Assays for jak1

jak1_assays <- jak1 %>% group_by(title,assay_description)%>%
  summarise(Count = n()) %>% arrange(desc(Count)) 
#jak1_assays <- jak1_assays %>% filter(Count >=6)
not_binding <- c()  #c(7,12,16,19,23,30,31,34,91,102,104,105,129,137) #not binding assays
 
#Keep binding assays

jak1_binding_assays <- jak1_assays 
binding_assays <- jak1_binding_assays$assay_description %>% unique()
jak1_binding <- jak1 %>% filter(assay_description == binding_assays[1])

#Take data for binding assays

for (i in 2:length(binding_assays)) {
  temp <- jak1 %>% filter(assay_description == binding_assays[i])
  jak1_binding <- rbind(jak1_binding,temp)
  rm(temp)
  print(i)
}

#Look for relations

jak1_rel <- jak1_binding %>% group_by(standard_relation)%>%
  summarise(Count = n()) %>% arrange(desc(Count)) 
jak1_binding_clean <- jak1_binding %>% filter(standard_relation == '=')
for (i in 1:length(jak1_binding$standard_relation)) {
  if (jak1_binding[i,12] == '>') {
    if (jak1_binding[i,13] >= 1000) {
      jak1_binding[i,12] <- '='
      jak1_binding_clean <- rbind(jak1_binding_clean,jak1_binding[c(i),])
    }
  }
  else if (jak1_binding[i,12] == '<') {
    if (jak1_binding[i,13] <= 100) {
      jak1_binding[i,12] <- '='
      jak1_binding_clean <- rbind(jak1_binding_clean,jak1_binding[c(i),])
    }
  }
  else if (jak1_binding[i,12] == '<=') {
    jak1_binding[i,12] <- '='
    jak1_binding_clean <- rbind(jak1_binding_clean,jak1_binding[c(i),])
  }
  else if (jak1_binding[i,12] == '>=') {
    jak1_binding[i,12] <- '='
    jak1_binding_clean <- rbind(jak1_binding_clean,jak1_binding[c(i),])
  }
}
jak1_binding_clean <- jak1_binding_clean %>% mutate(standard_value = -log10(standard_value/10^9)) 

###################
rm(jak1,jak1_assays,jak1_binding,jak1_binding_assays,jak1_rel) #free ram
##################


#Stage 2 Add Smiles and RDkit

chembl <- readRDS("RDS/chembl.RDS")
jak1_binding_clean <- left_join(jak1_binding_clean,chembl,by = c("compound_id"="chembl_id")) 

sum(is.na(jak1_binding_clean$canonical_smiles))

#Find not known canonical smiles from inchi key (PUBCHEM)

no_NAs <- jak1_binding_clean %>% filter(!is.na(canonical_smiles))
with_NAs <- jak1_binding_clean %>% filter(is.na(canonical_smiles))
with_NAs$canonical_smiles <- as.character(with_NAs$canonical_smiles)
with_NAs$standard_inchi_key.x <- as.character(with_NAs$standard_inchi_key.x)

a <- cts_convert(query = as.character(with_NAs$standard_inchi_key.x),from = 'inchikey',to='inchi' )


with_NAs$compound_id <- as.character(with_NAs$compound_id)
get_csid(with_NAs$standard_inchi_key.x[1],from = 'inchkey')

?cs_check_key()


sum(is.na(jak1_binding_clean$standard_inchi_key.x))

cir_query(with_NAs$standard_inchi_key.x[1],'smiles',resolver = 'stdinchikey')







jak1_binding_clean <- bind_rows(with_NAs,no_NAs)

unique_binding_smiles <- data.frame(jak1_binding_clean$canonical_smiles %>% unique()) 
colnames(unique_binding_smiles) <-"canonical_smiles"
write_csv(unique_binding_smiles,"jak-1/Binding Assays/unique_binding_canonical_smiles.csv")
write_csv(jak1_binding_clean,"jak-1/Binding Assays/jak1_binding_clean.csv") #first save

rm(no_NAs,with_NAs,jak1_binding,jak1_binding_assays)
rm(jak1_rel)
rm(chembl)
#### Go to python notebook Check_sim.ipynb

jak1_binding_clean <- read.csv("jak-1/Binding Assays/jak1_binding_clean.csv")
smiles_rdkit_binding <- read_csv("jak-1/Binding Assays/smiles_rdkit_binding.csv") %>% select(-X1)

#Load similarity results from python

binding_canonical_smiles_sim <- read_csv("jak-1/Binding Assays/binding_canonical_smiles_sim.csv") %>% select(-X1) #for unique rdkit

# Binary similarities

binding_canonical_smiles_sim <- binding_canonical_smiles_sim >= 0.995 
binding_canonical_smiles_sim <- binding_canonical_smiles_sim + 0

# Change to -1 under main diagonal

binding_canonical_smiles_sim[lower.tri(binding_canonical_smiles_sim,diag = TRUE)] <- -1
binding_canonical_smiles_sim <- data.frame(binding_canonical_smiles_sim)

# Check similars

similars <- 0
comp <- matrix(nrow = 1000,ncol = 2)

for (i in 1:length(binding_canonical_smiles_sim$X0)){
  for (j in i:length(binding_canonical_smiles_sim$X0)) {
    if (i != j){
      if (binding_canonical_smiles_sim[i,j] == 1) {
        similars <- similars + 1
        comp[similars,1] <- i
        comp[similars,2] <- j
      }
    }
  }
  print(i)
}

sim <- data.frame(comp)
sim <- sim[-c(131:1000),]

sim_summary <- sim %>% group_by(X1) %>%
  summarise(Count = n()) %>% arrange(desc(Count))

sim_summary2 <- sim %>% group_by(X2) %>%
  summarise(Count = n()) %>% arrange(desc(Count))

for (i in 1:length(sim$X1)){
  smiles_rdkit_binding$rdkit[sim[i,2]] <- smiles_rdkit_binding$rdkit[sim[i,1]]
  smiles_rdkit_binding$Atoms[sim[i,2]] <- smiles_rdkit_binding$Atoms[sim[i,1]]
  print(i)
}

rdkit_sum <- smiles_rdkit_binding %>% group_by(rdkit)%>%
  summarise(Count = n()) %>% arrange(desc(Count)) #correct after changing rdkits 
new_unique_rdkit <- data.frame(smiles_rdkit_binding$rdkit %>% unique())
colnames(new_unique_rdkit) <- "rdkit"

write_csv(new_unique_rdkit,"jak-1/Binding Assays/rdkit_after_check_sim.csv") ##################!!!!!!!!!!!

jak1_binding_clean <- left_join(jak1_binding_clean,smiles_rdkit_binding,by = "canonical_smiles") #left join after change (rdkit_after_change.csv file)

rm(sim,sim_summary,sim_summary2,smiles_rdkit_binding,rdkit_sum,binding_canonical_smiles_sim,comp)

# Stage 3

# Summary of rdkit on jak1_binding_clean

rdkit_summary <- jak1_binding_clean %>% group_by(rdkit)%>%
  summarise(Count = n()) %>% arrange(desc(Count)) 

# Left join rdkit summary on jak1 data

jak1_binding_clean <- left_join(jak1_binding_clean,rdkit_summary)
jak1_binding_clean$rdkit <- as.character(jak1_binding_clean$rdkit)


# Check Counts 

jak1_binding_clean_clean <- jak1_binding_clean %>% filter(Count == 1)
jak1_binding_clean_2 <- jak1_binding_clean %>% filter(Count == 2)
jak1_binding_clean_other <- jak1_binding_clean %>% filter(Count > 2 )

# Clean for count 2, mean value of IC50

jak1_binding_clean_2 <- jak1_binding_clean_2 %>% group_by(rdkit) %>% mutate(avg = mean(standard_value)) %>% mutate(diff = abs(diff(standard_value))) %>% mutate(percent = diff/avg) %>%
  ungroup()
jak1_binding_clean_2 <- jak1_binding_clean_2 %>% filter(percent < 0.3)
jak1_binding_clean_2 <- jak1_binding_clean_2 %>% mutate(standard_value = avg) %>% select(-c(avg,diff,percent))

jak1_binding_clean_clean <- bind_rows(jak1_binding_clean_clean,jak1_binding_clean_2)

# Clean for count > 2

jak1_binding_clean_other <- jak1_binding_clean_other %>% group_by(rdkit) %>% mutate(avg = mean(standard_value)) %>% mutate(std = abs(sd(standard_value))) %>% mutate(percent = std/avg) %>%
  ungroup()
jak1_binding_clean_other <- jak1_binding_clean_other %>% filter(percent <0.3)
jak1_binding_clean_other <- jak1_binding_clean_other %>% mutate(standard_value = avg) %>% select(-c(avg,std,percent)) 
  

jak1_binding_clean_clean <- bind_rows(jak1_binding_clean_clean,jak1_binding_clean_other)
jak1_binding_clean_clean <- jak1_binding_clean_clean %>% filter(Atoms >=10 & Atoms <= 70)
write_csv(jak1_binding_clean_clean,"jak-1/Binding Assays/jak1_binding_clean_clean.csv")

uni <- data.frame(jak1_binding_clean_clean$rdkit %>% unique())
colnames(uni) <- "rdkit"
write_csv(uni,"jak-1/Binding Assays/unique_binding_rdkit.csv")



data <- jak1_binding_clean_clean %>% select(rdkit,standard_value) %>% unique()

write_csv(data,"jak-1/Binding Assays/train_binding.csv")

rm(binding_canonical_smiles_sim,comp,data,new_unique_rdkit,jak1_binding_clean,jak1_binding_clean_2,jak1_binding_clean_other,jak1_binding_clean_clean,rdkit_sum,rdkit_summary
   ,sim,smiles_rdkit_binding,uni,unique_binding_smiles,binding_assays,similars)
############################################################## Not Binding Assays

# Stage 1

### Keep not binding assays

jak1_nb_assays <- jak1_assays[not_binding,]
not_binding_assays <- jak1_nb_assays$assay_description %>% unique()
jak1_not_binding <- jak1 %>% filter(assay_description == not_binding_assays[1])

### Take data for not binding assays

for (i in 2:length(not_binding_assays)) {
  temp <- jak1 %>% filter(assay_description == not_binding_assays[i])
  jak1_not_binding <- rbind(jak1_not_binding,temp)
  rm(temp)
  print(i)
}

#Look for relations

jak1_rel_not <- jak1_not_binding %>% group_by(standard_relation)%>%
  summarise(Count = n()) %>% arrange(desc(Count)) 
jak1_not_binding_clean <- jak1_not_binding %>% filter(standard_relation == '=')

for (i in 1:length(jak1_not_binding$standard_relation)) {
  if (jak1_not_binding[i,12] == '>') {
    if (jak1_not_binding[i,13] >= 1000) {
      jak1_not_binding[i,12] <- '='
      jak1_not_binding_clean <- rbind(jak1_not_binding_clean,jak1_not_binding[c(i),])
    }
  }
  else if (jak1_not_binding[i,12] == '<') {
    if (jak1_not_binding[i,13] <= 100) {
      jak1_not_binding[i,12] <- '='
      jak1_not_binding_clean <- rbind(jak1_not_binding_clean,jak1_not_binding[c(i),])
    }
  }
}

jak1_not_binding_clean <- jak1_not_binding_clean %>% mutate(standard_value = -log10(standard_value/10^9))
rm(jak1_not_binding,jak1_rel_not,jak1_assays,jak1_nb_assays,jak1)


write.csv(jak1_not_binding_clean,"jak1_not_binding_clean.csv")
###########################################################

# Stage 2

chembl <- readRDS("chembl.RDS")
jak1_not_binding_clean <- left_join(jak1_not_binding_clean,chembl,by = c("compound_id"="chembl_id"))

sum(is.na(jak1_not_binding_clean$canonical_smiles)) # All smiles are known.
rm(chembl)

unique_not_binding_canonical_smiles <- data.frame(jak1_not_binding_clean$canonical_smiles %>% unique()) 

colnames(unique_not_binding_canonical_smiles) <-"canonical_smiles"
write_csv(unique_not_binding_canonical_smiles,"Not Binding Assays/unique_not_binding_canonical_smiles.csv")
write_csv(jak1_not_binding_clean,"Not Binding Assays/jak1_not_binding_clean.csv") #first save

#### Go to python notebook Check_sim.ipynb
jak1_not_binding_clean <- read.csv("jak1_not_binding_clean.csv")
smiles_rdkit_not_binding <- read_csv("Not Binding Assays/smiles_rdkit_not_binding.csv") %>% select(-X1)

#Load similarity results from python

similarities <- read_csv("Not Binding Assays/not_binding_canonical_smiles_sim.csv") %>% select(-X1) #for unique rdkit

# Binary similarities

similarities <- similarities >= 0.995 
similarities <- similarities + 0

# Change to -1 under main diagonal

similarities[lower.tri(similarities,diag = TRUE)] <- -1
similarities <- data.frame(similarities)

# Check similars

similars <- 0
comp <- matrix(nrow = 200,ncol = 2)

for (i in 1:length(similarities$X0)){
  for (j in i:length(similarities$X0)) {
    if (i != j){
      if (similarities[i,j] == 1) {
        similars <- similars + 1
        comp[similars,1] <- i
        comp[similars,2] <- j
      }
    }
  }
  print(i)
}

sim <- data.frame(comp)
sim <- sim[-c(18:200),]

for (i in 1:17){
  smiles_rdkit_not_binding$rdkit[sim[i,2]] <- smiles_rdkit_not_binding$rdkit[sim[i,1]]
  smiles_rdkit_not_binding$Atoms[sim[i,2]] <- smiles_rdkit_not_binding$Atoms[sim[i,1]]
  print(i)
}

rdkit_sum_not_binding <- smiles_rdkit_not_binding %>% group_by(rdkit)%>%
  summarise(Count = n()) %>% arrange(desc(Count)) #correct after changing rdkits 


new_unique_rdkit_not_binding <- data.frame(smiles_rdkit_not_binding$rdkit %>% unique())
colnames(new_unique_rdkit_not_binding) <- "rdkit"

write_csv(new_unique_rdkit_not_binding,"Not Binding Assays/rdkit_not_binding_after_check_sim.csv") ##################!!!!!!!!!!!

jak1_not_binding_clean <- left_join(jak1_not_binding_clean,smiles_rdkit_not_binding,by = "canonical_smiles")

rm(comp,new_unique_rdkit_not_binding,rdkit_sum_not_binding,sim,similarities,smiles_rdkit_not_binding,unique_not_binding_canonical_smiles,not_binding
   ,not_binding_assays,similars)
# Stage 3

# Summary of rdkit on jak1_binding_clean

rdkit_summary_not_binding <- jak1_not_binding_clean %>% group_by(rdkit)%>%
  summarise(Count = n()) %>% arrange(desc(Count)) 

# Left join rdkit summary on jak1 data

jak1_not_binding_clean <- left_join(jak1_not_binding_clean,rdkit_summary_not_binding)
jak1_not_binding_clean$rdkit <- as.character(jak1_not_binding_clean$rdkit)


# Check Counts 

jak1_not_binding_clean_clean <- jak1_not_binding_clean %>% filter(Count == 1)
jak1_not_binding_clean_2 <- jak1_not_binding_clean %>% filter(Count == 2)
jak1_not_binding_clean_other <- jak1_not_binding_clean %>% filter(Count > 2 )

# Clean for count 2, mean value of IC50

jak1_not_binding_clean_2 <- jak1_not_binding_clean_2 %>% group_by(rdkit) %>% mutate(avg = mean(standard_value)) %>% mutate(diff = abs(diff(standard_value))) %>% mutate(percent = diff/avg) %>%
  ungroup()
jak1_not_binding_clean_2 <- jak1_not_binding_clean_2 %>% filter(percent < 0.3)
jak1_not_binding_clean_2 <- jak1_not_binding_clean_2 %>% mutate(standard_value = avg) %>% select(-c(avg,diff,percent))

jak1_not_binding_clean_clean <- bind_rows(jak1_not_binding_clean_clean,jak1_not_binding_clean_2)

# Clean for count > 2

jak1_not_binding_clean_other <- jak1_not_binding_clean_other %>% group_by(rdkit) %>% mutate(avg = mean(standard_value)) %>% mutate(std = abs(sd(standard_value))) %>% mutate(percent = std/avg) %>%
  ungroup()
jak1_not_binding_clean_other <- jak1_not_binding_clean_other %>% filter(percent <0.3)
jak1_not_binding_clean_other <- jak1_not_binding_clean_other %>% mutate(standard_value = avg) %>% select(-c(avg,std,percent)) 


jak1_not_binding_clean_clean <- bind_rows(jak1_not_binding_clean_clean,jak1_not_binding_clean_other)


jak1_not_binding_clean_clean <- jak1_not_binding_clean_clean %>% filter(Atoms >=10 & Atoms <= 70)
write_csv(jak1_not_binding_clean_clean,"Not Binding Assays/jak1_not_binding_clean_clean.csv")

uni <- data.frame(jak1_not_binding_clean_clean$rdkit %>% unique())
colnames(uni) <- "rdkit"
write_csv(uni,"Not Binding Assays/unique_not_binding_rdkit.csv")



data <- jak1_not_binding_clean_clean %>% select(rdkit,standard_value) %>% unique()

write_csv(data,"Not Binding Assays/train_not_binding.csv")

#####################################
# Load Cleaned data

#Binding
training_binding <- read.csv("jak-1/Binding Assays/train_binding.csv")

#Histogram of standard value
hist(training_binding$standard_value)

# Non binding
training_not_binding <- read.csv("Not Binding Assays/train_not_binding.csv")


# Binary standard value and label assay type

training_binding$Binary <- training_binding$standard_value >= 7
training_binding$Binary <- training_binding$Binary + 0
ratio_binding <- sum(training_binding$Binary == 1)/length(training_binding$Binary)*100
training_binding$assay_type <- "binding"

training_binding$not_sure <- (training_binding$standard_value > 6 & training_binding$standard_value <7.5) 
training_binding$not_sure <- training_binding$not_sure + 0
ratio_binding_not_sure <- sum(training_binding$not_sure == 1)/length(training_binding$Binary)*100

training_not_binding$Binary <- training_not_binding$standard_value >= 6.5
training_not_binding$Binary <- training_not_binding$Binary + 0
ratio_not_binding <- sum(training_not_binding$Binary == 1)/length(training_not_binding$Binary)*100
training_not_binding$assay_type <- "non_binding"

training_not_binding$not_sure <- (training_not_binding$standard_value > 6 & training_not_binding$standard_value <7) 
training_not_binding$not_sure <- training_not_binding$not_sure + 0


all_data <- training_binding #bind_rows(training_binding,training_not_binding)
write_csv(all_data,"jak-1/data/all_data_jak1_not_unique.csv") # σωστό



all_summary <- all_data %>% group_by(rdkit)%>%
  summarise(Count = n()) %>% arrange(desc(Count)) #correct after changing rdkits 

df <- left_join(all_data,all_summary,by = "rdkit")

duplicates <- df %>% filter(Count == 2)
binding_df <- duplicates %>% filter(assay_type != "non_binding")
binding_df_rdkit <- binding_df$rdkit %>% unique()

unique_df <- df %>% filter(Count == 1)
data_jak1_unique <- bind_rows(unique_df,binding_df)
data_jak1_unique <- data_jak1_unique %>% select(-Count)
bin_ratio <- sum(data_jak1_unique$Binary == 1)/length(data_jak1_unique$Binary)*100

not_sure_ratio <- sum(data_jak1_unique$not_sure)/nrow(all_data)*100
assay_type_ratio <- nrow(filter(data_jak1_unique,assay_type == "non_binding"))/nrow(data_jak1_unique)*100



write.csv(data_jak1_unique,"jak-1/data/all_data_jak1_unique.csv")
data_rdkit <- data.frame(data_jak1_unique$rdkit)
colnames(data_rdkit) <- "rdkit"
write.csv(data_rdkit,"jak-1/data/all_data_rdkit_unique.csv")



