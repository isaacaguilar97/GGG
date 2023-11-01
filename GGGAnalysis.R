library(tidyverse)
library(vroom)
library(tidymodels)

# Load the data -----------------------------------------------------------
# setwd('~/College/Stat348/GGG')

# Load data
missSet <- vroom('./trainWithMissingValues.csv')
trainSet <- vroom('./train.csv')

# dplyr::glimpse(ggg)
# plot_missing(ggg)

# Set my recipe 
my_recipe <- recipe(type~., data=missSet) %>%
  step_impute_knn(hair_length, impute_with = imp_vars(rotting_flesh, bone_length, has_soul), neighbors = 5) %>%
  step_impute_knn(rotting_flesh, impute_with = imp_vars(hair_length, bone_length, has_soul), neighbors = 5) %>%
  step_impute_knn(bone_length, impute_with = imp_vars(hair_length, has_soul), neighbors = 5)
  

# apply the recipe to your data
prep <- prep(my_recipe)
imputedSet <- bake(prep, new_data = ggg)

# Calculate RMSE of the imputations 
rmse_vec(trainSet[is.na(missSet)], imputedSet[is.na(missSet)])
