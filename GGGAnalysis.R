library(doParallel)

num_cores <- parallel::detectCores() #How many cores do I have?
cl <- makePSOCKcluster(num_cores)
registerDoParallel(cl)

library(tidyverse)
library(vroom)
library(tidymodels)
library(naivebayes)
library(discrim)
library(DataExplorer)
library(bonsai)
library(lightgbm)
library(embed)



# Load the data -----------------------------------------------------------
# setwd('~/College/Stat348/GGG')

# Load data
missSet <- vroom('./trainWithMissingValues.csv')
trainSet <- vroom('./train.csv')
testSet <- vroom('./test.csv')


# # EDA ---------------------------------------------------------------------
# 
dplyr::glimpse(trainSet)
# plot_missing(trainSet)
# plot_correlation(trainSet)
# plot_histogram(trainSet)
# GGally::ggpairs(trainSet)
# 
# # Set my recipe 
# my_recipe <- recipe(type~., data=trainSet) %>%
#   step_dummy(all_nominal_predictors()) %>%
#   step_normalize(all_numeric_predictors()) %>%
#   step_pca(all_predictors(), threshold=.9)
#   
#   
# 
# # apply the recipe to your data
# prep <- prep(my_recipe)
# imputedSet <- bake(prep, new_data = trainSet)
# 
# 
# # Calculate RMSE of the imputations   
# rmse_vec(trainSet[is.na(missSet)], imputedSet[is.na(missSet)])

# # Naive Bayes -------------------------------------------------------------
# 
# ## nb model
# nb_model <- naive_Bayes(Laplace=tune(), smoothness=tune()) %>%
#   set_mode("classification") %>%
#   set_engine("naivebayes") # install discrim library for the naivebayes eng
# 
# 
# # my_recipe_prep <- prep(my_recipe, data = amazon_train)
# # baked_data <- bake(my_recipe, new_data = NULL)
# 
# nb_wf <- workflow() %>%
#   add_recipe(my_recipe) %>%
#   add_model(nb_model)
# 
# ## Tune smoothness and Laplace here
# tuning_grid <- grid_regular(smoothness(),
#                             Laplace(),
#                             levels = 5)
# 
# ## Set up K-fold CV
# folds <- vfold_cv(trainSet, v = 5, repeats=1)
#
# ## Cross Validation
# CV_results <- nb_wf %>%
#   tune_grid(resamples=folds,
#             grid=tuning_grid,
#             metrics=metric_set(accuracy))
# 
# ## Find best tuning parameters
# bestTune <- CV_results %>%
#   select_best("accuracy")
# 
# 
# ## Predict
# predictions <- final_wf %>%
#   predict(new_data = testSet, type = "class")
# 
# #save(file="./MyFile.RData", list=c("amazon_predictions", "final_wf", "bestTune", "CV_results"))
# 
# # Format table
# testSet$type <- predictions$.pred_class
# results <- testSet %>%
#   select(id, type)
# 
# # get csv file
# vroom_write(results, 'GGGPredsnb.csv', delim = ",")
# 

# Neural Networks ---------------------------------------------------------

# nn_recipe <- recipe(type~., data=trainSet) %>%
#   step_dummy(all_nominal_predictors()) %>%
#   step_range(all_numeric_predictors(), min=0, max=1) #scale to [0,1]
# 
# # nn_recipe <- recipe(formula=, data=) %>% 
# #   update_role(id, new_role="id") %>% 
# #   step_...() %>% ## Turn color to factor then dummy encode color
# #   step_range(all_numeric_predictors(), min=0, max=1) #scale to [0,1]
# 
# nn_model <- mlp(hidden_units = tune(),
#                 epochs = 50 #or 100 or 250
#                 ) %>%
#   set_engine("nnet") %>% #verbose = 0 prints off less (or nnet)
#   set_mode("classification")
# 
# # Workflow
# nn_wf <- workflow() %>%
#   add_recipe(nn_recipe) %>%
#   add_model(nn_model)
# 
# # Tune
# nn_tuneGrid <- grid_regular(hidden_units(range=c(1, 10)),
#                             levels=10)
# 
# ## Set up K-fold CV
# folds <- vfold_cv(trainSet, v = 10, repeats=1)
# 
# # Cross Validation
# tuned_nn <- nn_wf %>%
#   tune_grid(resamples=folds,
#             grid=nn_tuneGrid,
#             metrics=metric_set(accuracy))
# 
# ## Find best tuning parameters
# bestTune <- tuned_nn  %>%
#   select_best("accuracy")
# 
# # Finalize Workflow
# final_wf <- nn_wf %>%
#   finalize_workflow(bestTune) %>%
#   fit(data=trainSet)
# 
# ## Predict
# predictions <- final_wf %>%
#   predict(new_data = testSet, type = "class")
# 
# # Format table
# testSet$type <- predictions$.pred_class
# results <- testSet %>%
#   select(id, type)
# 
# # get csv file
# vroom_write(results, 'GGGPredsnn.csv', delim = ",")
# 
# # Plot graph
# graph <- tuned_nn %>% collect_metrics() %>%
#   filter(.metric=="accuracy") %>%
#   ggplot(aes(x=hidden_units, y=mean)) + geom_line()
# graph
# save(file="./MyFile.RData", list=c("predictions", "tuned_nn"))


# BOOST TREES -------------------------------------------------------------
bt_recipe <- recipe(type~., data=trainSet) %>%
  step_dummy(all_nominal_predictors())

boost_model <- boost_tree(tree_depth=tune(),
                          trees=tune(),
                          learn_rate=tune()) %>%
  set_engine("lightgbm") %>% #or "xgboost" but lightgbm is faster
  set_mode("classification")

# Workflow
bt_wf <- workflow() %>%
  add_recipe(bt_recipe) %>%
  add_model(boost_model)

# Tune
tuneGrid <- grid_regular(tree_depth(), 
                            trees(),
                            learn_rate(),
                            levels = 5)

## Set up K-fold CV
folds <- vfold_cv(trainSet, v = 5, repeats=1)

# Cross Validation
tuned_bt <- bt_wf %>%
  tune_grid(resamples=folds,
            grid=tuneGrid,
            metrics=metric_set(accuracy))

## Find best tuning parameters
bestTune <- tuned_bt  %>%
  select_best("accuracy")

# Finalize Workflow
final_wf <- bt_wf %>%
  finalize_workflow(bestTune) %>%
  fit(data=trainSet)

## Predict
predictions <- final_wf %>%
  predict(new_data = testSet, type = "class")

# Format table
testSet$type <- predictions$.pred_class
results <- testSet %>%
  select(id, type)

# get csv file
vroom_write(results, 'GGGPredsbt.csv', delim = ",")

# # BART -------------------------------------------------------------------
# bt_recipe <- recipe(type~., data=trainSet) %>%
#   step_dummy(all_nominal_predictors())
# 
# bart_model <- bart(trees=tune()) %>% # BART figures out depth and learn_rate9
#   set_engine("dbarts") %>% # might need to install10
#   set_mode("classification")
# 
# 
# # Workflow
# bt_wf <- workflow() %>%
#   add_recipe(bt_recipe) %>%
#   add_model(bart_model)
# 
# # Tune
# tuneGrid <- grid_regular(trees(),
#                          levels = 5)
# 
# ## Set up K-fold CV
# folds <- vfold_cv(trainSet, v = 5, repeats=1)
# 
# # Cross Validation
# tuned_bt <- bt_wf %>%
#   tune_grid(resamples=folds,
#             grid=tuneGrid,
#             metrics=metric_set(accuracy))
# 
# ## Find best tuning parameters
# bestTune <- tuned_bt  %>%
#   select_best("accuracy")
# 
# # Finalize Workflow
# final_wf <- bt_wf %>%
#   finalize_workflow(bestTune) %>%
#   fit(data=trainSet)
# 
# ## Predict
# predictions <- final_wf %>%
#   predict(new_data = testSet, type = "class")
# 
# # Format table
# testSet$type <- predictions$.pred_class
# results <- testSet %>%
#   select(id, type)
# 
# # get csv file
# vroom_write(results, 'GGGPredsbart.csv', delim = ",")

# stopCluster(cl)
