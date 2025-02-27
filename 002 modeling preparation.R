
# Load Required Packages
required_packages <- c(
  "tidyverse", "xgboost", "ranger", "lightgbm", "glmnet",
  "doParallel", "foreach", "cluster", "caret", "Metrics", "doSNOW", "progress", "reshape2", "ggplot2", "LiblineaR"
)

installed_packages <- rownames(installed.packages())

for (pkg in required_packages) {
  if (!pkg %in% installed_packages)
    install.packages(pkg)
}

# Load packages
library(tidyverse)
library(xgboost)
library(ranger)
library(lightgbm)
library(glmnet)
library(doParallel)
library(foreach)
library(cluster)
library(caret)
library(Metrics)
library(doSNOW)
library(progress)
library(reshape2)
library(ggplot2)
library(LiblineaR)

# Data Preparation
data <- daneOK %>% select(
  class4, lon, lat, locPdens.s, locAggAgri.s, locAggProd.s, locAggConstr.s, locAggServ.s,
  locHH.s, locHightech.s, locBIG.s, locAggTOTAL.s, locLQ.s, 
  dist_core_10, dist_core_25, dist_core_50, 
  dist_midsize_10, dist_midsize_25, dist_midsize_50,
  dist_regional_10, dist_regional_25, dist_regional_50,
  dist_localbig_10, dist_localbig_25, dist_localbig_50,
  dist_localsmall_10, dist_localsmall_25, dist_localsmall_50,
  COREfirms, COREpopul
)

# Convert target variable to factor
data$class4 <- as.factor(data$class4)

# Create Spatial Folds Using K-means Clustering
set.seed(123)  # For reproducibility

# Number of spatial folds
k <- 5

# Perform K-means clustering
coords <- data %>% select(lon, lat)
km <- kmeans(coords, centers = k, nstart = 25)

# Assign fold numbers to the dataset
data$fold <- km$cluster

# Define Feature Columns
# Exclude 'class4', 'lon', 'lat', and 'fold' from features
feature_cols <- setdiff(colnames(data), c('class4', 'lon', 'lat', 'fold'))

# Ensure consistent factor levels
class_levels <- levels(data$class4)

# Create Directory to Save Results
dir.create("model_results", showWarnings = FALSE)

# Cross-validation Loop with Progress Messages and Saving Intermediate Results
results_list <- list()

for (current_fold in 1:k) {
  cat("Processing fold", current_fold, "of", k, "\n")
  
  # Split data into training and testing sets based on folds
  train_data <- data %>% filter(fold != current_fold)
  test_data <- data %>% filter(fold == current_fold)
  
  # Prepare feature matrices and target vectors
  x_train <- as.matrix(train_data[, feature_cols])
  y_train <- factor(train_data$class4, levels = class_levels)
  
  x_test <- as.matrix(test_data[, feature_cols])
  y_test <- factor(test_data$class4, levels = class_levels)
  
  # Initialize a data frame to store performance metrics for this fold
  fold_results <- data.frame(fold = current_fold)
  
  ### 1. XGBoost ###
  try({
    cat("Training XGBoost model for fold", current_fold, "\n")
    dtrain <- xgb.DMatrix(data = x_train, label = as.numeric(y_train) - 1)
    
    xgb_params <- list(
      objective = "multi:softmax",
      num_class = length(class_levels),
      eval_metric = "mlogloss",
      verbosity = 0
    )
    
    xgb_model <- xgb.train(
      params = xgb_params,
      data = dtrain,
      nrounds = 100,
      verbose = 0
    )
    
    xgb_preds <- predict(xgb_model, x_test)
    xgb_preds <- factor(xgb_preds + 1, levels = seq_along(class_levels), labels = class_levels)
    
    # Calculate accuracy
    xgb_accuracy <- mean(xgb_preds == y_test)
    fold_results$xgb_accuracy <- xgb_accuracy
    
    # Save results
    xgb_results <- list(
      fold = current_fold,
      predictions = xgb_preds,
      accuracy = xgb_accuracy
    )
    saveRDS(xgb_results, file = paste0("model_results/xgb_results_fold_", current_fold, ".rds"))
    
    cat("XGBoost model completed for fold", current_fold, "with accuracy:", xgb_accuracy, "\n")
  }, silent = FALSE)
  
  ### 2. Random Forest (using ranger) ###
  try({
    cat("Training Random Forest model for fold", current_fold, "\n")
    rf_model <- ranger(
      formula = class4 ~ ., 
      data = train_data[, c('class4', feature_cols)], 
      num.trees = 100,
      classification = TRUE,
      verbose = FALSE,
      probability = FALSE
    )
    
    rf_preds <- predict(rf_model, data = test_data[, feature_cols])$predictions
    rf_preds <- factor(rf_preds, levels = class_levels)
    
    rf_accuracy <- mean(rf_preds == y_test)
    fold_results$rf_accuracy <- rf_accuracy
    
    # Save results
    rf_results <- list(
      fold = current_fold,
      predictions = rf_preds,
      accuracy = rf_accuracy
    )
    saveRDS(rf_results, file = paste0("model_results/rf_results_fold_", current_fold, ".rds"))
    
    cat("Random Forest model completed for fold", current_fold, "with accuracy:", rf_accuracy, "\n")
  }, silent = FALSE)
  
  ### 3. LightGBM ###
  try({
    cat("Training LightGBM model for fold", current_fold, "\n")
    lgb_train <- lgb.Dataset(data = x_train, label = as.numeric(y_train) - 1)
    
    lgb_params <- list(
      objective = "multiclass",
      num_class = length(class_levels),
      metric = "multi_logloss",
      verbosity = -1
    )
    
    lgb_model <- lgb.train(
      params = lgb_params,
      data = lgb_train,
      nrounds = 100,
      verbose = -1
    )
    
    lgb_preds <- predict(lgb_model, x_test)
    lgb_preds <- max.col(matrix(lgb_preds, ncol = length(class_levels), byrow = TRUE))
    lgb_preds <- factor(lgb_preds, levels = seq_along(class_levels), labels = class_levels)
    
    lgb_accuracy <- mean(lgb_preds == y_test)
    fold_results$lgb_accuracy <- lgb_accuracy
    
    # Save results
    lgb_results <- list(
      fold = current_fold,
      predictions = lgb_preds,
      accuracy = lgb_accuracy
    )
    saveRDS(lgb_results, file = paste0("model_results/lgb_results_fold_", current_fold, ".rds"))
    
    cat("LightGBM model completed for fold", current_fold, "with accuracy:", lgb_accuracy, "\n")
  }, silent = FALSE)
  
  ### 4. LiblineaR ###
  try({
    cat("Training LiblineaR model for fold", current_fold, "\n")
    
    # Set type = 0 for multi-class classification using L2-regularized logistic regression (primal)
    liblinear_model <- LiblineaR(
      data = x_train_scaled,
      target = y_train_numeric,
      type = 0,
      cost = 1,
      bias = TRUE,
      verbose = FALSE
    )
    
    # Predict on test data
    liblinear_preds <- predict(liblinear_model, x_test_scaled)$predictions
    liblinear_preds <- factor(liblinear_preds, levels = 1:length(class_levels), labels = class_levels)
    
    # Calculate accuracy
    liblinear_accuracy <- mean(liblinear_preds == y_test)
    
    # Save results
    fold_results <- list(
      fold = current_fold,
      predictions = liblinear_preds,
      actuals = y_test,
      accuracy = liblinear_accuracy
    )
    saveRDS(fold_results, file = paste0("model_results/liblinear_results_fold_", current_fold, ".rds"))
    
    cat("LiblineaR model completed for fold", current_fold, "with accuracy:", liblinear_accuracy, "\n")
    
    # Append to results list
    results_list[[current_fold]] <- data.frame(fold = current_fold, accuracy = liblinear_accuracy)
    
  }, silent = FALSE)
  
  # Save fold results
  saveRDS(fold_results, file = paste0("model_results/fold_results_", current_fold, ".rds"))
  
  # Append to results list
  results_list[[current_fold]] <- fold_results
  
  cat("Completed fold", current_fold, "\n\n")
}


load_model_results <- function(model_name) {
  result_files <- list.files("model_results", pattern = paste0(model_name, "_results_fold_\\d+\\.rds"), full.names = TRUE)
  if (length(result_files) == 0) {
    return(data.frame(fold = integer(0), accuracy = numeric(0)))
  }
  results_list <- lapply(result_files, readRDS)
  results_df <- do.call(rbind, lapply(results_list, function(res) data.frame(fold = res$fold, accuracy = res$accuracy)))
  return(results_df)
}

# Load results for each model
xgb_results_df <- load_model_results("xgb")
rf_results_df <- load_model_results("rf")
lgb_results_df <- load_model_results("lgb")
liblinear_results_df <- load_model_results("liblinear")

#liblinear_results_df <- results_df %>% mutate(model = "LiblineaR")
#glmnet_results_df <- load_model_results("glmnet")

# Combine all results into one data frame for plotting
all_results <- rbind(
  data.frame(fold = xgb_results_df$fold, accuracy = xgb_results_df$accuracy, model = "XGBoost"),
  data.frame(fold = rf_results_df$fold, accuracy = rf_results_df$accuracy, model = "Random Forest"),
  data.frame(fold = lgb_results_df$fold, accuracy = lgb_results_df$accuracy, model = "LightGBM"),
  data.frame(fold = liblinear_results_df$fold, accuracy = liblinear_results_df$accuracy, model = "LiblineaR")
  #data.frame(fold = glmnet_results_df$fold, accuracy = glmnet_results_df$accuracy, model = "GLMNet")
)

# Plot accuracy across folds for each model
ggplot(all_results, aes(x = as.factor(fold), y = accuracy, color = model, group = model)) +
  geom_line() +
  geom_point() +
  labs(title = "Model Accuracy Across Folds", x = "Fold", y = "Accuracy") +
  theme_minimal()

# Final Mean Accuracies
mean_accuracies <- all_results %>%
  group_by(model) %>%
  summarise(mean_accuracy = mean(accuracy, na.rm = TRUE))

print("Final Mean Accuracies for Each Model:")
print(mean_accuracies)




#data$fold - >  plot with this


# Prepare the full dataset
x_all <- data[, feature_cols]
y_all <- factor(data$class4, levels = class_levels)


# Prepare the full dataset
x_all <- data[, feature_cols]
y_all <- factor(data$class4, levels = class_levels)


# Retrain the best model - XGBoost

# XGBoost
y_all_numeric <- as.numeric(y_all) - 1
dall <- xgb.DMatrix(data = as.matrix(x_all), label = y_all_numeric)

xgb_params <- list(
  objective = "multi:softmax",
  num_class = length(class_levels),
  eval_metric = "mlogloss",
  verbosity = 0
)

xgb_final_model <- xgb.train(
  params = xgb_params,
  data = dall,
  nrounds = 100,
  verbose = 0
)

# Save the model
saveRDS(xgb_final_model, file = "final_models/xgb_final_model.rds")

# Feature Importance
importance_matrix <- xgb.importance(feature_names = colnames(x_all), model = xgb_final_model)
print("Feature Importance:")
print(importance_matrix)

# Confusion Matrix
xgb_preds <- predict(xgb_final_model, as.matrix(x_all))
xgb_preds <- factor(xgb_preds + 1, levels = seq_along(class_levels), labels = class_levels)

confusion_mat <- confusionMatrix(xgb_preds, y_all, mode = "everything")
print("Confusion Matrix:")
print(confusion_mat)



# importance
# get the feature real names
names <-  colnames(data[,-1])
# compute feature importance matrix
importance_matrix = xgb.importance(feature_names = names, model = bst_model)
head(importance_matrix)
importance_matrix

gp = xgb.ggplot.importance(importance_matrix)
print(gp)


# RETRAIN XBOOST FOR SHINY
# Prepare the data
x_all <- as.matrix(data[, feature_cols])
y_all <- as.numeric(data$class4) - 1  # labels starting from 0

# Define parameters
xgb_params <- list(
  objective = "multi:softprob",  # Change to 'multi:softprob' to get probabilities
  num_class = length(levels(data$class4)),
  eval_metric = "mlogloss",
  verbosity = 0
)

# Train the model
xgb_final_model <- xgb.train(
  params = xgb_params,
  data = xgb.DMatrix(data = x_all, label = y_all),
  nrounds = 100,
  verbose = 0
)

# Save the model
saveRDS(xgb_final_model, file = "final_models/xgb_final_model_labels.rds")



### HERE TO RERUN LiblineaR #################

#install.packages("LiblineaR")
library(LiblineaR)

# Ensure consistent factor levels
class_levels <- levels(data$class4)

# Initialize a list to store results
results_list <- list()

set.seed(123)

# Cross-validation loop
for (current_fold in 1:k) {
  cat("Processing fold", current_fold, "of", k, "\n")
  
  # Split data into training and testing sets based on folds
  train_data <- data %>% filter(fold != current_fold)
  test_data <- data %>% filter(fold == current_fold)
  
  # Prepare feature matrices and target vectors
  x_train <- as.matrix(train_data[, feature_cols])
  y_train <- train_data$class4
  
  x_test <- as.matrix(test_data[, feature_cols])
  y_test <- test_data$class4
  
  # Standardize the features
  scale_params <- preProcess(x_train, method = c("center", "scale"))
  x_train_scaled <- predict(scale_params, x_train)
  x_test_scaled <- predict(scale_params, x_test)
  
  # Convert target variable to numeric
  y_train_numeric <- as.integer(factor(y_train, levels = class_levels))
  y_test_numeric <- as.integer(factor(y_test, levels = class_levels))
  
  # Train the LiblineaR model
  try({
    cat("Training LiblineaR model for fold", current_fold, "\n")
    
    # Set type = 0 for multi-class classification using L2-regularized logistic regression (primal)
    liblinear_model <- LiblineaR(
      data = x_train_scaled,
      target = y_train_numeric,
      type = 0,
      cost = 1,
      bias = TRUE,
      verbose = FALSE
    )
    
    # Predict on test data
    liblinear_preds <- predict(liblinear_model, x_test_scaled)$predictions
    liblinear_preds <- factor(liblinear_preds, levels = 1:length(class_levels), labels = class_levels)
    
    # Calculate accuracy
    liblinear_accuracy <- mean(liblinear_preds == y_test)
    
    # Save results
    fold_results <- list(
      fold = current_fold,
      predictions = liblinear_preds,
      actuals = y_test,
      accuracy = liblinear_accuracy
    )
    saveRDS(fold_results, file = paste0("liblinear_results_fold_", current_fold, ".rds"))
    
    cat("LiblineaR model completed for fold", current_fold, "with accuracy:", liblinear_accuracy, "\n")
    
    # Append to results list
    results_list[[current_fold]] <- data.frame(fold = current_fold, accuracy = liblinear_accuracy)
    
  }, silent = FALSE)
  
  cat("Completed fold", current_fold, "\n\n")
}


### Get the final results ###################
# Loading Results for Each Model
load_model_results <- function(model_name) {
  result_files <- list.files("model_results", pattern = paste0(model_name, "_results_fold_\\d+\\.rds"), full.names = TRUE)
  if (length(result_files) == 0) {
    return(data.frame(fold = integer(0), accuracy = numeric(0)))
  }
  results_list <- lapply(result_files, readRDS)
  results_df <- do.call(rbind, lapply(results_list, function(res) data.frame(fold = res$fold, accuracy = res$accuracy)))
  return(results_df)
}

# Load results for each model
xgb_results_df <- load_model_results("xgb")
rf_results_df <- load_model_results("rf")
lgb_results_df <- load_model_results("lgb")
liblinear_results_df <- load_model_results("liblinear")

#liblinear_results_df <- results_df %>% mutate(model = "LiblineaR")
#glmnet_results_df <- load_model_results("glmnet")

# Combine all results into one data frame for plotting
all_results <- rbind(
  data.frame(fold = xgb_results_df$fold, accuracy = xgb_results_df$accuracy, model = "XGBoost"),
  data.frame(fold = rf_results_df$fold, accuracy = rf_results_df$accuracy, model = "Random Forest"),
  data.frame(fold = lgb_results_df$fold, accuracy = lgb_results_df$accuracy, model = "LightGBM"),
  data.frame(fold = liblinear_results_df$fold, accuracy = liblinear_results_df$accuracy, model = "LiblineaR")
  #data.frame(fold = glmnet_results_df$fold, accuracy = glmnet_results_df$accuracy, model = "GLMNet")
)

# Plot accuracy across folds for each model
ggplot(all_results, aes(x = as.factor(fold), y = accuracy, color = model, group = model)) +
  geom_line() +
  geom_point() +
  labs(title = "Model Accuracy Across Folds", x = "Fold", y = "Accuracy") +
  theme_minimal()

# Final Mean Accuracies
mean_accuracies <- all_results %>%
  group_by(model) %>%
  summarise(mean_accuracy = mean(accuracy, na.rm = TRUE))

print("Final Mean Accuracies for Each Model:")
print(mean_accuracies)





