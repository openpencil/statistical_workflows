#----------------------------------------------------------------------------------
# Generates a mock dataset with a continuous response variable.
# Demonstrates the cross-validation procedure for a regression with xgboost models
#----------------------------------------------------------------------------------
# 
# 
#### Install packages if not installed already, and load libraries 

installpackages <- function(pkgs) {
  #' 
  #' @param pkgs array array of package names
  #' 
  #' @examples 
  #' pkgs <- c("data.table", "parallel", "xgboost", "plyr")
  #' 
  #' @output No explicit output. Installs packages and loads them in the environment
  #' 
  # Find which package are not installed
  pkgs_miss <- pkgs[which(!pkgs %in% installed.packages()[, 1])]
  # Install missing packages including their dependencies
  if (length(pkgs_miss) > 0) {
    install.packages(pkgs_miss, dependencies = TRUE)
    message(sprintf("The following packages were installed:\n %s", 
                    paste0(pkgs_miss, collapse = ",\n ")))
  }
  else if (length(pkgs_miss) == 0) {
    message("\n Listed packages are already installed!\n")
  }
  # Load packages not already loaded
  attached <- search()
  attached_pkgs <- attached[grepl("package", attached)]
  need_to_attach <- pkgs[which(!pkgs %in% gsub("package:", "", attached_pkgs))]
  if (length(need_to_attach) > 0) {
    for (i in 1:length(need_to_attach)) {
      library(need_to_attach[i], character.only = TRUE)
    }
  }
  if (length(need_to_attach) == 0) {
    message("Listed packages were already loaded!\n")
  }
}

#### Create test and training sets for k-fold cross-validation

create_test_train_sets <- function(data_table, folds, seed) {
  #' 
  #' @param data_table data.table a data.table with a response variable and covariates of interest
  #' @param folds integer number of train-test partitions in the cross-validation
  #' @param seed integer a seed to make the train-test partitions reproducible
  #' 
  #' @examples 
  #' data_table <- data.table(matrix(sample(x = 10:100, size = 500, replace = TRUE) 
  #'                      + rnorm(n = 500, mean = 0, sd = 1), nrow = 50, 
  #'                      dimnames = list(paste0("Sample", 1:50), 
  #'                                      c("response", paste0("Covar", 1:9)))))
  #' folds <- 10
  #' seed <- 1234
  #' 
  #' @output list A list of train-test dataset pairs in data.table format. The length 
  #' of the list is the number of folds specified. The training dataset in each 
  #' train-test pair contains nrow(data_table) - nrow(data_table)/folds rows and 
  #' ncol(data_table) columns. The test dataset in each train-test pair contains 
  #' nrow(data_table)/folds and ncol(data_table) columns. 
  #' 
  set.seed(seed)
  # Assign fold index to each row in x matrix
  foldindex <- sample(rep(x = (1:folds), times = (nrow(data_table)/folds)), replace = F)
  # Assign test/train partitions
  assigntesttrain <- function(fold){
    testset <- data_table[which(foldindex == fold), ]
    trainset <- data_table[which(foldindex != fold), ]
    out <- list(trainset, testset)
    names(out) <- c("train", "test")
    return(out)
    }
  # List of test/train data subsets. Number of test/train sets == number of folds
  testtrain <- lapply(1:folds, assigntesttrain)
  return(testtrain)
}

#### Run regression as a bootstrap with k-fold cross-validation to produce a CI 
#### around the cross-validated prediction (and error).

# Run one instance of xgboost with crossvalidation
run_xgboost <- function(datamatrix, response_varname, folds, seed){
  #' 
  #' @param datamatrix matrix a matrix with a response variable and covariates of interest
  #' @param response_varname string name of the response variable
  #' @param folds integer number of train-test partitions in the cross-validation
  #' @param seed integer a seed to make the train-test partitions reproducible
  #' 
  #' @examples 
  #' datamatrix <- matrix(sample(x = 10:100, size = 500, replace = TRUE) 
  #'                      + rnorm(n = 500, mean = 0, sd = 1), nrow = 50, 
  #'                      dimnames = list(paste0("Sample", 1:50), 
  #'                                      c("response", paste0("Covar", 1:9))))
  #' response_varname <- "response"
  #' folds <- 10
  #' seed <- 1234
  #' 
  #' @output list A list with as many elements as number of folds. Each element
  #' contains a xgboost model_object, predictions from the model, root-mean-squared
  #' error, mean absolute error and the test-train dataset pair used for model-building
  #' and for computing the errors.
  #'
  # Generate one resample of the input data
  data_table <- data.table(datamatrix)
  resample <- sample(x = nrow(data_table), size = nrow(data_table), replace = TRUE)
  # Generate a random seed
  randomseed <- round(runif(n = 1, min = 1, max = 100000), 0)
  test_train_sets <- create_test_train_sets(data_table = data_table[resample,],
                                            folds = 10, seed = seed)
  train_xgbmodel <- mclapply(test_train_sets, function(element) {
    # Debug
    # element <- test_train_sets[[1]]
    # 
    covariates_train <- data.matrix(element$train)
    covariates_train <- covariates_train[, -grep(response_varname, ignore.case = TRUE, colnames(element$train))]
    # Run the boosting algorithm for the given training set
    model_object <- xgboost(data = covariates_train,
                         label = element$train[, get(response_varname), with = TRUE], 
                         # Max depth of the tree. Small trees = less overfitting
                         max.depth = 4, 
                         # Eta: Learning rate. Higher the rate, the gradient 
                         # method moves towards the minima faster. Smaller rates
                         # are slower and require more computation but have higher
                         # potential of reaching the true optimum.
                         eta = 1,
                         nthread = 2, 
                         # nround: Number of trees in the model
                         nround = 10,
                         # The objective function 
                         objective = "reg:linear")
    covariates_test <- data.matrix(element$test)
    covariates_test <- covariates_test[, -grep(response_varname, ignore.case = TRUE, colnames(element$train))]
    predictions <- predict(model_object, covariates_test)
    # Root mean squared error
    rmse_predictions <- sqrt(mean((predictions - element$test[, get(response_varname), with = TRUE])^2))
    # Mean absolute error
    mae_predictions <- mean(abs(predictions - element$test[, get(response_varname), with = TRUE]))
    return(list(model_object = model_object,
                predictions = predictions,
                rmse = rmse_predictions,
                mae = mae_predictions,
                test_train_dataset = element))
    }, mc.cores = 12)
  return(train_xgbmodel)
}

# Run bootstrapped resampling
bootstrapped_xgboost_analysis <- function(number_bootstrap_resamples,
                                                    datamatrix,
                                                    response_varname,
                                                    folds,
                                                    seed,
                                                    output_directory,
                                                    filename){
  #'
  #' @param number_bootstrap_resamples integer number of bootstrap resamples to run. 
  #' @param datamatrix matrix a matrix with a response variable and covariates of interest
  #' @param response_varname string name of the response variable
  #' @param folds integer number of train-test partitions in the cross-validation
  #' @param seed integer a seed to make the train-test partitions reproducible
  #' @param output_directory string name of the directory to save output. 
  #' @param filename string name of the output file
  #' 
  #' @examples 
  #' number_bootstrap_resamples <- 10
  #' datamatrix <- matrix(sample(x = 10:100, size = 500, replace = TRUE) 
  #'                      + rnorm(n = 500, mean = 0, sd = 1), nrow = 50, 
  #'                      dimnames = list(paste0("Sample", 1:50), 
  #'                                      c("response", paste0("Covar", 1:9))))
  #' response_varname <- "response"
  #' folds <- 10
  #' seed <- 1234
  #' output_directory <- "."
  #' filename <- "demo_bootstrap_xgboost"
  #' 
  #' @output list A list with as many elements as the number of bootstrap 
  #' resamples. Each element is a list and the output of the bootstrap_xgboost
  #' function. The output of the bootstrap_xgboost function contains a xgboost 
  #' model_object, predictions from the model, root-mean-squared-error, 
  #' mean absolute error and the test-train dataset pair used for model-building
  #' and for computing the errors. The function also saves this list in the 
  #' specified output directory.
  #'
  #' 
  # Run xgboost model as many times as the number of bootstrap resamples
  model_output <- lapply(1:number_bootstrap_resamples, function(numboot){
    run_xgboost(datamatrix = datamatrix, 
                      response_varname = response_varname, 
                      folds = folds, 
                      seed = seed)
  })
  saveRDS(object = model_output,
          file = sprintf("%s/%s_%s.RDS", output_directory, filename, gsub("-", "", Sys.Date())))
  return(model_output)
}

####  Summarize the results of the bootstrapped xgboost model
summarize_model <- function(bootstrap_xgb, get_importance, response_varname){
  #' 
  #' @param bootstrap_xgb list list contain model objects from bootstrapped xgboost analysis
  #' @param get_importance boolean if TRUE, compute model importance
  #' @param response_varname string name of the response variable
  #' 
  #' @examples 
  #' bootstrap_xgb <- bootstrap_xgb
  #' get_importance <- TRUE
  #' response_varname <- "response"
  #' 
  #' @output list List containing either median and CI of variable importances 
  #' or the median and CI of errors (rmse/mae) across bootstrap resamples
  #' 
  bootci <- sapply(1:length(bootstrap_xgb), function(iter) {
       # debug options
       # split <- 1
       # iter <- 1
       importance_rmse_mae <- sapply(1:length(bootstrap_xgb[[1]]), function(split) {
         rmse <- bootstrap_xgb[[iter]][[split]]$rmse
         mae <- bootstrap_xgb[[iter]][[split]]$mae
         model <- xgb.dump(bootstrap_xgb[[iter]][[split]]$model_object)
         covariate_names <- setdiff(dimnames(data.matrix(datamatrix))[[2]], response_varname)
         # matrix with following metrics:
         # gain: contribution of each feature to the model
         # cover: how many observations are related to this feature
         # weight: relative number of times that a feature has been used for splitting within trees
         importance_matrix <- xgb.importance(covariate_names,
                                             model = bootstrap_xgb[[iter]][[split]]$model_object)
         out <- data.table(rmse = rmse, 
                           mae = mae,
                           importance = importance_matrix)
         return(out)
         }, simplify = F)
       errply <- ldply(importance_rmse_mae)
       err_summary <- errply[, c("rmse", "mae")]
       importance_summary <- data.table(errply[, setdiff(colnames(errply), c("rmse", "mae"))])
       # average importance across bootstrapped models
       importance_summary <- importance_summary[ ,sum(importance.Gain)/length(bootstrap_xgb), 
                                                 by = importance.Feature]
       errmean <- colMeans(err_summary)
       if (get_importance == TRUE) {
         out <- importance_summary
         } else {
           out <- errmean
           }
       return(out)
       }, simplify = F)
  bind_input <- ldply(bootci)
  if (get_importance == FALSE) {
    rmse_ci <- quantile(bind_input$rmse, c(0.025, 0.5, 0.975))
    mae_ci <- quantile(bind_input$mae, c(0.025, 0.5, 0.975))
    return(list(rmse_ci = rmse_ci,
                mae_ci = mae_ci))  
  } else {
    importances <- data.table(bind_input)
    importance_ci <- importances[, list(lower = quantile(V1, 0.025), 
                                        median = quantile(V1, 0.5), 
                                        upper = quantile(V1, 0.975)), 
                                 by = importance.Feature]
    setkey(x = importance_ci, median)
    return(importance_ci)
  }
}


#### Function calls

# 1) Install packages, Load libraries
installpackages(pkgs =  c("data.table", "parallel", "xgboost", "plyr"))

# 2) Generate test-train sets
test_train_sets <- create_test_train_sets(data_table = data.table(datamatrix), 
                                          folds = 10, 
                                          seed = 101)

# 3) Run bootstrap xgboost regression analysis
bootstrap_xgb <- bootstrapped_xgboost_analysis(number_bootstrap_resamples = 10, 
                                               folds = folds, 
                                               output_directory = ".", 
                                               filename = "xgb_bootstrap_demo", 
                                               datamatrix = datamatrix, 
                                               response_varname = response_varname,
                                               seed = seed)

# 4) Compute median and CI of variable importance
boot_importance <- summarize_model(bootstrap_xgb = bootstrap_xgb, 
                                   get_importance = TRUE, 
                                   response_varname = response_varname)  

# 5) Compute median and CI of rmse and mae
boot_errors <- summarize_model(bootstrap_xgb = bootstrap_xgb, 
                               get_importance = FALSE, 
                               response_varname = response_varname)  
