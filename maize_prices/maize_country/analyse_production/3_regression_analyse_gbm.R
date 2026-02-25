# c price forecast; country

# table of contents:
# 0. general settings
# 1. data import and preparation of summarising environment 
# 2. first run - ML models (GBM)
### distribution: type of loss function to optimize during boosting
  # 'gaussian' (prefered) for continuous y, squared error
  # 'bernoulli' (default) for logistic regression, 0-1 outcomes
### n.trees: Number of trees/gradient boosting iterations (default = 100). 
#++ more trees = stable results (lower error)
#-- more trees = slower process; risk of overfitting; no need in small datasets
### interaction.depth: *Maximum* nodes/splits per tree (default = 1), seq(1, 7, by = 2)
#++ higher depth = more complexity to the model and requires less trees.
#-- higher depth = slower model; risk of overfitting. No need in small datasets
### shrinkage: learning rate (default = 0.1). 
## shrinkage is used for reducing/shrinking, the impact of each tree. keep within range 0.001–0.3.
# In practice shrinkage cn be as small as possible and then select $T$ by cross-validation. 
#++ small shrinkage = robust model; easier to stop prior to overfitting
#-- small shrinkage = slower process; risk of overfitting
### bag.fraction: %/training set obs randomly selected to propose the next tree (default = 0.5)
# When close to 1, may cause overfitting. Better to optimize using gbm.perf(...,method="OOB")
# 5. Summarise & Save

#### load packages ####
library(caret)
library(gbm) 
library(tidyverse)

#c = 'soybean'

setwd(paste0('~/price_forecasting/',c,'_prices'))
source(paste0('master_cmaf_',c,'.R'))

# pmonth=2
# lags=3

setwd(wd_country_analyse_proc)
rm(list=setdiff(ls(), 
                c('wd_main', 'wd_country_analyse_proc',
                  'pmonth', 'lags','c')))

print(paste('month =', month.name[pmonth],'for', lags, 'months horizon',
            'started at', Sys.time()))
seed = 4
################################################################################
#--------------------- 1. preparation of input ~ output information 
################################################################################
error <- function(obs, pred){
  abs(obs - pred)
}

#define rmse function
rmse <- function(predicted,observed){ 
  sqrt(mean(
    (as.numeric(predicted) - as.numeric(observed)
    )^2))
}

#-------------------------------------------------------------------------------------------------
# 1.1. define main characters to loop on ###
#-------------------------------------------------------------------------------------------------  

### General parameters:
# analyseing model
mod_name = "gbm"

#geographic scale
geo = "country"

# model input (default)
def_input = "production"

# m
m = pmonth

#l
l = lags

# lags to be excluded from dataset
lag_exclude = 0:(l-1)
pattern_exclude = paste0('_',lag_exclude)

#--------------------------------------------------------------------------------
# 1.2. load data and adjust for forecast ###
#--------------------------------------------------------------------------------
load("analyse_data.RData")

# filter relative to month m
my_data <- my_data[[def_input]][my_data[[def_input]][,'month'] == pmonth,]

# filter to include only lags>=l
my_data <- my_data %>%
  dplyr::select(! ends_with(pattern_exclude)) %>%
  # remove if column includes only NA
  select_if(~!all(is.na(.)))

df = my_data %>%
  dplyr::select(-feature_head, -date) %>%
  as.matrix()


y = which(colnames(df) == 'obs.')
variables <- colnames(df[,-(1:y)])
variables

# learning pmonth
# start testing from year 2000
begin <- which(my_data$year==2000)
n_obs <- as.numeric(nrow(df))
### RF specific
# t
tree_seq <- seq(50, 500, by = 50) # specify numbers of trees

# d
max_depth = seq(1, 7, by = 2) # min. obs. in a node default = 1

# shrink (don't control)
learning_rate = c(0.05,0.1)

#-------------------------------------------------------------------------------------------------
# 1.3. create main environments to save results on ###
#-------------------------------------------------------------------------------------------------
rolls <- numeric(8*(n_obs))

# create a matrix to record all obs.~pred. prices
records <- matrix(data = rolls, ncol = 8)
colnames(records) = c("obs.", "pred.", "error", "year_test", 
                      'lags',
                      "tree_seq", "depth", "extra")

# importance ranking:
### MODEL #  var | inc_mse | f.model | tree_seq | month | LO_year | depth
rank_ <- matrix(NA, ncol = 12, nrow = 0) %>% as.data.frame() 
colnames(rank_) = c("var", "contribution", "analyse_model",  
                    "month", "year_test", 'lags', 
                    "geo_scale", "d_input", 
                    "tree_seq", "depth", "mtry", "extra")


################################################################################
#------------------------------ 2. run model -----------------------------------
################################################################################

# analyse prices (RCV) using different numbers of trees (tree_seq) 
#                                      and depth of model (max_depth)
### gbm: 
# the model ranks the variables by their importance, each time giving up one year
# the features with the highest overall impact will be included in the final regression formulas

# create a regression formula of Price ~ x's
reg_form <- formula( #P(t) = f{P(t-1), x(t)}
  paste(colnames(df)[y],paste(colnames(df)[-(1:y)],collapse='+'),
        sep='~'))

print(reg_form)

# Set a seed for reproducibility
set.seed(seed)
for (i in begin:n_obs){
  print(paste0('analyse for ', 
               df[i,'year'], '/', df[n_obs,'year']))
  
  training <- df[-i,-(1:(y-1))] # -c(date)
  testing <- matrix(df[i,-(1:(y-1))], nrow = 1)
  
  colnames(testing) = colnames(training)
  
  observed <- testing[1,1] # observed price in year i
  year_test <- df[i,'year'] # year to analyse
  
  # matrix to save parameters
  # run | depth | tree_seq | extra (shrinkage, cp, AIC)
  situation <- expand.grid(trees = tree_seq, depth = max_depth, shrinkage = learning_rate)
  situation <- cbind(run = 1:nrow(situation), situation)
  situation <- as.matrix(situation)
  
  #---------------- temperate results saved here -------------------------
  # create an empty records matrix containing Error column
  records_temp <- matrix(numeric(ncol(records) * nrow(situation)), 
                         nrow = nrow(situation), ncol = ncol(records))
  colnames(records_temp) <- colnames(records)
  
  MOD <- list() # save models here
  
  for (run in 1:nrow(situation)){
    t = situation[run,'trees']
    d = situation[run,'depth']
    shrink = situation[run,'shrinkage']
    
    temp_ <- gbm(formula = reg_form,
                 data = as.data.frame(training), 
                 n.trees = t,
                 distribution = "gaussian", 
                 interaction.depth = d,
                 shrinkage = shrink)
    
    MOD[[run]] <- temp_
    
    # predict price of year i
    pred_ <- predict.gbm(temp_, newdata = as.data.frame(testing),
                         n.trees = t, type = "response")
    
    records_temp[run,] <- c(observed, pred_,
                            error(observed, pred_),
                            year_test,
                            lags,
                            t, d, shrink)
  }
  
  # subset to min. error
  records_temp <- records_temp[records_temp[,'error'] == min(records_temp[,'error']),]
  records_temp <- matrix(records_temp, ncol = ncol(records))
  colnames(records_temp) = colnames(records)
  # leave the lightest model, in case of several rows with equal error
  records_temp <- records_temp[1,]
  
  situation <- situation %>%
    as.data.frame() %>%
    filter(depth == records_temp['depth'] & 
             trees == records_temp['tree_seq'] &
             shrinkage == records_temp['extra']) %>%
    # ensure only 1 row
    slice_head(n = 1)
  
  situation
  
  # Given the best set of hyperparameters, 
  # try to get any additional improvement in the out-of-sample error.
  set.seed(seed)
  t <- gbm.perf(
    object = MOD[[situation$run]], # best model
    method = "test", # compute an out-of-sample estimate
    plot.it = F
  )
  d = situation$depth
  shrink = situation$shrinkage
  
  if(length(t) > 0){
    print('success')
    temp_ <- gbm(formula = reg_form,
                 data = as.data.frame(training),
                 n.trees = t,
                 distribution = "gaussian",
                 interaction.depth = d,
                 shrinkage = shrink)
    
    MOD[[run+1]] <- temp_
    
    # predict price of year i
    pred_ <- predict.gbm(temp_, newdata = as.data.frame(testing),
                         n.trees = t, type = "response")
    
    if(error(observed, pred_) < records_temp['error'])
    records_temp <- c(observed, pred_,
                      error(observed, pred_),
                      year_test,
                      t, d, shrink)
  } else {NULL}
  
  records[i,] <- records_temp

  temp_ <- # model to keep
    MOD[[situation$run]]

  # relative importance in training years
  x = varImp(temp_, numTrees = situation$trees)
  #x = sort(x$Overall, decreasing = T) %>% as.data.frame()
  colnames(x) = "contribute"

  x = x %>% tibble::rownames_to_column() %>%
    dplyr::rename_all(.funs = ~ c('var','contribute')) %>%
    dplyr::mutate(forecast_model = mod_name,
                  month = m,
                  year_test = year_test,
                  lags = l,
                  g_scale = geo, d_input = def_input,
                  tree_seq = situation$trees, 
                  depth = situation$depth,
                  mtry = NA,
                  extra = situation$shrinkage)


  rank_ <- rbind(rank_, x)
}

records <- records %>%
  as.data.frame() %>%
  mutate(forecast_model = mod_name, month = m, .before = year_test) %>%
  mutate(lags = l,
         g_scale = geo, d_input = def_input, .after = year_test) %>%
  mutate(mtry = NA, .after = depth)

################################################################################
#-------------- 5. Summarise & Save
################################################################################

titles <- c(paste0('rank_',mod_name,'_',def_input,'_',m,'lag',l,'.RData'),
            paste0('records_',mod_name,'_',def_input,'_',m,'lag',l,'.RData'))

files <- list(rank_, records)

setwd(wd_country_analyse_proc)
save(rank_, file = titles[1])
save(records, file = titles[2])






