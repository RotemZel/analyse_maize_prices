# c price forecast; country, yield data
#browseURL("https://www.r-bloggers.com/2017/07/generalized-additive-models/")
#browseURL('https://www.mainard.co.uk/post/why-mgcv-is-awesome/#generalized-additive-models-gams')


# table of contents:
# 0. general settings
# 1. data import and preparation of summarising environment 
# 2. test distribution of p: normal/not normal
# 3. run gam() regression with multiple versions, save as seperated models:
### formula: 
##### a. linear regression (default) → method = 'glm.fit'
##### b. nonlinear (smooth) interactions between p~X → if norm.dist: method = [glm.fit', 'GCV.Cp']  
##### c. nonlinear (tensor) two-dimensional interactions between p~X. 
# 3. Summarise & Save

#### load packages ####
#library(broom) # Summarizes key information about models
require(mgcv) # load the GAMs library
library(caret)  # For relative importance
library(tidyverse)

#c = 'soybean'

setwd(paste0('~/price_forecasting/',c,'_prices'))
source(paste0('master_cmaf_',c,'.R'))

#pmonth=2
#lags=3

setwd(wd_country_analyse_proc)
rm(list=setdiff(ls(), 
                c('wd_main', 'wd_country_analyse_proc',
                  'pmonth', 'lags','c')))

print(paste('month =', month.name[pmonth],'for', lags, 'months horizon',
            'started at', Sys.time()))

print(paste('month =', month.name[pmonth]))

################################################################################
#--------------------- 1. preparation of input ~ output information 
################################################################################
aic_opt <- function(lm_mod){
  aic_mod <- broom::glance(lm_mod)$AIC
  return((aic_mod))
}

error <- function(obs, pred){
  abs(obs - pred)
}

#-------------------------------------------------------------------------------------------------
# 1.1. define main characters to loop on ###
#-------------------------------------------------------------------------------------------------  

### General parameters:
# analyseing model
mod_name = "gam"

#geographic scale
geo = "country"

# model input (default)
def_input = "yield"

# m
m = pmonth

#l
l = lags

# lags to be excluded from dataset
lag_exclude = 0:(l-1)
pattern_exclude = paste0('_',lag_exclude)

#-------------------------------------------------------------------------------------------------
# 1.2. define main characters to loop on ###
#-------------------------------------------------------------------------------------------------  
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

# GAM allows different model structures, including linear and nonlinear components. 
# Build formula with smooth terms s() for each predictor.
# k (basis dimension) is kept low because training sets are small (~20-50 obs).
# select=TRUE adds an extra shrinkage penalty that can zero out irrelevant smooths.

# --- Correlation filter (same as LM) ---
# Remove highly correlated predictors before fitting
df_numeric <- df %>%
  as.data.frame() %>%
  dplyr::select(-year, -month, -obs.)

cor_matrix <- cor(df_numeric, use = "pairwise.complete.obs")
high_corr_threshold <- 0.9
high_corr_pairs <- which(abs(cor_matrix) > high_corr_threshold, arr.ind = TRUE)
high_corr_pairs <- high_corr_pairs[high_corr_pairs[, 1] != high_corr_pairs[, 2], ]
vars_to_remove <- unique(rownames(high_corr_pairs))

if (length(vars_to_remove) > 0) {
  df <- df %>%
    as.data.frame() %>%
    dplyr::select(-all_of(vars_to_remove)) %>%
    as.matrix()
}

y = which(colnames(df) == 'obs.')
variables <- colnames(df[,-(1:y)])

# --- Build smooth formula ---
# Set max basis dimension relative to sample size: k must be < n_training
n_training <- n_obs - 1  # LOO removes 1 observation
max_k <- min(5, floor(n_training / 3))  # conservative: at most 5, or n/3
max_k <- max(max_k, 3)  # at least 3 (minimum for a smooth)

# Wrap each predictor in s() with restricted k
smooth_terms <- paste0("s(", variables, ", k=", max_k, ")")
reg_form_smooth <- as.formula(
  paste("obs.", paste(smooth_terms, collapse = " + "), sep = " ~ ")
)

# Fallback: linear formula in case GAM with smooths fails
reg_form_linear <- formula(paste(colnames(df)[y], 
                                 paste(variables, collapse = '+'), sep = '~'))

method_setter = 'REML'  # REML is preferred over GCV.Cp for smoothing parameter estimation

#-------------------------------------------------------------------------------------------------
# 1.3. create main environments to save results on ###
#-------------------------------------------------------------------------------------------------
rolls <- numeric(4*(n_obs))

# create a matrix to record all obs.~pred. prices
records <- matrix(data = rolls, ncol = 4)
colnames(records) = c("obs.", "pred.", "error", "year_test") # extra = glm.fit/GCV.Cp

# importance ranking:
### MODEL #  var | inc_mse | f.model | tree_seq | month | LO_year | depth
rank_ <- matrix(NA, ncol = 12, nrow = 0) %>% as.data.frame() 
colnames(rank_) = c("var", "contribution", "analyse_model",  
                    "month", "year_test", 'lags',
                    "geo_scale", "d_input", 
                    "tree_seq", "depth", "mtry", "extra")

################################################################################
#----------------------------- 2. Run the model --------------------------------
################################################################################

set.seed(4)
for (i in begin:n_obs) 
{
  print(paste0('analyse for ', 
               df[i,'year'], '/', df[n_obs,'year']))
  
  training <- df[-i,-(1:(y-1))] # -c(date)
  testing <- matrix(df[i,-(1:(y-1))], nrow = 1)
  
  colnames(testing) = colnames(training)
  
  observed <- testing[1,1] # observed price in year i
  year_test <- df[i,'year'] # year to analyse
  
  # gam: Generalized additive models ###
  #-------------------------------------------------------------------------------------------------
  # Try smooth formula first; fall back to linear if it fails
  # (can happen when a predictor has too few unique values for a smooth)
  temp_ <- tryCatch(
    gam(reg_form_smooth,
        method = method_setter,
        select = TRUE,  # allows smooths to be penalised to zero
        data = as.data.frame(training)),
    error = function(e) {
      message(paste("Smooth GAM failed for year", year_test, "- falling back to linear:", e$message))
      gam(reg_form_linear,
          method = method_setter,
          data = as.data.frame(training))
    }
  )
  
  # predict price of year i
  pred_ <- predict.gam(temp_,
                       newdata = as.data.frame(testing),
                       type = 'response')
  
  # relative importance in training years
  # calculate relative importance scaled to 100
  x = varImp(temp_)
  #x = sort(x$Overall, decreasing = T) %>% as.data.frame()
  colnames(x) = "contribute"
  
  x = x %>% tibble::rownames_to_column() %>%
    dplyr::rename(var = rowname) %>%
    dplyr::mutate(forecast_model = mod_name,
                  month = m,
                  year_test = year_test,
                  lags = l,
                  g_scale = geo, d_input = def_input,
                  tree_seq = NA, depth = NA, mtry = NA,
                  extra = method_setter)
  
  rank_ <- rbind(rank_, x)
  
  records[i,] <- c(observed, pred_,
                   error(observed, pred_),
                   year_test)
}

records <- records %>%
  as.data.frame() %>%
  mutate(extra = method_setter) %>%
  mutate(forecast_model = mod_name,
         month = m, .before = year_test) %>%
  mutate(lags = l,
         g_scale = geo, d_input = def_input, 
         tree_seq = NA, depth = NA, mtry = NA, .after = year_test)

################################################################################
#-------------- 4. Summarise & Save
################################################################################

titles <- c(paste0('rank_',mod_name,'_',def_input,'_',m,'lag',l,'.RData'),
            paste0('records_',mod_name,'_',def_input,'_',m,'lag',l,'.RData'))

files <- list(rank_, records)

setwd(wd_country_analyse_proc)
save(rank_, file = titles[1])
save(records, file = titles[2])