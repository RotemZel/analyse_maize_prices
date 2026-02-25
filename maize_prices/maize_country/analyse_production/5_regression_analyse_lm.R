#02-09-2023 c price forecast; countrys
# url("http://www.sthda.com/english/articles/38-regression-model-validation/158-regression-model-accuracy-metrics-r-square-aic-bic-cp-and-more/")

# table of contents:
# 0. general settings
# 1. data import and preparation of summarising environment 
# 2. run linear regression with the Akaike’s Information Criteria:
### AIC search for appropriate subsets (removes non-contributing variables) 
### direction = control the search method (default = "backward")
### Forward stepwise: start with null, add best predictor one by one.
### Backward stepwise: start with full model, remove worst predictor one by one.
#-- lower AIC = better model
# ! stepwise AIC aims to min. AIC, not to min the forecasting errors.
### AIC is a version of AIC corrected for small sample sizes. 
# 3. Summarise & Save

#### load packages ####
library(broom) #summarizes key information about models
library(car)
library(caret) # for relative importance
library(tidyverse)

#c = 'soybean'

setwd(paste0('~/price_forecasting/',c,'_prices'))
source(paste0('master_cmaf_',c,'.R'))
# 
# pmonth=2
# lags=3

setwd(wd_country_analyse_proc)
rm(list=setdiff(ls(), 
                c('wd_main', 'wd_country_analyse_proc',
                  'pmonth', 'lags','c')))

print(paste('month =', month.name[pmonth],'for', lags, 'months horizon',
            'started at', Sys.time()))

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

#define rmse function
rmse <- function(predicted,observed){ 
  sqrt(mean(
    (as.numeric(predicted) - as.numeric(observed)
    )^2))
}

# 1.1. define main characters to loop on ### 

### General parameters:
# analyseing model
mod_name = "lm"

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

# to select a formula-based model by AIC.
# aic_d
direction = c("both", "backward", "forward")


########### 1.2. define main characters to loop on ###
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

################ 1.3. create main environments to save results on ###
rolls <- numeric(6*(n_obs))

# create a matrix to record all obs.~pred. prices
records <- matrix(data = rolls, ncol = 6)
colnames(records) = c("obs.", "pred.", "error", "year_test", 'lags', 
                      "extra")

# importance ranking:
### MODEL #  var | inc_mse | f.model | tree_seq | month | LO_year | depth
rank_ <- matrix(NA, ncol = 12, nrow = 0) %>% as.data.frame() 
colnames(rank_) = c("var", "contribution", "analyse_model",  
                    "month", "year_test", 'lags', 
                    "geo_scale", "d_input", 
                    "tree_seq", "depth", "mtry", "extra")


#------------- 2. Detect & Remove High Collinearity -------------------
#browseURL("https://www.statology.org/multicollinearity-regression/")

# Step 1: Remove non-numeric columns (if any) and the target variable (response variable) 'obs.'
df_numeric <- df %>%
  as.data.frame() %>%
  dplyr::select(-year, -month, -obs.)  # Exclude year, month, and response variable if necessary

# Step 2: Calculate the correlation matrix
cor_matrix <- cor(df_numeric, use = "pairwise.complete.obs")

# Step 3: Define a threshold for high correlation (e.g., 0.9)
high_corr_threshold <- 0.9

# Step 4: Find pairs of variables with high correlation
high_corr_pairs <- which(abs(cor_matrix) > high_corr_threshold, arr.ind = TRUE)

# Step 5: Exclude self-correlations (diagonal elements)
high_corr_pairs <- high_corr_pairs[high_corr_pairs[, 1] != high_corr_pairs[, 2], ]

# Step 6: Get the unique variables to remove (from one side of each correlated pair)
vars_to_remove <- unique(rownames(high_corr_pairs))

# Step 7: Remove these variables from the original dataframe `df`
df <- df %>%
  as.data.frame() %>%
  dplyr::select(-all_of(vars_to_remove)) %>%
  as.matrix()

# Step 8: Check the cleaned dataframe
# print(names(df))

y = which(colnames(df)=='obs.')

################################################################################
#----------------------------- 3. Run the model   ------------------------------
################################################################################

# create a regression formula of Price ~ x's
reg_form <- formula( #P(t) = f{P(t-1), x(t)}
  paste(colnames(df)[y],paste(colnames(df)[-(1:y)],collapse='+'),
        sep='~'))

print(reg_form)

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
  
  situation <- cbind(run = 1:length(direction), direction)
  situation <- as.matrix(situation)
  
  lm_temp_ <- lm(reg_form, 
                 data = as.data.frame(training))
  
  #---------------- temperate results saved here -------------------------
  
  MOD <- list() # save xgboost models here
  
  #-------------------------------------------------------------------------------------------------
  # traditional linear model, Akaike information criterion (AIC)###
  #-------------------------------------------------------------------------------------------------  
  for (run in 1:nrow(situation))
  {
    print(paste(year_test, 'direction =', run))
    aic_d = situation[run,'direction']
    #chooses only most important vars (AIC)
    temp_ <- step(lm_temp_, direction = aic_d,
                  trace = 0) #remove trace to get information
    
    MOD[[run]] <- temp_
  }
  
  situation <- cbind(run = 1:length(direction), 
                     aic = sapply(X = MOD, FUN = aic_opt))
  
  situation <- situation[which.min(situation[,'aic']),]
  
  temp_ <- MOD[[situation['run']]]
  
  # predict price of year i
  pred_ <- predict.lm(temp_, 
                      newdata = as.data.frame(testing),
                      type = "response")
  
  records[i,] <- c(observed, pred_,
                   error(observed, pred_),
                   year_test, 
                   lags,
                   situation['aic'])
  
  # variable importance in training years
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
                  tree_seq = NA, depth = NA,
                  mtry = NA,
                  extra = situation['aic'])
  
  rank_ <- rbind(rank_, x)
}

records <- records %>%
  as.data.frame() %>%
  mutate(forecast_model = mod_name, month = m, .before = year_test) %>%
  mutate(lags = l, 
         g_scale = geo, d_input = def_input, 
         tree_seq = NA, depth = NA,
         mtry = NA, .after = year_test)

################################################################################
#-------------- 4. Summarise & Save
################################################################################

titles <- c(paste0('rank_',mod_name,'_',def_input,'_',m,'lag',l,'.RData'),
            paste0('records_',mod_name,'_',def_input,'_',m,'lag',l,'.RData'))

files <- list(rank_, records)

setwd(wd_country_analyse_proc)
save(rank_, file = titles[1])
save(records, file = titles[2])






