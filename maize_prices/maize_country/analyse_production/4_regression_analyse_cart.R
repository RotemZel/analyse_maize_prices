#c price forecast; countrys

# table of contents:
# 0. general settings
# 1. data import and preparation of summarising environment 
# 2. run cart regression with parameters to tune, through rpart.control():
# browseURL('https://uc-r.github.io/regression_trees#tune')
### complexity (cp): splits per tree; cp = nodes-1 (1 = no splits, default = 0.01). 
#++ high cp = small tree.
#-- low cp = risk of overfitting.
### maxcompete: numbers of vars to print, by impact per node (default = 4, visual only)
### minsplit (not so useful): minimum observations in a node for spliting (default = 20)
#++ small shrinkage (0) =
#-- small shrinkage (0) = slower process; risk of overfitting
### minbucket: The minimum number of observations in a terminal node (default = minsplit/3)
### maxdepth: maximum depth of any node of the final tree
#++ higher depth = more complexity to the model and requires less trees.
#-- higher depth = slower model; in 1 tree complexity is more important
# 3. Summarise & Save

library(caret)
library(rpart)
library(tidyverse)

#c = 'soybean'

setwd(paste0('~/price_forecasting/',c,'_prices'))
source(paste0('master_cmaf_',c,'.R'))

# pmonth=2
# lags=3

setwd(wd_country_analyse_proc)
rm(list=setdiff(ls(), 
                c('wd_main', 'wd_country_analyse_proc',
                  'pmonth', 'lags', 'c')))

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
mod_name = "cart"

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

#-------------------------------------------------------------------------------------------------
# 1.2. define main characters to loop on ###
#-------------------------------------------------------------------------------------------------  
load("analyse_data.RData")
colnames(my_data[[def_input]])

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

### CART specific
t = 1
# TBC: rpart.control()
# cp (0.01)
complexity = rev(seq(0.001, 0.01, length.out = 5))

# d (30; max. allowed = 30)
max_depth = 21:30 # min. obs. in a node

#-------------------------------------------------------------------------------------------------
# 1.3. create main environments to save results on ###
#-------------------------------------------------------------------------------------------------
rolls <- numeric(8*(n_obs))

# create a matrix to record all obs.~pred. prices
records <- matrix(data = rolls, ncol = 8)
colnames(records) = c("obs.", "pred.", "error", "year_test", 'lags', 
                      "tree_seq", "depth", "extra")

# importance ranking:
### MODEL #  var | inc_mse | f.model | tree_seq | month | LO_year | depth
rank_ <- matrix(NA, ncol = 12, nrow = 0) %>% as.data.frame() 
colnames(rank_) = c("var", "contribution", "analyse_model",  
                    "month", "year_test", 'lags', 
                    "geo_scale", "d_input", 
                    "tree_seq", "depth", "mtry", "extra")

################################################################################
#----------------------------- Run the model   ---------------------------------
################################################################################

# create a regression formula of Price ~ x's
reg_form <- formula( #P(t) = f{P(t-1), x(t)}
  paste(colnames(df)[3],paste(colnames(df)[-(1:3)],collapse='+'),
        sep='~'))

print(reg_form)

set.seed(seed)
for (i in begin:n_obs) 
{
  print(paste0('analyse for ', 
               df[i,'year'], '/', df[n_obs,'year']))
  
  training <- df[-i,-(1:(y-1))] # -c(date)
  testing <- matrix(df[i,-(1:(y-1))], nrow = 1)
  
  colnames(testing) = colnames(training)
  
  observed <- testing[1,1] # observed price in year i
  year_test <- df[i,'year'] # year to analyse
  
  # matrix to save parameters
  # run | depth | tree_seq | extra (shrinkage, cp, AIC)
  situation <- expand.grid(depth = max_depth, cp = complexity)
  situation <- cbind(run = 1:nrow(situation), situation)
  situation <- as.matrix(situation)
  
  #---------------- temperate results saved here -------------------------
  # create an empty records matrix containing Error column
  records_temp <- matrix(numeric(ncol(records)* nrow(situation)), 
                         nrow = nrow(situation), ncol = ncol(records))
  colnames(records_temp) <- colnames(records)
  
  MOD <- list() # save models here
  
  for (run in 1:nrow(situation)){
    #-------------------------------------------------------------------------------------------------  
    d = situation[run,'depth']
    cp = situation[run,'cp']
    
    #-------------------------------------------------------------------------------------------------
    # rpart: Recursive Partitioning and Regression Trees ###
    #-------------------------------------------------------------------------------------------------  
    tree_control = rpart.control(cp = cp, maxdepth = d)
    
    temp_ <- #if shrinkage is too large, run a bad model 
      rpart(reg_form, data = as.data.frame(training), method = "anova", 
            control = tree_control)
    
    MOD[[run]] <- temp_
    
    # predict price of year i
    pred_ <- predict(temp_, newdata = as.data.frame(testing),
                     type = "vector")
    
    records_temp[run,] <- c(observed, pred_,
                            error(observed, pred_),
                            year_test,
                            lags,
                            t, d, cp)
  }
  # subset to min. error
  records_temp <- records_temp[records_temp[,'error'] == min(records_temp[,'error']),]
  records_temp <- as.matrix(records_temp, ncol = ncol(records))
  # leave the lightest model, in case of several rows with equal error
  records_temp <- records_temp[1,]
  
  records[i,] <- records_temp
  
  situation <- situation %>%
    as.data.frame() %>%
    filter(depth == records_temp['depth'] & 
             cp == records_temp['extra']) %>%
    # ensure only 1 row
    slice_head(n = 1)
  
  situation
  
  temp_ <- # model to keep
    MOD[[situation$run]]
  
  # relative importance in training years
  x = varImp(temp_, surrogates = F, competes = T)
  #x = sort(x$Overall, decreasing = T) %>% as.data.frame()
  colnames(x) = "contribute"
  
  x = x %>% tibble::rownames_to_column() %>% 
    dplyr::rename(var = rowname) %>%
    dplyr::mutate(forecast_model = mod_name,  
                  month = m, 
                  year_test = year_test,
                  lags,
                  g_scale = geo, d_input = def_input,
                  tree_seq = t, depth = situation$depth,
                  mtry = NA,
                  extra = situation$cp)
  
  rank_ <- rbind(rank_, x)
}

records <- records %>%
  as.data.frame() %>%
  mutate(forecast_model = mod_name, month = m, .before = year_test) %>%
  mutate(lags,
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






