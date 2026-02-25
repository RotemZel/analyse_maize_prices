#31-04-2024 c price analyse; country

# table of contents:
# 0. general settings
# 1. data import and preparation of summarising environment 
# 2. run randomForest regression with parameters to tune:
### n.trees: number of independent trees/bootstrap sample (default = 500). 
#++ more trees = stable results (up to a certain point)
#-- more trees = slower process; risk of overfitting
### mtry: number of randomly selected predictors in each split (default = X/3)
#++ higher mtry = more likely to select important features
#-- higher mtry = less randomnnes, similar to CART. Sensitive to outliers. 
# 3. Summarise & Save

library(caret)
library(randomForest)
library(tidyverse)

#c = 'soybean'

setwd(paste0('~/price_forecasting/',c,'_prices'))
source(paste0('master_cmaf_',c,'.R'))

#pmonth=8
#lags=6

setwd(wd_country_analyse_proc)
rm(list=setdiff(ls(), 
                c('wd_main', 'wd_country_analyse_proc',
                  'pmonth','lags','c')))

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
mod_name = "rf"

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
tree_seq <- c(50,100,200) # specify numbers of trees

# mtry_
mtry_seq <- c(5:floor((length(variables)/2)))
#-------------------------------------------------------------------------------------------------
# 1.3. create main environments to save results on ###
#-------------------------------------------------------------------------------------------------
rolls <- numeric(7*(n_obs))

# create a matrix to record all obs.~pred. prices
records <- matrix(data = rolls, ncol = 7)
colnames(records) = c("obs.", "pred.", "error", "year_test", 'lags',
                      "tree_seq", "mtry")

# importance ranking:
### MODEL #  var | inc_mse | f.model | tree_seq | month | LO_year | depth
rank_ <- matrix(NA, ncol = 12, nrow = 0) %>% as.data.frame() 
colnames(rank_) = c("var", "contribution", "forecast_model",  
                    "month", "year_test", 'lags', 
                    "geo_scale", "d_input", 
                    "tree_seq", "depth", "mtry", "extra")

################################################################################
#----------------------------- Run the model   ---------------------------------
################################################################################

# create a regression formula of Price ~ x's
reg_form <- formula( #P(t) = f{P(t-1), x(t)}
  paste(colnames(df)[y],paste(colnames(df)[-(1:y)],collapse='+'),
        sep='~'))

print(reg_form)

set.seed(seed)
for (i in begin:n_obs) 
{
  print(paste0('analyse for ', 
               df[i,'year'], '/', df[n_obs,'year']))
  
  training <- df[-i,-(1:(y-1))] # -c(date)
  testing <- matrix(data = df[i,-(1:(y-1))], nrow = 1) # identify price in year i
  colnames(testing) = colnames(training)
  
  observed <- testing[1,1] # observed price in year i
  year_test <- df[i,'year'] # year to analyse
  
  # Tune Using Grid Search
  # run | mtry | tree_seq | extra (shrinkage, cp, AIC)
  situation <- expand.grid(trees = tree_seq, mtry = mtry_seq)
  situation <- cbind(run = 1:nrow(situation), situation)
  situation <- as.matrix(situation)
  
  #---------------- temperate results saved here -------------------------
  # create an empty records matrix containing Error column
  records_temp <- matrix(numeric(ncol(records) * nrow(situation)), 
                         nrow = nrow(situation), ncol = ncol(records))
  colnames(records_temp) <- colnames(records)
  
  MOD <- list() # save models here
  
  set.seed(seed)
  for (run in 1:nrow(situation))
    {
    
    t = situation[run,'trees']
    mtry_ = situation[run,'mtry']
    
    
    temp_ <- randomForest(reg_form, data = training,
                          type = "regression", importance = T,
                          ntree = t, mtry = mtry_)
    
    MOD[[run]] <- temp_
    
    # predict price of year i
    pred_ <- predict(temp_, newdata = testing,
                     type = "response")
    
    records_temp[run,] <- c(observed, pred_,
                            error(observed, pred_),
                            year_test,
                            lags,
                            t, mtry_)
  }
  
  # subset to min. error
  records_temp <- records_temp[order(records_temp[,'error'], decreasing = F),]
  # leave min error, in case of several rows with equal error
  records_temp <- records_temp[1,]
  
  records[i,] <- records_temp
  
  situation <- situation %>%
    as.data.frame() %>%
    rbind(c(run, t, mtry_)) %>%
    filter(mtry == records_temp['mtry'] & 
             trees == records_temp['tree_seq']) %>%
    # ensure only 1 row
    slice_head(n = 1)
    
  situation
    
  temp_ <- # model to keep
    MOD[[situation$run]]
  
  # relative importance in training years
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
                  tree_seq = situation$trees,
                  depth = NA, mtry = situation$mtry,
                  extra = NA
    )
  
  rank_ <- rbind(rank_, x)
}

records <- records %>%
  as.data.frame() %>%
  mutate(forecast_model = mod_name, month = m, .before = year_test) %>%
  mutate(lags = l, 
         g_scale = geo, d_input = def_input, .after = year_test) %>%
  mutate(depth = NA, .before = mtry) %>%
  mutate(extra = NA)

################################################################################
#-------------- 4. Summarise & Save
################################################################################

titles <- c(paste0('rank_',mod_name,'_',def_input,'_',m,'lag',l,'.RData'),
            paste0('records_',mod_name,'_',def_input,'_',m,'lag',l,'.RData'))

files <- list(rank_, records)

setwd(wd_country_analyse_proc)
save(rank_, file = titles[1])
save(records, file = titles[2])


