#!/bin/bash

cd ~/price_forecasting
# Loop over commodities months; lags
for c in maize; do

# Generate the parameter file
param_file="params_maize_analyse_region_production.txt"
: > $param_file
for i in {1..12}; do
  for l in {1..12}; do
    echo "$i $l '~/price_forecasting/maize_prices/maize_region/analyse_production/2_regression_analyse_rf.R'" >> $param_file
    echo "$i $l '~/price_forecasting/maize_prices/maize_region/analyse_production/3_regression_analyse_gbm.R'"  >> $param_file
    echo "$i $l '~/price_forecasting/maize_prices/maize_region/analyse_production/4_regression_analyse_cart.R'" >> $param_file
    echo "$i $l '~/price_forecasting/maize_prices/maize_region/analyse_production/5_regression_analyse_lm.R'" >> $param_file
    echo "$i $l '~/price_forecasting/maize_prices/maize_region/analyse_production/6_regression_analyse_gam.R'" >> $param_file
    done
done
done

# Get the number of lines in the parameter file
num_lines=$(wc -l < $param_file)

echo "There are ${num_lines} parameter sets.  use --array=0-$(($num_lines -1))%100"

# Submit the SLURM job with the correct array size
## num_lines=15 ## TEST
sbatch --array=0-$(($num_lines - 1))%100 ./task_analyse_maize_region_production.slurm
