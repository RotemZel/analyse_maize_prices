#!/bin/bash

#This task script expects to be called by another script which sets up the params_maize_analyse_country_yield.txt file and passes the correct --array parameter value

#SBATCH --job-name=country_yield_analysis       # initial name; will be overridden per task
#SBATCH --output=logs/zlog-%x.%A_%a.out
#SBATCH --array=0-719
#SBATCH --mail-user=rotem.zelingher@wu.ac.at
#SBATCH --mail-type=END,FAIL
#SBATCH --partition=short        # by time: compute-test;small jobs;compute*
#SBATCH --mem=4GB
##SBATCH --cpus-per-task=2		# Allocate 1 core per task
##SBATCH --error=zlog-%x.%A_%a.err
##SBATCH --mem-per-cpu=32G

. /home/rzelingh/price_forecasting/init_modules_rlibs.sh
module purge
module load r/4.5.1-gcc-15.1.0-2gjbwfy

echo "I am job array task ID $SLURM_ARRAY_TASK_ID"

cd ~/price_forecasting

# Read the parameters for the current task
param_file="params_maize_analyse_country_yield.txt"
read pmonth lags script <<< $(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" $param_file)

# --- rename this array task in Slurm (per-task job name) ---
job_name="${c} analysis ${pmonth} at lag ${lags}"

# call model with this task's parameters
echo "pmonth=$pmonth" "lags=$lags" "source($script)"
Rscript -e "pmonth=$pmonth" -e "c='maize'" -e "lags=$lags" -e "source($script)"

