#!/bin/bash
 
#########################
## SLURM JOB COMMANDS ###
#########################
#SBATCH --partition=maxgpu            ## or allgpu / cms / cms-uhh / maxgpu
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --job-name StochMH           # give job unique name
#SBATCH --output ./run_files/training-%j.out      # terminal output
#SBATCH --error ./run_files/training-%j.err
#SBATCH --mail-type END
#SBATCH --mail-user sebastian.bieringer@desy.de
#SBATCH --constraint=P100
#SBATCH --exclude=max-cmsg007
 
##SBATCH --nodelist=max-cmsg[001-008]         # you can select specific nodes, if necessary

 
#####################
### BASH COMMANDS ###
#####################
 
## examples:
 
# source and load modules (GPU drivers, anaconda, .bashrc, etc)
source ~/.bashrc

# activate your conda environment the job should use
conda activate bayesconda
 
# go to your folder with your python scripts
cd /home/bierings/MCMC_by_backprob/

echo $(date +"%Y%m%d_%H%M%S") $SLURM_JOB_ID $SLURM_NODELIST $SLURM_JOB_GPUS  >> cuda_vis_dev.txt

# run
python3 StochasticMH_Adam_baseline.py --folder="$1" --n_points=$2 --rho=$3 --sigma_data=$4 --lambda_factor=$5