#!/bin/bash

#SBATCH --job-name=esmep1distill         # Give the job a name
#SBATCH --output=/storage/home/sqd5856/Desktop/default/protein_design/LatentMD/MD_Design/esmep1_distill_out/md_latent_%j.out
#SBATCH --error=/storage/home/sqd5856/Desktop/default/protein_design/LatentMD/MD_Design/esmep1_distill_out/md_latent_error_%j.txt
#SBATCH --account=yjz5549_a_gpu             # Specify the account (your paid account)
#SBATCH --partition=sla-prio           # Specify the partition
#SBATCH --nodes=1                      # Request 1 node
#SBATCH --ntasks=1                     # Request 1 task (1 CPU)
#SBATCH --cpus-per-task=36 
#SBATCH --mem=256G                       # Request 1GB of memory
#SBATCH --gpus=1                       # Request 1 GPU
#SBATCH --time=72:01:00                # Set the job's max runtime


echo "Job Start Time:" $(date)             # Records the job start time

# Set CUDA visible devices to use all 1 GPUs
module load anaconda
conda activate PD
export CUDA_VISIBLE_DEVICES=0,1

# Run the Python script
CUDA_VISIBLE_DEVICES=0,1 python esm_distill_train.py --config esm650_distill.yaml

echo "Job End Time:" $(date)               # Records the job end time

