#!/bin/bash

#SBATCH --job-name=md_latent          # Give the job a name
#SBATCH --error=md_latent_error_%j.txt   # Error file (%j will be replaced with job ID)
#SBATCH --account=yjz5549_a_gpu             # Specify the account (your paid account)
#SBATCH --partition=sla-prio           # Specify the partition
#SBATCH --nodes=1                      # Request 1 node
#SBATCH --ntasks=1                     # Request 1 task (1 CPU)
#SBATCH --cpus-per-task=20 
#SBATCH --mem=128G                       # Request 1GB of memory
#SBATCH --gpus=1                       # Request 1 GPU
#SBATCH --time=120:01:00                # Set the job's max runtime


echo "Job Start Time:" $(date)             # Records the job start time

# Set CUDA visible devices to use all 1 GPUs
export CUDA_VISIBLE_DEVICES=0

# Run the Python script
CUDA_VISIBLE_DEVICES=0 python main1115.py --config main_config.yaml

echo "Job End Time:" $(date)               # Records the job end time

