#!/bin/bash

#SBATCH --job-name=T1R_MSA      # Job name
#SBATCH --output=/storage/home/sqd5856/Desktop/default/protein_design/LatentMD/boltz/PD_predictions_T1R23/boltz_%j.out
#SBATCH --error=/storage/home/sqd5856/Desktop/default/protein_design/LatentMD/boltz/PD_predictions_T1R23/boltz_error_%j.txt
#SBATCH --account=yjz5549_a_gpu       # Specify your account
#SBATCH --partition=sla-prio          # Use the appropriate partition
#SBATCH --nodes=1                     # Request 1 node
#SBATCH --ntasks=1                    # Request 1 task (1 CPU)
#SBATCH --cpus-per-task=20            # Use 20 CPU cores
#SBATCH --mem=128G                     # Request 128GB of memory
#SBATCH --gpus=1                       # Request 1 GPU
#SBATCH --time=48:01:00                # Set maximum runtime (48 hours)

echo "Job Start Time:" $(date)  # Log job start time

# Load required modules
module load anaconda
conda activate Bolts1
module load gcc/12.2.0

# Set GPU visibility
export CUDA_VISIBLE_DEVICES=0

# Move to the script directory (adjust if needed)
cd /storage/home/sqd5856/Desktop/default/protein_design/LatentMD/boltz/

# Ensure script has execute permission
chmod +x PD_design.sh

# Run the batch script
./PD_design.sh

echo "Job End Time:" $(date)  # Log job end time
