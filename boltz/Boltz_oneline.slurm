#!/bin/bash

#SBATCH --job-name=T1R_MSA      # Job name
#SBATCH --output=/storage/home/sqd5856/Desktop/default/protein_design/LatentMD/boltz/T1R2_AMAI_COMMERICAL_out/boltz_%j.out
#SBATCH --error=/storage/home/sqd5856/Desktop/default/protein_design/LatentMD/boltz/T1R2_AMAI_COMMERICAL_out/boltz_error_%j.txt
#SBATCH --account=yjz5549_cr_default       # Specify your account
#SBATCH --partition=standard          # Use the appropriate partition
#SBATCH --nodes=1                     # Request 1 node
#SBATCH --ntasks=1                    # Request 1 task (1 CPU)
#SBATCH --mem=128G                     # Request 128GB of memory
#SBATCH --gres=gpu:a100:1
#SBATCH --time=00:31:00                # Set maximum runtime (48 hours)

echo "Job Start Time:" $(date)  # Log job start time

# Load required modules
module load anaconda
conda activate Bolts1
module load gcc/12.2.0

# Set GPU visibility
export CUDA_VISIBLE_DEVICES=0

# Move to the script directory (adjust if needed)

boltz predict /storage/home/sqd5856/Desktop/default/protein_design/LatentMD/boltz/T1R2_AMAI_COMMERICAL/Q8TE_A-amai.fasta --out_dir /storage/home/sqd5856/Desktop/default/protein_design/LatentMD/boltz/T1R2_AMAI_COMMERICAL_out --cache /storage/home/sqd5856/Desktop/default/protein_design/LatentMD/boltz/.boltz --accelerator gpu --recycling_steps 10 --sampling_steps 200 --diffusion_samples 5 --step_scale 1.638 --output_format pdb --num_workers 20 --use_msa_server

echo "Job End Time:" $(date)  # Log job end time
