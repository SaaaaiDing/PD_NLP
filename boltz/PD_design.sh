#!/bin/bash

# Set paths
INPUT_DIR="/storage/home/sqd5856/Desktop/default/protein_design/LatentMD/boltz/T1R2_T1R3/"
OUTPUT_DIR="/storage/home/sqd5856/Desktop/default/protein_design/LatentMD/boltz/PD_predictions_T1R23/"
CACHE_DIR="/storage/home/sqd5856/Desktop/default/protein_design/LatentMD/boltz/.boltz"
CHECKPOINT="None"  

# Prediction parameters
ACCELERATOR="gpu"
RECYCLING_STEPS=5
DIFFUSION_SAMPLES=5
SAMPLING_STEPS=50
OUTPUT_FORMAT="pdb"
NUM_WORKERS=20
STEP_SCALE=1.638



# Ensure output directory exists
mkdir -p "$OUTPUT_DIR"

# Iterate through all receptor directories
for receptor_dir in "$INPUT_DIR"/*/; do
    receptor_name=$(basename "$receptor_dir")  # Extract receptor name
    receptor_output_dir="$OUTPUT_DIR/$receptor_name"  # Output directory for this receptor

    mkdir -p "$receptor_output_dir"  # Create output directory

    # Iterate through each binder FASTA file
    for fasta_file in "$receptor_dir"/*.fasta; do
        binder_name=$(basename "$fasta_file" .fasta)  # Extract binder name
        output_binder_dir="$receptor_output_dir/$binder_name"  # Output path for this binder

        mkdir -p "$output_binder_dir"  # Ensure output path exists

        # Run Boltz prediction
        echo "Predicting: $binder_name..."
        boltz predict "$fasta_file" \
            --out_dir "$output_binder_dir" \
            --cache "$CACHE_DIR" \
            --accelerator "$ACCELERATOR" \
            --recycling_steps "$RECYCLING_STEPS" \
            --sampling_steps "$SAMPLING_STEPS" \
            --diffusion_samples "$DIFFUSION_SAMPLES" \
            --step_scale "$STEP_SCALE" \
            --output_format "$OUTPUT_FORMAT" \
            --num_workers "$NUM_WORKERS" \
            --use_msa_server  
    done
done

echo "All Boltz predictions are complete!"
