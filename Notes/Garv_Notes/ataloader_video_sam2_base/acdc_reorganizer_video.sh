#!/bin/bash

# Source directory for slices
SLICES_DIR="ACDC/ACDC_training_slices" 
# Destination directory
VIDEO_SLICES_DEST="ACDC/video_slices"
# Ground truth destination directory
GT_VIDEO_SLICES_DEST="ACDC/gt_video_slices"

# Create destination directories
mkdir -p "$VIDEO_SLICES_DEST"
mkdir -p "$GT_VIDEO_SLICES_DEST"

echo "Processing ACDC training slices for video segmentation..."

# Extract unique patient IDs from slice filenames
patient_pattern="patient[0-9]+"
patient_ids=$(ls "$SLICES_DIR" | grep -o "$patient_pattern" | sort -u)

for patient_id in $patient_ids; do
    echo "Processing $patient_id slices..."
    
    # Create patient directory in destination
    mkdir -p "$VIDEO_SLICES_DEST/$patient_id"
    mkdir -p "$GT_VIDEO_SLICES_DEST/$patient_id"
    
    # Get all slice files for this patient
    patient_files=$(ls "$SLICES_DIR" | grep "^$patient_id")
    
    # Copy files to destination, preserving frame information for temporal consistency
    for file in $patient_files; do
        cp "$SLICES_DIR/$file" "$VIDEO_SLICES_DEST/$patient_id/"
        echo "Copied $file to $VIDEO_SLICES_DEST/$patient_id/"
        
        # Extract frame and slice information for ground truth preparation
        if [[ $file =~ ${patient_id}_frame([0-9]+)_slice_([0-9]+)\.h5 ]]; then
            frame_num="${BASH_REMATCH[1]}"
            slice_num="${BASH_REMATCH[2]}"
            
            # Create a symbolic link to the original file as ground truth
            # In a real scenario, you would have actual ground truth files
            # Here we're just creating placeholders
            ln -sf "$VIDEO_SLICES_DEST/$patient_id/$file" "$GT_VIDEO_SLICES_DEST/$patient_id/${patient_id}_frame${frame_num}_slice_${slice_num}_gt.h5"
            echo "Created ground truth link for $file"
        fi
    done
done

echo "All patient data processed completely for video segmentation."
