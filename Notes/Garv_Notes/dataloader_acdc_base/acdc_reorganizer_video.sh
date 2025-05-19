#!/bin/bash
SLICES_DIR="ACDC/ACDC_training_slices" 
VIDEO_SLICES_DEST="ACDC/video_slices"
GT_VIDEO_SLICES_DEST="ACDC/gt_video_slices"

mkdir -p "$VIDEO_SLICES_DEST"
mkdir -p "$GT_VIDEO_SLICES_DEST"

echo "Processing ACDC training slices for video segmentation..."

# extract unique patient IDs from slice filenames
patient_pattern="patient[0-9]+"
patient_ids=$(ls "$SLICES_DIR" | grep -o "$patient_pattern" | sort -u)

for patient_id in $patient_ids; do
    echo "Processing $patient_id slices..."
    
    # create patient directory in destination
    mkdir -p "$VIDEO_SLICES_DEST/$patient_id"
    mkdir -p "$GT_VIDEO_SLICES_DEST/$patient_id"
    
    # get all slice files for this patient
    patient_files=$(ls "$SLICES_DIR" | grep "^$patient_id")
    
    # copy files to destination, preserving frame information for temporal consistency
    for file in $patient_files; do
        cp "$SLICES_DIR/$file" "$VIDEO_SLICES_DEST/$patient_id/"
        echo "Copied $file to $VIDEO_SLICES_DEST/$patient_id/"
        
        # extract frame and slice information for ground truth preparation
        if [[ $file =~ ${patient_id}_frame([0-9]+)_slice_([0-9]+)\.h5 ]]; then
            frame_num="${BASH_REMATCH[1]}"
            slice_num="${BASH_REMATCH[2]}"
            
            # create a symbolic link to the original file as ground truth
            # need to have actual ground truth files
            # here we're just creating placeholders (just for testing)
            ln -sf "$VIDEO_SLICES_DEST/$patient_id/$file" "$GT_VIDEO_SLICES_DEST/$patient_id/${patient_id}_frame${frame_num}_slice_${slice_num}_gt.h5"
            echo "Created ground truth link for $file"
        fi
    done
done

echo "All patient data processed completely for video segmentation."
