#!/bin/bash

# Change path
SOURCE_DIR="/Users/marcia/Desktop/datasets/ACDC/training"
GT_DEST="/Users/marcia/acdc_data/gt"
IMG_DEST="/Users/marcia/acdc_data/img"

mkdir -p "$GT_DEST"
mkdir -p "$IMG_DEST"

all_patients=($(ls -d "$SOURCE_DIR"/patient*))

echo "Found ${#all_patients[@]} patients for processing."

for patient_dir in "${all_patients[@]}"; do
    patient_id=$(basename "$patient_dir")

    for file in "$patient_dir"/*; do
        base_filename=$(basename "$file")

        if [[ "$base_filename" == *"_4d.nii.gz" ]]; then
            echo "Skipping $file"
            continue
        fi

        if [[ "$base_filename" == *"_gt.nii.gz" ]]; then
            cp "$file" "$GT_DEST/$base_filename"
            echo "Copied $file to $GT_DEST"
        else
            cp "$file" "$IMG_DEST/$base_filename"
            echo "Copied $file to $IMG_DEST"
        fi
    done
done

echo "All patient folders processed completely."
