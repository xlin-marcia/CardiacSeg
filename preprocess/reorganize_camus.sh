#!/bin/bash

# Change Path
SOURCE_DIR="/Users/marcia/Desktop/datasets/CAMUS_public/database_nifti"
GT_DEST="/Users/marcia/camus_data/gt"
IMG_DEST="/Users/marcia/camus_data/img"

mkdir -p "$GT_DEST"
mkdir -p "$IMG_DEST"

all_patients=($(ls -d "$SOURCE_DIR"/patient*))

echo "Found ${#all_patients[@]} patients for processing."

for patient_dir in "${all_patients[@]}"; do
    patient_id=$(basename "$patient_dir")

    for file in "$patient_dir"/*; do
        base_filename=$(basename "$file")

        if [[ "$base_filename" == *"4CH_ED_gt.nii.gz" ]] || [[ "$base_filename" == *"4CH_ES_gt.nii.gz" ]]; then
            cp "$file" "$GT_DEST/$base_filename"
            echo "Moved $file to $GT_DEST/$base_filename"

        elif [[ "$base_filename" == *"4CH_ED.nii.gz" ]] || [[ "$base_filename" == *"4CH_ES.nii.gz" ]]; then
            cp "$file" "$IMG_DEST/$base_filename"
            echo "Moved $file to $IMG_DEST/$base_filename"

        else
            echo "Skipping $file (not a target phase)"
        fi
    done
done

echo "All patients and matching files processed."
