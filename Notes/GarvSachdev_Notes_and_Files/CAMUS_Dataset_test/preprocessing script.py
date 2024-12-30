import os
import nibabel as nib
import numpy as np
from scipy.ndimage import zoom
from pathlib import Path

# Parameters
INPUT_DIR = "database_nifti"
SPLIT_DIR = "database_split"
OUTPUT_DIR = "preprocessed_data"
TARGET_SIZE = (224, 224)

def load_nii(file_path):
    """Load NIfTI file and return the image data."""
    return nib.load(file_path).get_fdata()

def resize_image(image, target_size):
    """Resize a 3D image to the target size."""
    factors = [t / s for t, s in zip(target_size, image.shape[:2])]
    resized_image = zoom(image, factors, order=1)
    return resized_image

def normalize_image(image):
    """Normalize image to [0, 1] range."""
    return (image - np.min(image)) / (np.max(image) - np.min(image))

def process_and_save(patient_id, split, output_dir):
    """Process images and save them in the specified output directory."""
    patient_dir = Path(INPUT_DIR) / patient_id
    output_split_dir = Path(output_dir) / split
    output_split_dir.mkdir(parents=True, exist_ok=True)
    
    for view in ["2CH", "4CH"]:
        for phase in ["ED", "ES"]:
            try:
                # Paths for images and ground truths
                image_path = patient_dir / f"{patient_id}_{view}_{phase}.nii.gz"
                gt_path = patient_dir / f"{patient_id}_{view}_{phase}_gt.nii.gz"

                if image_path.exists() and gt_path.exists():
                    image = load_nii(image_path)
                    gt = load_nii(gt_path)

                    image_resized = resize_image(image, TARGET_SIZE)
                    gt_resized = resize_image(gt, TARGET_SIZE)

                    image_normalized = normalize_image(image_resized)

                    # Save preprocessed files
                    np.save(output_split_dir / f"{patient_id}_{view}_{phase}.npy", image_normalized)
                    np.save(output_split_dir / f"{patient_id}_{view}_{phase}_gt.npy", gt_resized)
            except Exception as e:
                print(f"Error processing {patient_id} {view} {phase}: {e}")

def get_split_patients(split_file):
    """Read patient IDs from split file."""
    with open(split_file, "r") as file:
        return [line.strip() for line in file.readlines()]

def main():
    # Create output directory
    # Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

    # Load splits
    splits = {
        "train": get_split_patients(Path(SPLIT_DIR) / "subgroup_training.txt"),
        "val": get_split_patients(Path(SPLIT_DIR) / "subgroup_validation.txt"),
        "test": get_split_patients(Path(SPLIT_DIR) / "subgroup_testing.txt"),
    }

    # Process each split
    for split, patients in splits.items():
        print(f"Processing {split} split with {len(patients)} patients...")
        for patient_id in patients:
            process_and_save(patient_id, split, OUTPUT_DIR)

    print("Preprocessing complete!")

if __name__ == "__main__":
    main()
