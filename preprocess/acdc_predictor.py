import numpy as np
import torch
import matplotlib.pyplot as plt
import os
import re
import SimpleITK as sitk
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

np.random.seed(3)

device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
print(f"using device: {device}")

sam2_checkpoint = "checkpoints/sam2.1_hiera_large.pt"
model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"
sam2_model = build_sam2(model_cfg, sam2_checkpoint, device=device)
predictor = SAM2ImagePredictor(sam2_model)

def load_slice(file_path, slice_index, as_rgb=False):
    """Load specific slice from 3D medical image file and optionally convert to RGB format."""
    try:
        image = sitk.ReadImage(file_path)
        image_array = sitk.GetArrayFromImage(image)
        
        # Check if slice_index is within bounds
        if slice_index >= image_array.shape[0]:
            print(f"Warning: slice_index {slice_index} is out of bounds for image with shape {image_array.shape}. Skipping slice.")
            return None
            
        selected_slice = image_array[slice_index]
        
        # Handle case where min equals max
        min_val = np.min(selected_slice)
        max_val = np.max(selected_slice)
        
        if min_val == max_val:
            normalized_image = np.zeros_like(selected_slice)
        else:
            normalized_image = (selected_slice - min_val) / (max_val - min_val) * 255
            
        grayscale_image = normalized_image.astype(np.uint8)
        
        if as_rgb:
            grayscale_image = np.stack([grayscale_image] * 3, axis=-1)
            
        return grayscale_image
    except Exception as e:
        print(f"Error loading slice from {file_path}: {e}")
        return None

def predict_and_save(predictor, image_slice, gt_image, save_path, input_point, input_label):
    """Predict mask and save the combined GT and predicted image with input points visualized."""
    try:
        predictor.set_image(image_slice)
        masks, scores, _ = predictor.predict(
            point_coords=input_point,
            point_labels=input_label,
            multimask_output=False,
        )

        mask = masks[0]

        fig, axes = plt.subplots(1, 2, figsize=(10, 5))

        # GT display
        axes[0].imshow(gt_image)
        axes[0].set_title("Ground Truth")
        axes[0].axis('off')

        # Prediction display
        axes[1].imshow(image_slice)
        axes[1].imshow(mask, alpha=0.5)
        axes[1].set_title("Predicted with Input Points")
        axes[1].axis('off')

        # Overlay input points with stars
        for (x, y), label in zip(input_point, input_label):
            color = 'green' if label == 1 else 'red'
            axes[1].plot(x, y, marker='*', color=color, markersize=15, markeredgecolor='white', markeredgewidth=1.5)

        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()

        print(f"Saved combined GT and predicted image to {save_path}")
        return True

    except Exception as e:
        print(f"Error predicting and saving {save_path}: {e}")
        return False

def get_patient_frames(data_dir, patient_id):
    """Get all frame files for a specific patient."""
    frames = []
    pattern = re.compile(f"{patient_id}_frame\\d+\\.nii\\.gz$")
    
    for file in os.listdir(data_dir):
        if pattern.match(file):
            # Extract frame name
            frame_match = re.search(r'(frame\d+)', file)
            if frame_match:
                frame_name = frame_match.group(1)
                if frame_name not in frames:
                    frames.append(frame_name)
    
    return frames

def process_medical_images(img_dir="med/acdc/img", gt_dir="med/acdc/gt", output_dir="output", slices=10):
    """Process all patients and their frames in the dataset."""
    # Get list of patient IDs from the files in img_dir
    patient_pattern = re.compile(r'(patient\d+)_frame\d+\.nii\.gz')
    patient_ids = set()
    
    for file in os.listdir(img_dir):
        match = patient_pattern.match(file)
        if match:
            patient_ids.add(match.group(1))
    
    patient_ids = sorted(list(patient_ids))
    print(f"Found {len(patient_ids)} patients")
    
    # Process each patient
    for patient_id in patient_ids:
        print(f"Processing {patient_id}")
        
        frames = get_patient_frames(img_dir, patient_id)
        print(f"  Found {len(frames)} frames for {patient_id}: {frames}")
        
        # Process each frame
        for frame in frames:
            frame_path = f"{img_dir}/{patient_id}_{frame}.nii.gz"
            gt_path = f"{gt_dir}/{patient_id}_{frame}_gt.nii.gz"
            
            if not os.path.exists(gt_path):
                print(f"  GT file for {patient_id}, {frame} not found, skipping.")
                continue
                
            print(f"  Processing {patient_id}, {frame}")
            
            # input points and labels
            input_point = np.array([[100, 130], [100, 160]])
            input_label = np.array([1, 0])
            
            # Create output directory for this patient/frame
            save_dir = f"{output_dir}/{patient_id}/{frame}"
            os.makedirs(save_dir, exist_ok=True)
            
            # Process each slice
            for slice_idx in range(slices):
                slice_img = load_slice(frame_path, slice_idx, as_rgb=True)
                gt_img = load_slice(gt_path, slice_idx, as_rgb=True)
                
                # If slice_img or gt_img is None, skip this slice
                if slice_img is None or gt_img is None:
                    continue
                    
                save_path = f"{save_dir}/slice_{slice_idx+1}.png"
                
                # Predict and save the result
                predict_and_save(predictor, slice_img, gt_img, save_path, input_point, input_label)

# Change directory
if __name__ == "__main__":
    process_medical_images(
        img_dir="med/acdc/img",
        gt_dir="med/acdc/gt",
        output_dir="output_acdc",
        slices=10 
    )