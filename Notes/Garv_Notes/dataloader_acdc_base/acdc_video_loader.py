import numpy as np
import torch
import matplotlib.pyplot as plt
import os
import re
import h5py
import SimpleITK as sitk
import cv2
from sam2.build_sam import build_sam2_video_predictor
from sam2.sam2_video_predictor import SAM2VideoPredictor

np.random.seed(3)

device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
print(f"using device: {device}")

sam2_checkpoint = "checkpoints/sam2.1_hiera_large.pt"
model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"
sam2_model = build_sam2_video_predictor(model_cfg, sam2_checkpoint, device=device)

def load_h5_sequence(file_paths):
    """
    Load a sequence of H5 files to create a temporal sequence.
    Each H5 file contains a slice at a specific time point.
    
    Args:
        file_paths: List of paths to H5 files
        
    Returns:
        Dictionary mapping slice indices to sequences of frames
    """
    # Group files by slice index
    slice_groups = {}
    pattern = re.compile(r'patient\d+_frame(\d+)_slice_(\d+)\.h5')
    
    for file_path in file_paths:
        base_name = os.path.basename(file_path)
        match = pattern.match(base_name)
        if match:
            frame_num = int(match.group(1))
            slice_num = int(match.group(2))
            
            if slice_num not in slice_groups:
                slice_groups[slice_num] = []
            
            slice_groups[slice_num].append((frame_num, file_path))
    
    # Sort each slice group by frame number
    for slice_num in slice_groups:
        slice_groups[slice_num].sort(key=lambda x: x[0])
    
    # Load sequences for each slice
    slice_sequences = {}
    for slice_num, frame_files in slice_groups.items():
        frames = []
        for _, file_path in frame_files:
            try:
                with h5py.File(file_path, 'r') as h5f:
                    # Assuming the image data is stored under a key like 'image'
                    if 'image' in h5f:
                        image_data = h5f['image'][()]
                    else:
                        # Try to get the first dataset in the file
                        for key in h5f.keys():
                            if isinstance(h5f[key], h5py.Dataset):
                                image_data = h5f[key][()]
                                break
                    
                    frames.append(image_data)
            except Exception as e:
                print(f"Error loading H5 file {file_path}: {e}")
                continue
        
        if frames:
            slice_sequences[slice_num] = np.array(frames)
    
    return slice_sequences

def load_volume_sequence(file_paths):
    """
    Load a sequence of 3D volumes to create a 4D temporal sequence.
    This is used for video segmentation where each volume represents a time point.
    
    Args:
        file_paths: List of paths to NIFTI files
        
    Returns:
        4D numpy array with dimensions (time, slice, height, width)
    """
    volumes = []
    
    for file_path in file_paths:
        try:
            image = sitk.ReadImage(file_path)
            image_array = sitk.GetArrayFromImage(image)
            volumes.append(image_array)
        except Exception as e:
            print(f"Error loading volume from {file_path}: {e}")
            return None
    
    # Stack volumes along a new temporal dimension
    sequence = np.stack(volumes, axis=0)
    return sequence

def convert_to_rgb(sequence):
    """
    Convert a grayscale sequence to RGB format.
    
    Args:
        sequence: Numpy array of shape (T, H, W) or (T, H, W, 1)
        
    Returns:
        Numpy array of shape (T, H, W, 3)
    """
    # Normalize and convert to RGB
    normalized_sequence = []
    for frame in sequence:
        min_val = np.min(frame)
        max_val = np.max(frame)
        
        if min_val == max_val:
            normalized_frame = np.zeros_like(frame)
        else:
            normalized_frame = (frame - min_val) / (max_val - min_val) * 255
        
        grayscale_frame = normalized_frame.astype(np.uint8)
        
        # If the frame is already 3-channel, use it directly
        if len(grayscale_frame.shape) == 3 and grayscale_frame.shape[2] == 3:
            rgb_frame = grayscale_frame
        else:
            # Otherwise, convert to 3-channel
            if len(grayscale_frame.shape) == 3:
                grayscale_frame = grayscale_frame.squeeze(-1)
            rgb_frame = np.stack([grayscale_frame] * 3, axis=-1)
        
        normalized_sequence.append(rgb_frame)
    
    return np.array(normalized_sequence)

def load_ground_truth_sequence(patient_id, slice_idx, data_dir="ACDC/gt_video_slices"):
    """
    Load ground truth masks for a specific patient and slice sequence.
    
    Args:
        patient_id: The patient ID
        slice_idx: The slice index
        data_dir: Directory containing ground truth data
        
    Returns:
        numpy array: Sequence of ground truth masks
    """
    gt_path = f"{data_dir}/{patient_id}"
    if not os.path.exists(gt_path):
        print(f"Ground truth directory for {patient_id} not found")
        return None
    
    # Look for ground truth files matching the pattern
    gt_files = []
    pattern = re.compile(f"{patient_id}_frame(\\d+)_slice_{slice_idx}_gt\\.h5$")
    
    for file in os.listdir(gt_path):
        match = pattern.match(file)
        if match:
            frame_num = int(match.group(1))
            gt_files.append((frame_num, os.path.join(gt_path, file)))
    
    # Sort by frame number
    gt_files.sort(key=lambda x: x[0])
    
    # Load ground truth masks
    gt_sequence = []
    for _, file_path in gt_files:
        try:
            with h5py.File(file_path, 'r') as h5f:
                if 'mask' in h5f:
                    mask_data = h5f['mask'][()]
                else:
                    # Try to get the first dataset
                    for key in h5f.keys():
                        if isinstance(h5f[key], h5py.Dataset):
                            mask_data = h5f[key][()]
                            break
                
                # Convert mask to RGB for visualization
                rgb_mask = np.zeros((mask_data.shape[0], mask_data.shape[1], 3), dtype=np.uint8)
                
                # Color code different structures
                # Class 1: LV (Red)
                rgb_mask[mask_data == 1] = [255, 0, 0]
                # Class 2: Myocardium (Green)
                rgb_mask[mask_data == 2] = [0, 255, 0]
                # Class 3: RV (Blue)
                rgb_mask[mask_data == 3] = [0, 0, 255]
                
                gt_sequence.append(rgb_mask)
        except Exception as e:
            print(f"Error loading ground truth file {file_path}: {e}")
    
    if not gt_sequence:
        return None
    
    return np.array(gt_sequence)

def get_patient_h5_files(data_dir, patient_id):
    """Get all H5 files for a specific patient."""
    files = []
    pattern = re.compile(f"{patient_id}_frame\\d+_slice_\\d+\\.h5$")
    
    for file in os.listdir(data_dir):
        if pattern.match(file):
            files.append(os.path.join(data_dir, file))
    
    return files

def get_patient_frame_sequences(data_dir, patient_id):
    """Get all frame files for a specific patient, organized by sequence."""
    frames = {}
    pattern = re.compile(f"{patient_id}_frame(\\d+)\\.nii\\.gz$")
    
    for file in os.listdir(data_dir):
        match = pattern.match(file)
        if match:
            frame_num = match.group(1)
            frames[frame_num] = os.path.join(data_dir, file)
    
    # Sort frames by frame number
    sorted_frames = [frames[k] for k in sorted(frames.keys())]
    return sorted_frames

def predict_and_save_video(predictor, video_sequence, gt_sequence, save_path, input_point=None, input_label=None):
    """
    Predict masks for a video sequence and save the results.
    Uses the SAM2 video predictor to leverage temporal consistency.
    
    Args:
        predictor: SAM2VideoPredictor instance
        video_sequence: Sequence of frames (T, H, W, C)
        gt_sequence: Sequence of ground truth masks (T, H, W, C)
        save_path: Path to save visualization
        input_point: Optional points for prompting
        input_label: Optional labels for prompting
    """
    try:
        # Initialize the video predictor state
        inference_state = predictor.init_state(video_sequence)
        
        # If input points are not provided, use default points
        if input_point is None or input_label is None:
            # Default points for cardiac structures
            input_point = np.array([[100, 130], [100, 160]])
            input_label = np.array([1, 0])
        
        # Add initial points for tracking
        obj_id = 1  # Assign a unique ID to the object we want to track
        frame_idx = 0  # Start with the first frame
        
        _, out_obj_ids, out_mask_logits = predictor.add_new_points_or_box(
            inference_state=inference_state,
            frame_idx=frame_idx,
            obj_id=obj_id,
            points=input_point,
            labels=input_label,
        )
        
        # Propagate through the video
        video_segments = {}
        for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state):
            video_segments[out_frame_idx] = {
                out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
                for i, out_obj_id in enumerate(out_obj_ids)
            }
        
        # Create visualization for each frame
        fig, axes = plt.subplots(len(video_segments), 2, figsize=(10, 5 * len(video_segments)))
        
        for i, (frame_idx, masks) in enumerate(video_segments.items()):
            if len(video_segments) == 1:
                ax_row = axes
            else:
                ax_row = axes[i]
                
            # GT display
            ax_row[0].imshow(gt_sequence[frame_idx])
            ax_row[0].set_title(f"Frame {frame_idx} - Ground Truth")
            ax_row[0].axis('off')
            
            # Prediction display
            ax_row[1].imshow(video_sequence[frame_idx])
            for obj_id, mask in masks.items():
                ax_row[1].imshow(mask, alpha=0.5)
            ax_row[1].set_title(f"Frame {frame_idx} - Predicted")
            ax_row[1].axis('off')
            
            # Overlay input points with stars on the first frame
            if frame_idx == 0:
                for (x, y), label in zip(input_point, input_label):
                    color = 'green' if label == 1 else 'red'
                    ax_row[1].plot(x, y, marker='*', color=color, markersize=15, markeredgecolor='white', markeredgewidth=1.5)
        
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()
        
        print(f"Saved video segmentation results to {save_path}")
        return True
    
    except Exception as e:
        print(f"Error predicting and saving video {save_path}: {e}")
        return False

def process_medical_videos(img_dir=None, gt_dir=None, output_dir="output_acdc_video", use_h5=True):
    """
    Process all patients and their frame sequences in the dataset for video segmentation.
    
    Parameters:
    - img_dir: Directory containing image files. If None, defaults based on use_h5 flag.
    - gt_dir: Directory containing ground truth files. If None, defaults based on use_h5 flag.
    - output_dir: Directory to save output visualizations.
    - use_h5: Whether to use H5 files (slices) or NIFTI files (volumes).
    """
    # Set default directories based on use_h5 flag if not provided
    if img_dir is None:
        img_dir = "ACDC/video_slices" if use_h5 else "ACDC/ACDC_training_volumes"
    
    if gt_dir is None:
        gt_dir = "ACDC/gt_video_slices" if use_h5 else "ACDC/gt_video_volumes"
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Get list of patient IDs from the files in img_dir
    patient_pattern = re.compile(r'(patient\d+)_frame\d+\.nii\.gz')
    h5_patient_pattern = re.compile(r'(patient\d+)_frame\d+_slice_\d+\.h5')
    patient_ids = set()
    
    if os.path.exists(img_dir):
        for file in os.listdir(img_dir):
            if use_h5:
                match = h5_patient_pattern.match(file)
            else:
                match = patient_pattern.match(file)
            
            if match:
                patient_ids.add(match.group(1))
    else:
        # If img_dir doesn't exist, check if it's a directory structure with patient subdirectories
        if os.path.exists(os.path.dirname(img_dir)):
            for dir_name in os.listdir(os.path.dirname(img_dir)):
                patient_dir = os.path.join(os.path.dirname(img_dir), dir_name)
                if os.path.isdir(patient_dir) and dir_name.startswith("patient"):
                    patient_ids.add(dir_name)
    
    patient_ids = sorted(list(patient_ids))
    print(f"Found {len(patient_ids)} patients")
    
    # Process each patient
    for patient_id in patient_ids:
        print(f"Processing {patient_id}")
        
        # Create output directory for this patient
        save_dir = f"{output_dir}/{patient_id}"
        os.makedirs(save_dir, exist_ok=True)
        
        if use_h5:
            # Process H5 files (slices with temporal information)
            patient_dir = os.path.join(img_dir, patient_id) if os.path.exists(os.path.join(img_dir, patient_id)) else img_dir
            h5_files = get_patient_h5_files(patient_dir, patient_id)
            if not h5_files:
                print(f"  No H5 files found for {patient_id}, skipping.")
                continue
            
            print(f"  Found {len(h5_files)} H5 files for {patient_id}")
            slice_sequences = load_h5_sequence(h5_files)
            
            # Process each slice sequence
            for slice_idx, sequence in slice_sequences.items():
                # Convert to RGB for visualization
                rgb_sequence = convert_to_rgb(sequence)
                
                # Try to load ground truth if available
                gt_sequence = load_ground_truth_sequence(patient_id, slice_idx, gt_dir)
                if gt_sequence is None or len(gt_sequence) != len(rgb_sequence):
                    print(f"  Using input sequence as ground truth for {patient_id}, slice {slice_idx}")
                    gt_sequence = rgb_sequence.copy()
                else:
                    print(f"  Loaded ground truth masks for {patient_id}, slice {slice_idx}")
                
                save_path = f"{save_dir}/slice_{slice_idx}_video.png"
                predict_and_save_video(
                    predictor=sam2_model,
                    video_sequence=rgb_sequence,
                    gt_sequence=gt_sequence,
                    save_path=save_path
                )
        
        else:
            # Process NIFTI files (volumes with temporal information)
            frame_files = get_patient_frame_sequences(img_dir, patient_id)
            if not frame_files:
                print(f"  No frame files found for {patient_id}, skipping.")
                continue
            
            print(f"  Found {len(frame_files)} frames for {patient_id}")
            
            # Load the sequence of volumes
            volume_sequence = load_volume_sequence(frame_files)
            if volume_sequence is None:
                continue
            
            # For each slice in the volumes, create a video sequence
            for slice_idx in range(volume_sequence.shape[1]):  # Loop through slices
                slice_sequence = volume_sequence[:, slice_idx]  # Get all frames for this slice
                rgb_sequence = convert_to_rgb(slice_sequence)
                
                # Try to load ground truth if available
                gt_sequence = None
                if os.path.exists(gt_dir):
                    gt_files = get_patient_frame_sequences(gt_dir, patient_id)
                    if gt_files:
                        gt_volume_sequence = load_volume_sequence(gt_files)
                        if gt_volume_sequence is not None and gt_volume_sequence.shape[1] > slice_idx:
                            gt_sequence = convert_to_rgb(gt_volume_sequence[:, slice_idx])
                
                if gt_sequence is None or len(gt_sequence) != len(rgb_sequence):
                    print(f"  Using input sequence as ground truth for {patient_id}, slice {slice_idx}")
                    gt_sequence = rgb_sequence.copy()
                else:
                    print(f"  Loaded ground truth masks for {patient_id}, slice {slice_idx}")
                
                save_path = f"{save_dir}/slice_{slice_idx}_video.png"
                predict_and_save_video(
                    predictor=sam2_model,
                    video_sequence=rgb_sequence,
                    gt_sequence=gt_sequence,
                    save_path=save_path
                )

# Main execution
if __name__ == "__main__":
    # First, make the reorganizer script executable
    import os
    os.system("chmod +x sam2/acdc_reorganizer_video.sh")
    
    # Run the reorganizer script to prepare data for video segmentation
    print("Running reorganizer script to prepare data...")
    os.system("./sam2/acdc_reorganizer_video.sh")
    print("Data preparation complete.")
    
    # Process using H5 files from organized video slices
    # This is the primary approach for SAM2 video segmentation with ACDC
    process_medical_videos(
        img_dir="ACDC/video_slices",
        output_dir="output_acdc_video",
        use_h5=True
    )
