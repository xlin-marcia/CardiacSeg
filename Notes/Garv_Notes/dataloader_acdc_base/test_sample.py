import os
import numpy as np
from sam2.acdc_video_loader import load_h5_sequence, convert_to_rgb, predict_and_save_video
from sam2.build_sam import build_sam2_video_predictor

# Select a patient and slice to test
patient_id = "patient001"
slice_idx = 5

# Load the model
sam2_checkpoint = "checkpoints/sam2.1_hiera_large.pt"
model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"
sam2_model = build_sam2_video_predictor(model_cfg, sam2_checkpoint)

# Load the data
data_dir = f"ACDC/video_slices/{patient_id}"
files = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith('.h5')]
slice_sequences = load_h5_sequence(files)

# Process the selected slice
if slice_idx in slice_sequences:
    sequence = slice_sequences[slice_idx]
    rgb_sequence = convert_to_rgb(sequence)
    
    # Define custom points for prompting (optional)
    # These points target the left ventricle
    input_point = np.array([[100, 130]])
    input_label = np.array([1])
    
    # Save the results
    save_path = f"output_acdc_video/{patient_id}_slice_{slice_idx}_test.png"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    predict_and_save_video(
        predictor=sam2_model,
        video_sequence=rgb_sequence,
        gt_sequence=rgb_sequence.copy(),  # Using input as GT for visualization
        save_path=save_path,
        input_point=input_point,
        input_label=input_label
    )
    
    print(f"Results saved to {save_path}")
else:
    print(f"Slice {slice_idx} not found for {patient_id}")