import numpy as np
import torch
import matplotlib.pyplot as plt
import os
import SimpleITK as sitk
import PIL.Image
from PIL.Image import Resampling
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

np.random.seed(3)

device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")

sam2_checkpoint = "checkpoints/sam2.1_hiera_large.pt"
model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"
sam2_model = build_sam2(model_cfg, sam2_checkpoint, device=device)
predictor = SAM2ImagePredictor(sam2_model)

def sitk_load(filepath: str):
    image = sitk.ReadImage(str(filepath))
    info = {
        "origin": image.GetOrigin(),
        "spacing": image.GetSpacing(),
        "direction": image.GetDirection()
    }
    im_array = np.squeeze(sitk.GetArrayFromImage(image))
    return im_array, info

def resize_image(image: np.ndarray, size: tuple, resample: Resampling = Resampling.NEAREST):
    return np.array(PIL.Image.fromarray(image).resize(size, resample=resample))

def resize_image_to_isotropic(image: np.ndarray, spacing: tuple, resample: Resampling = Resampling.NEAREST):
    scaling = np.array(spacing) / min(spacing)
    new_height, new_width = (np.array(image.shape) * scaling).round().astype(int)
    return resize_image(image, (new_width, new_height), resample), min(spacing)

def load_image(file_path, as_rgb=False, isotropic=False):
    try:
        im_array, info = sitk_load(file_path)

        if isotropic:
            im_array, _ = resize_image_to_isotropic(im_array, spacing=info["spacing"][:2])

        min_val, max_val = np.min(im_array), np.max(im_array)
        normalized = np.zeros_like(im_array) if min_val == max_val else \
            ((im_array - min_val) / (max_val - min_val) * 255).astype(np.uint8)

        return np.stack([normalized]*3, axis=-1) if as_rgb else normalized

    except Exception as e:
        print(f"Error loading image from {file_path}: {e}")
        return None

def predict_and_save(predictor, image, gt_image, save_path, input_point, input_label):
    try:
        predictor.set_image(image)
        masks, scores, _ = predictor.predict(
            point_coords=input_point,
            point_labels=input_label,
            multimask_output=False,
        )

        mask = masks[0]

        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        axes[0].imshow(gt_image)
        axes[0].set_title("Ground Truth")
        axes[0].axis('off')

        axes[1].imshow(image)
        axes[1].imshow(mask, alpha=0.5)
        axes[1].set_title("Predicted")
        axes[1].axis('off')

        for (x, y), label in zip(input_point, input_label):
            color = 'green' if label == 1 else 'red'
            axes[1].plot(x, y, '*', color=color, markersize=15, markeredgecolor='white')

        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()
        print(f"Saved: {save_path}")

    except Exception as e:
        print(f"Prediction failed for {save_path}: {e}")

def process_medical_images(img_dir, gt_dir, output_dir):
    input_point = np.array([[270, 220], [450, 250]])
    input_label = np.array([1, 0])

    os.makedirs(output_dir, exist_ok=True)
    img_files = [f for f in os.listdir(img_dir) if f.endswith(".nii.gz") and not f.endswith("_gt.nii.gz")]

    for img_file in img_files:
        patient_id = img_file.replace(".nii.gz", "")
        gt_file = f"{patient_id}_gt.nii.gz"

        img_path = os.path.join(img_dir, img_file)
        gt_path = os.path.join(gt_dir, gt_file)

        if not os.path.exists(gt_path):
            print(f"GT not found for {img_file}, skipping.")
            continue

        img = load_image(img_path, as_rgb=True, isotropic=True)
        gt_img = load_image(gt_path, as_rgb=True, isotropic=True)

        if img is None or gt_img is None:
            continue

        save_path = os.path.join(output_dir, f"{patient_id}.png")
        predict_and_save(predictor, img, gt_img, save_path, input_point, input_label)

# Change Directory
if __name__ == "__main__":
    process_medical_images(
        img_dir="med/camus/img",
        gt_dir="med/camus/gt",
        output_dir="output_camus"
    )