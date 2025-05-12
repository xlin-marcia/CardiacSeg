import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
import os

class FrameMemoryBank:
    def __init__(self):
        self.frames = {}
        self.access_counts = {}
        self.precision_map = {}

    def add_frame(self, key, frame, bits=8):
        self.frames[key] = frame.astype(np.float32)  
        self.access_counts[key] = 0
        self.precision_map[key] = bits

    def retrieve_frame(self, key):
        if key in self.access_counts:
            self.access_counts[key] += 1
            return self.frames[key]
        else:
            print(f"[WARN] Frame {key} not found.")
            return None

    def update_precision(self, high_threshold=10, low_threshold=3):  
        # This updates the precision of frames based on access counts
        for key, count in self.access_counts.items():
            if count > high_threshold:
                self.precision_map[key] = 32
            elif count < low_threshold:
                self.precision_map[key] = 8
            else:
                self.precision_map[key] = 16

            original_frame = self.frames[key]
            quantized_frame = self.quantize(original_frame, self.precision_map[key])
            self.frames[key] = quantized_frame  # Replace in memory

    def quantize(self, frame, bits):
        if frame is None:
            return None
        if bits >= 32:
            return frame.astype(np.float32)
        max_val = np.max(frame)
        min_val = np.min(frame)
        scale = (max_val - min_val) / (2**bits - 1)
        if scale == 0:
            return frame
        quantized = np.round((frame - min_val) / scale) * scale + min_val
        return quantized.astype(np.float32)

    def get_quantized_frame(self, key):
        # Now the frame is already quantized and stored in memory
        return self.retrieve_frame(key)

def load_image_slices(filepath):
    print(f"[INFO] Loading: {filepath}")
    img = nib.load(filepath)
    data = img.get_fdata()

    slices = []

    if data.ndim == 4:
        # Shape: (H, W, D, T)
        for t in range(data.shape[-1]):
            for z in range(data.shape[2]):
                slices.append((f"t{t}_z{z}", data[:, :, z, t]))
    elif data.ndim == 3:
        # Shape: (H, W, D)
        for z in range(data.shape[2]):
            slices.append((f"z{z}", data[:, :, z]))
    else:
        raise ValueError(f"Unsupported data shape: {data.shape}")
    
    print(f"[INFO] Extracted {len(slices)} 2D slices.")
    return slices

def visualize_frame(frame, title="Frame"):
    if frame is None:
        print("[WARN] Empty frame passed to visualizer.")
        return
    if frame.ndim != 2:
        raise ValueError(f"Cannot visualize frame with shape {frame.shape}")
    
    plt.imshow(frame, cmap='gray')
    plt.title(title)
    plt.axis('off')
    plt.show()

# --- Main Execution ---

if __name__ == "__main__":
    filepath = "/path/to/nii.gz"  # Replace with your file

    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")

    slices = load_image_slices(filepath)

    memory_bank = FrameMemoryBank()

    # Stores only first 50 slices
    for key, slice_2d in slices[:50]:
        memory_bank.add_frame(key, slice_2d)

    print("[INFO] Available keys:", list(memory_bank.frames.keys())[:10])

    # Simulate access patterns
    for _ in range(15):
        _ = memory_bank.get_quantized_frame("t0_z5")
    for _ in range(5):
        _ = memory_bank.get_quantized_frame("t0_z2")
    for _ in range(1):
        _ = memory_bank.get_quantized_frame("t0_z7")

    # This will quantize and update stored frames
    memory_bank.update_precision()

    for key in ["t0_z5", "t0_z2", "t0_z7"]:
        if key in memory_bank.frames:
            quant_frame = memory_bank.get_quantized_frame(key)
            print(f"[INFO] {key} → {memory_bank.precision_map[key]} bits")
            visualize_frame(quant_frame, title=f"{key} ({memory_bank.precision_map[key]} bits)")
        else:
            print(f"[WARN] {key} not present in memory bank.")
