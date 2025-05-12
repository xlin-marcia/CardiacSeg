import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
import os
import time
from fifo import FIFOMemoryBank
from lru import LRUMemoryBank

def load_image_slices(filepath):
    img = nib.load(filepath)
    data = img.get_fdata()
    slices = []
    if data.ndim == 4:
        for t in range(data.shape[-1]):
            for z in range(data.shape[2]):
                slices.append((f"t{t}_z{z}", data[:, :, z, t]))
    elif data.ndim == 3:
        for z in range(data.shape[2]):
            slices.append((f"z{z}", data[:, :, z]))
    return slices

def visualize_frame(frame, title="Frame"):
    if frame is None:
        return
    plt.imshow(frame, cmap='gray')
    plt.title(title)
    plt.axis('off')
    plt.show()

def benchmark_memory_bank(bank, slices):
    start = time.time()
    for key, slice_2d in slices[:50]:
        bank.add_frame(key, slice_2d)
    for _ in range(15):
        _ = bank.retrieve_frame("t0_z5")
    for _ in range(5):
        _ = bank.retrieve_frame("t0_z2")
    for _ in range(1):
        _ = bank.retrieve_frame("t0_z7")
    end = time.time()
    return end - start

if __name__ == "__main__":
    filepath = "/Users/balajis/Desktop/NTU/ACDC/ACDC_Dataset/testing/patient101/patient101_4d.nii.gz"  
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")

    slices = load_image_slices(filepath)

    # FIFO Test
    fifo_bank = FIFOMemoryBank(capacity=50)
    fifo_time = benchmark_memory_bank(fifo_bank, slices)

    # LRU Test
    lru_bank = LRUMemoryBank(capacity=50)
    lru_time = benchmark_memory_bank(lru_bank, slices)

    print(f"[RESULT] FIFO Time: {fifo_time:.4f}s")
    print(f"[RESULT] LRU Time:  {lru_time:.4f}s")

    # Visualization for Verification
    for bank, name in [(fifo_bank, "FIFO"), (lru_bank, "LRU")]:
        for key in ["t0_z5", "t0_z2", "t0_z7"]:
            frame = bank.retrieve_frame(key)
            if frame is not None:
                visualize_frame(frame, title=f"{name} - {key}")
