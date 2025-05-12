import numpy as np
from collections import deque

class FIFOMemoryBank:
    def __init__(self, capacity=50):
        self.frames = {}
        self.queue = deque()
        self.capacity = capacity
        self.precision_map = {}

    def add_frame(self, key, frame, bits=8):
        if len(self.queue) >= self.capacity:
            oldest_key = self.queue.popleft()
            del self.frames[oldest_key]
            del self.precision_map[oldest_key]
        self.frames[key] = frame.astype(np.float32)
        self.queue.append(key)
        self.precision_map[key] = bits

    def retrieve_frame(self, key):
        return self.frames.get(key, None)

    def quantize(self, frame, bits):
        if frame is None or bits >= 32:
            return frame.astype(np.float32) if frame is not None else None
        max_val, min_val = np.max(frame), np.min(frame)
        scale = (max_val - min_val) / (2**bits - 1) if max_val != min_val else 1
        quantized = np.round((frame - min_val) / scale) * scale + min_val
        return quantized.astype(np.float32)
