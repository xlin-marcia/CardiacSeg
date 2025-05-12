import numpy as np
from collections import OrderedDict

class LRUMemoryBank:
    def __init__(self, capacity=50):
        self.frames = OrderedDict()
        self.precision_map = {}
        self.capacity = capacity

    def add_frame(self, key, frame, bits=8):
        if key in self.frames:
            self.frames.move_to_end(key)
        self.frames[key] = frame.astype(np.float32)
        self.precision_map[key] = bits
        if len(self.frames) > self.capacity:
            self.frames.popitem(last=False)

    def retrieve_frame(self, key):
        if key in self.frames:
            self.frames.move_to_end(key)
            return self.frames[key]
        return None

    def quantize(self, frame, bits):
        if frame is None or bits >= 32:
            return frame.astype(np.float32) if frame is not None else None
        max_val, min_val = np.max(frame), np.min(frame)
        scale = (max_val - min_val) / (2**bits - 1) if max_val != min_val else 1
        quantized = np.round((frame - min_val) / scale) * scale + min_val
        return quantized.astype(np.float32)
