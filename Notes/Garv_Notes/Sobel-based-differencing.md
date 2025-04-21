# Selective Frame Feature Caching using Sobel-Based Novelty Detection
This module introduces a lightweight mechanism for selectively caching image features during video inference by detecting significant changes between frames. Instead of processing every frame independently, it only computes new features when the current frame differs substantially from previously seen ones.

## Key Features:
**Sobel-Based Novelty Detection**: Uses mean absolute difference between Sobel gradient magnitudes to measure visual change between frames.

**Selective Caching**: Only frames that are significantly different from cached memory frames are processed; otherwise, cached features are reused.

**Memory Management**: Maintains a fixed-size memory of the most recent N significant frames (default is 6), replacing the oldest entries when full.

**Efficient Reuse**: Reduces redundant computation by reusing features from the last stored frame when current changes are below a defined threshold.

## Why could help:
Could improve efficiency in video segmentation by avoiding unnecessary backbone computations on similar or static frames.

Offers a parameterized approach (tau_grad) to control sensitivity to visual changes.

Could useful in low-motion video scenarios such as medical image sequences, where frame-to-frame redundancy is common.

## Components:
**compute_grad_diff_metric**: Computes Sobel gradient-based difference between two frames. added in modeling/sam2_utils.py.

**is_significant**: Determines whether a frame is novel enough to warrant recomputation. added in modeling/sam2_utils.py.

**_get_image_feature (modified in sam2_video_predictor.py)**: Uses the above logic to either compute new features or retrieve from cache. changed the get image feature function in sam2_video_predictor.py.

## Code

1. in sam2_utils.py
   ```python
    from scipy import ndimage
    
    def compute_grad_diff_metric(img1: torch.Tensor, img2: torch.Tensor) -> float:
        """
        Compute mean absolute difference of Sobel gradient magnitudes between two frames.
        
        Args:
            img1, img2: torch.Tensor of shape (C,H,W) or (H,W), values in [0,255].
                        If 3‑channel, it averages to grayscale first.
        
        Returns:
            float: mean(|G2 - G1|) where Gx, Gy are Sobel gradients.
        """
        # move to CPU numpy
        arr1 = img1.detach().cpu().numpy()
        arr2 = img2.detach().cpu().numpy()
        
        # if RGB, convert to grayscale by averaging channels
        if arr1.ndim == 3:
            arr1 = arr1.mean(axis=0)
            arr2 = arr2.mean(axis=0)
        
        # Sobel gradients
        gx1 = ndimage.sobel(arr1, axis=1)  # horizontal
        gy1 = ndimage.sobel(arr1, axis=0)  # vertical
        gx2 = ndimage.sobel(arr2, axis=1)
        gy2 = ndimage.sobel(arr2, axis=0)
        
        # gradient magnitudes
        mag1 = np.hypot(gx1, gy1)
        mag2 = np.hypot(gx2, gy2)
        
        # mean absolute difference
        diff = np.abs(mag2 - mag1)
        return diff.mean()
    
    
    def is_significant(
        new_frame: torch.Tensor,
        memory_frames: list[torch.Tensor],
        tau_grad: float = 10.0
    ) -> bool:
        """
        Decide if `new_frame` is sufficiently different from *all* `memory_frames`
        based on Sobel‑gradient difference.
        
        Args:
            new_frame: torch.Tensor of shape (C,H,W) or (H,W), values [0,255].
            memory_frames: list of torch.Tensor, same shape as new_frame.
            tau_grad: threshold on mean gradient‑diff to consider “novel”.
        
        Returns:
            True if memory is empty OR for every old frame, 
            compute_grad_diff_metric(old, new_frame) >= tau_grad.
        """
        if not memory_frames:
            return True
        for old in memory_frames:
            if compute_grad_diff_metric(old, new_frame) < tau_grad:
                # too similar to this old frame
                return False
        return True

   ```

2. in sam2_video_predictor.py
   ```python
       @torch.inference_mode()
       def _get_image_feature(self, inference_state, frame_idx, batch_size):
           """
           Compute or retrieve selective cached image features using Sobel‑diff novelty.
           """
           from modeling.sam2_utils import is_significant
   
           # grab raw frame tensor
           raw = inference_state["images"][frame_idx].to(inference_state["device"]).float()
           # get or init lists
           feats_list = inference_state["cached_features"].setdefault(frame_idx, [])
           raws_list = inference_state["cached_raws"].setdefault(frame_idx, [])
   
           # decide if we need to compute new features
           if not feats_list or is_significant(new_frame=raw, memory_frames=raws_list, tau_grad=10.0):
               # compute fresh
               image = raw.unsqueeze(0)  # 1×C×H×W
               backbone_out = self.forward_image(image)
               # append selective
               feats_list.append((image, backbone_out))
               raws_list.append(raw)
               # enforce slot limit
               if len(feats_list) > self.max_mem_slots:
                   feats_list.pop(0)
                   raws_list.pop(0)
           else:
               # reuse last cached
               image, backbone_out = feats_list[-1]
   
           # now expand per-object
           expanded_image = image.expand(batch_size, -1, -1, -1)
           expanded_backbone_out = {
               "backbone_fpn": backbone_out["backbone_fpn"].copy(),
               "vision_pos_enc": backbone_out["vision_pos_enc"].copy(),
           }
           for i, feat in enumerate(expanded_backbone_out["backbone_fpn"]):
               expanded_backbone_out["backbone_fpn"][i] = feat.expand(
                   batch_size, -1, -1, -1
               )
           for i, pos in enumerate(expanded_backbone_out["vision_pos_enc"]):
               expanded_backbone_out["vision_pos_enc"][i] = pos.expand(
                   batch_size, -1, -1, -1
               )
   
           features = self._prepare_backbone_features(expanded_backbone_out)
           return (expanded_image,) + features
   ```
