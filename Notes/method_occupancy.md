# Research on method for reducing occupancy

## Occupancy Rate

- Definition of Occupancy rate:

    proportion of the unique or distinct entries that are currently active or used in the memory bank

- How to measure:

    $\frac{Number\,of\, active\, memory\, entries}{Total \, memory \, capacity}$

- Be careful of what we are optimizing, it might be either storage or occupancy rate.-> follow definition


## Optical Flow Analysis

(Reduce Memory Usage)

**Mask Propagation** (NO)
 
1. Run the segmentation model once on the first frame

2. Use optical flow to shift the mask from that key frame onto the next frame (tracking how pixel move between frames, instead of running the model all over again for the next frame)

3. The shifted mask can be used directly, or as a starting guess for refining the segmentation


**Key Frame Scheduling** (NO)

1. Pick certain frames as "key frames"

2. Use SAM2 only on those key frames 

3. For all other frames, use optical flow to interpolate segmentation result

**Selective memory update with flow cues**

1. Use optical flow magnitude and patterns to see how much information a frame brings

2. If a threshold is met, then triggers the model to store the frame

**Related Paper**

[Moving Object Segmentation: All You Need Is SAM (and Flow)](https://arxiv.org/pdf/2404.12389)


**Problems**

- Occlusion and re-appearance 

    When the object is fiully occluded or leaves the frame, the flow cannot continue the track



## Edge Change Ratio (ECR)

- ECR is a metric from video analysis that quantifies how much the edge structure of an image changees from one frame to the next

- To compute ECR between frame n-1 and frame n, both frames are first converted to binary edge maps, then .....


## Structural Similarity Index (SSIM)

- A metric that compares images based on luminance, contrast, and structure. 

- Ranges from 0-1 (1-identical), setting a SSIM threshold allows a system to automatically filter out frames that are too similar to the last stored one.
