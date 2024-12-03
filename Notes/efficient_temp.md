## Reduce Storage Occupancy (memory bank)

- Significance Analysis:
    - Determine a measurement metric for change in object feature across frames
    (Euclidean distance in feature space; Structural Similarity Index; optical flow...)
    - Set up a threshold or scoring mechanism to identify high-significance frames.

- Frame Filtering:
    - Implement a temporal frame selection algorithm based on IoU, object visibility, or change in object features across frames (metric from above step)

- Memory Optimization:
    - Retain only keyframe embeddings in the memory bank.
    - Store compact representations of less significant frames (lower resolution or reduced feature dimension)

- Need to research:

    - best metric to measure change in object feature across frames
    - what is an appropriate threshold to identify significant frames?

- Need to Read:

    - Best Frame Selection in a Short Video (WACV 2020)
        - D-CNN based appoach to select frames with high semantic significance, which aligns with the goal of retaining key frames and discarding redundant ones
    - Fast Template Matching and Update for Video Object Tracking and Segmentation (CVPR 2020)
        - utilizes IoU-based matching to generate preliminary results for object tracking and segmentation. It discusses the processing of current frames by adopting IoU-based matching to generate temporary preliminary results, which can inform the analysis of object pointer tokens and IoU scores during inference
    - SAM2Long: Enhancing SAM 2 for Long Video Segmentation with a Training-Free Memory Tree
        - constrained tree memory structure that maintains multiple memory pathways over time, allowing various masks to be stored as memory at each time step. This approach addresses error accumulation in video segmentation and offers strategies for retaining frames with high semantic significance
    - Per-Clip Video Object Segmentation (CVPR 2022)
        - treating video object segmentation as clip-wise mask propagation. It introduces a per-clip inference scheme that provides accuracy gains by clip-level optimization and efficiency gains by parallel computation of multiple frames



**similar methods might be applied to the memory block for inter-slice?**