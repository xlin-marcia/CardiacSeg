## 1. Reduce Storage Occupancy (memory bank)

- Tasks:

    - Analyze object pointer tokens and IoU/occlusion scores
    - Retain frames with high semantic significance
    - Discard redundant frames while preserving segmentation accuracy

- Implementation:

    - Semantic Significance Analysis:
        - Extract IoU predictions, object pointer tokens, and occlusion scores during inference.
        - Set up a threshold or scoring mechanism to identify high-significance frames.

    - Frame Filtering:
        - Implement a temporal frame selection algorithm based on IoU, object visibility, or change in object features across frames.

    - Memory Optimization:
        - Retain only keyframe embeddings in the memory bank.
        - Store compact representations of less significant frames (lower resolution or reduced feature dimension)

- Need to Read:

    - Best Frame Selection in a Short Video (WACV 2020)
        - D-CNN based appoach to select frames with high semantic significance, which aligns with the goal of retaining key frames and discarding redundant ones
    - Fast Template Matching and Update for Video Object Tracking and Segmentation (CVPR 2020)
        - utilizes IoU-based matching to generate preliminary results for object tracking and segmentation. It discusses the processing of current frames by adopting IoU-based matching to generate temporary preliminary results, which can inform the analysis of object pointer tokens and IoU scores during inference
    - SAM2Long: Enhancing SAM 2 for Long Video Segmentation with a Training-Free Memory Tree
        - constrained tree memory structure that maintains multiple memory pathways over time, allowing various masks to be stored as memory at each time step. This approach addresses error accumulation in video segmentation and offers strategies for retaining frames with high semantic significance
    - Per-Clip Video Object Segmentation (CVPR 2022)
        - treating video object segmentation as clip-wise mask propagation. It introduces a per-clip inference scheme that provides accuracy gains by clip-level optimization and efficiency gains by parallel computation of multiple frames


## 2. Computational Efficiency

- Tasks:

    - Replace dense temporal embeddings with distilled temporal tokens
    - Use attention weights to aggregate features into compact temporal summary tokens
    - Incorporate temporal aggregation in the memory encoder

- Implementation:

    - Temporal Embedding Distillation:
        
        - Apply attention mechanisms to aggregate adjacent temporal embeddings into a smaller number of summary tokens.
        - Implement a distillation framework to learn representations that capture temporal dynamics effectively.

    - Temporal Aggregation in Memory Encoder:
        
        - Modify the memory encoder to include temporal summarization layers (e.g., self-attention or RNN-based layers).
        - Experiment with different levels of temporal granularity for aggregation (e.g., short-term vs. long-term dependencies).

    - Efficient Cross-Attention:
        
        - Optimize cross-attention mechanisms between object pointer tokens and memory embeddings by sparsifying attention weights or using low-dimensional embeddings.

- Need to read

    - Distilling Temporal Knowledge with Masked Feature Reconstruction for 3D Object Detection (arXiv 2024)

