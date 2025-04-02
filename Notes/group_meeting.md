## 17/Mar 

- Analyzed memory bank section of SAM2

- Short brainstorming on the method of reducing occupancy 

- Methods to be explored:

    - Counting the most frequently retrieved frame and then store less significant frames in FP16

    - Uniform maniford approximation and projection for dimension reduction

    - Cosine similarity

    - Frame Differencing: Compute the absolute difference between consecutive frames. Significant changes indicate motion or scene transitions. ​

    - Histogram Comparison: Analyze color histograms of successive frames; notable differences can signal scene changes. ​
    
    - Optical Flow Analysis: Measure pixel movement between frames to detect motion intensity and direction. ​

    - Edge Change Ratio (ECR): Assess variations in edge information to detect scene transitions. ​

    - Structural Similarity Index (SSIM): Evaluate perceptual differences between frames, focusing on luminance, contrast, and structure.

## 31/Mar

- Optimizing occupancy rate or reducing memory usage

- optical flow is most likely doing just what memory attention and memory bank is currently doing

- There exist problem with FIFO of memory bank, it may not be necessary to optimize, because it only stores 6 frames

- Could Up sample for echocardiogram

    - boost singnal to noise ratio
    
    - false contouring

    - convert to frequency domain through fft

- Down sample MRI

- skip connection could work for first and end frame


## Meeting with Huihui 2/Apr

- FIFO: How the memory contribute to the current task, may store efficiently only the useful frames (not storing every frame)

    - Which ones to select to make it out - make sure variety

    - compare the effect of segmentation

    - change the strategy of FIFO -> affect the accuracy of segmentation (ablation test)


- Loss function: wassertein 0.3, 0.2, 0.2, 0.3, use this for later training

- Use Lora fine tuning for encoder (ablation study on this)




