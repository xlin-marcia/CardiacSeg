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