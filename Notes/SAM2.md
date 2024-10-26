# Task

- Promptable Visual Segmentation (PVS) task

- Task starts by taking point, box, mask as input on any frame of the video to identify ROI

- Then 'masklet' (spatio-temporal mask) is predicted  

- After this, provide more prompt on other frame to refine this mask iteratively


# Model

- Handle both images and videos

- Main innovation:

    - A memory that stores information about the object and the previous interaction

    - Memory Attention module to attend to the previous memories of target object

    - When it is applied to image, the memory is empty, the model behave like SAM


# Dataset

- Generate training data by using the model in the loop and the annotators to interactively annotate challenging new data

# Experiment

## Metrics
