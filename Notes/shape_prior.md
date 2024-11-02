## [AnatSwin](https://www.sciencedirect.com/science/article/pii/S0925231224001504)

(Neurocomputing 2024)

![anatswin_arch](../asset/anatswin_arch.png)

- **Encoder**

    - Takes template label image and pseudo label image as input

    - temlate label image is GT, pswudo label image is from registration model

    - The transformer is chosen for this architecture because of its strength in capturing long-range correlations, which are essential for representing both tissue morphology and spatial relationships between tissues. 
    
    - The encoder first divides each input image into patches, embedding them into vectors of length 128. Then, multi-scale features are extracted using a modified Swin Transformer (Swin-B version) with three layers, generating hierarchical features for both template and pseudo-label images.

    - The features extracted from the two inputs undergo a comprehensive interaction through the FI block to generate the interacted features. These interacted features are then propagated back to their respective branches for further processing. 

    - FIE(⋅) denotes the feature interaction operation through the proposed FI block in the encoder.

    - Fused by weighted addition operation.


- **Decoder**

    -  Take the hierarchical features extracted from the encoder and guide the network to apply the correct anatomical constraints from the template to the pseudo-label.

    - Feature Fusion: The decoder fuses features from both template and pseudo-label branches using an FI (Feature Interaction) block, creating combined features.

    - FID(⋅) denotes the feature fusion through the proposed FI block in the decoder.

    - Concatenation: The fused features are then merged (concatenated) to form a deep, shared feature.

    - This deep-level feature is combined with shallow-level features through up-sampling and convolution. This process gradually combines information from different layers to create a more detailed feature map at each level.

    - F Conv(⋅) comprises of an up-sampling operation, and two convolutional operations with 3 × 3 kernel size, each followed by a batch normalization layer and a rectified linear unit (ReLU) activation function.

    - βi1 and βi2 are two learnable parameters that adaptively weight the contributions of features during training.

![fi_block](../asset/fi_block.png)

- F1 Block:

    - The FI block is also designed to facilitate information interaction between features at the same hierarchy and learn the correlation between them, thereby improving the ability to capture anatomical structures.

    

## [Learning with Explicit Shape Priors for Medical  Image Segmentation](https://arxiv.org/pdf/2303.17967)

(TMI 2024)





## [Anatomy-Guided Pathology Segmentation](https://arxiv.org/pdf/2407.05844)

(MICCAI 2024)



## [Cardiac Segmentation With Strong  Anatomical Guarantees](https://arxiv.org/pdf/2006.08825)

(TMI 2020)



## [Accurate Airway Tree Segmentation in CT Scans](https://arxiv.org/pdf/2306.09116)

(TMI 2023)



