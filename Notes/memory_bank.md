
## SAM2 Memory Bank 

### Memory Encoder:

1. A convolutional module downsample the output mask --> output
2. Unprocessed embedding from iamge encoder
3. Both 1 and 2 are fused together by light-weright convolutional layer, forms memory.

### Memory Bank:

- retains information:

    1. past predictions (for target objects, up to N frames)

    2. prompt (up to M frames)

- Both sets of memories are stored as **spatial feature map**.

- It stores a **list of object pointers as lightweight vectors**.

    - Based on the mask decoder outpout tokens to help model recognize object across frames.

- Memory attention will cross-attends both spatial feature maps and object pointer

- for recent frames, embed temporal position information --> track object motion


#### Base on the code:

**Object pointer**

- obj_ptr is one of the output (sam_outputs) from SAM decoder, along with masks and object confidence scores (inside function track_step)

- sam_output_tokens comes from self.sam_mask_decoder, which generates a sequence of tokens

- sam_output_token = sam_output_tokens[:, 0] extracts the first token, treating it as a high-level object representation

- The linear layer self.obj_ptr_proj projects the feature vector.

- Dimension: ```obj_ptr.shape == (B, C)```

    - B = Batch size (number of processed video frames)

    - C = Feature dimension (same as the SAM decoder output token size, set to hidden_dim)

- The obj_ptr is stored for every frame that undergoes segmentation tracking.

    From track_step, after being computed, obj_ptr is immediately stored:

        current_out["obj_ptr"] = obj_ptr

- How is obj_ptr used?

    - Memory Attention for Object Tracking

        - The model retrieves obj_ptr from past frames when tracking a new frame. (This happens inside _prepare_memory_conditioned_features)

        - These past object pointers are concatenated into memory attention

        - The transformer-based memory attention then cross-attends to obj_ptr to refine tracking

    - Handling Occlusions and Object Existence

        - If an object disappears or is occluded, obj_ptr helps retain its identity

        - The function _forward_sam_heads adjusts obj_ptr based on whether an object is detected

        - If soft_no_obj_ptr=True, the model softly mixes obj_ptr with a predefined "no-object" embedding
    
- Past values are stored in:
    ```
    to_cat_memory.append(obj_ptrs)
    to_cat_memory_pos_embed.append(obj_pos)
    ```
    - obj_ptrs holds past object identity embeddings
    
    - obj_pos stores temporal information about past frames

- new obj_ptr stored:

    ```current_out["obj_ptr"] = obj_ptr```

    - ```current_out["obj_ptr"]``` is overwritten each time ```track_step``` is called

- When tracking a new frame, past obj_ptr values are retrieved and used for memory attention, but they are not stored in current_out anymore.

- Inside _prepare_memory_conditioned_features:

```
if self.use_obj_ptrs_in_encoder:
    pos_and_ptrs = [
        (t_diff, out["obj_ptr"]) for t, out in selected_cond_outputs.items()]
```

- This extracts obj_ptr from past conditioning frames before processing the new frame.

- These obj_ptr values are stored in a separate memory structure (not current_out).


**Spatial feature map**

- How is it generated?
    - Spatial feature maps are generated from the image encoder (`image_encoder`), which extracts multi-scale visual features from input frames.

    - Inside `forward_image()`:
    ```python
    def forward_image(self, img_batch: torch.Tensor):
        backbone_out = self.image_encoder(img_batch)
    ```
    - If ```use_high_res_features_in_sam=True```, additional feature processing occurs for high-resolution segmentation.

- Where is it stored?
    - Spatial feature maps are stored in `backbone_out["backbone_fpn"]`:
    ```
    backbone_out = {
        "backbone_fpn": [feat_lvl_0, feat_lvl_1, feat_lvl_2],  # Multi-resolution features
        "vision_pos_enc": [pos_enc_lvl_0, pos_enc_lvl_1, pos_enc_lvl_2],  # Positional encodings
    }
    ```

- Dimension
    ```
    feature_map.shape == (B, C, H, W)
    ```
    - `B` = Batch size (number of frames)
    - `C` = Feature channels (depth of representation)
    - `H, W` = Height and width of the spatial feature map

    - **Example** (for a 512×512 input with `backbone_stride=16`):
    ```python
    feature_map.shape = (B, 256, 32, 32)
    ```

- How is it retrieved?
    - The function `_prepare_backbone_features()` extracts feature maps from `backbone_out`:
    ```
    def _prepare_backbone_features(self, backbone_out):
        feature_maps = backbone_out["backbone_fpn"][-self.num_feature_levels :]
        vision_pos_embeds = backbone_out["vision_pos_enc"][-self.num_feature_levels :]
    ```
    - This selects the appropriate number of feature levels based on `num_feature_levels`

- How is it updated?
    - The spatial feature map is updated per frame using memory attention

    - In `_prepare_memory_conditioned_features()`:
    ```
    pix_feat_with_mem = self.memory_attention(
        curr=current_vision_feats,
        curr_pos=current_vision_pos_embeds,
        memory=memory,
        memory_pos=memory_pos_embed,
        num_obj_ptr_tokens=num_obj_ptr_tokens,
    )
    ```
    - New feature maps (`current_vision_feats`) are retrieved
    - Memory attention integrates past feature maps (`memory`)
    - A new spatial feature map (`pix_feat_with_mem`) is generated, merging past and present frame knowledge.



### Methods to reduce occupancy

#### 1. Significance Analysis
- Goal: Identify which frames contain meaningful changes and should be stored.
- Approach:
    - Determine a measurement metric for change in object features across frames:
        - Frame Differencing: Compute the absolute difference between consecutive frames. Significant changes indicate motion or scene transitions. ​

        - Histogram Comparison: Analyze color histograms of successive frames; notable differences can signal scene changes. ​
        
        - Optical Flow Analysis: Measure pixel movement between frames to detect motion intensity and direction. ​

        - Edge Change Ratio (ECR): Assess variations in edge information to detect scene transitions. ​

        - Structural Similarity Index (SSIM): Evaluate perceptual differences between frames, focusing on luminance, contrast, and structure.
    
    - Considerations:

        - Threshold Setting: Define appropriate thresholds for each metric to distinguish significant changes from minor variations.​

        - Computational Load: Balance the accuracy of significance detection with processing time and resource consumption.


#### 2. Frame Filtering
- Goal: Reduce redundant storage by keeping only relevant frames.
- Approach:
    - Implement a temporal frame selection algorithm:
        - IoU-Based Selection: Only store frames where IoU between object masks changes significantly
        - Feature-Based Selection: Compare feature vector shifts (from Step 1) to filter out redundant frames
        - Genetic Algorithms: Utilize optimization strategies to select key frames based on predefined criteria. ​
        - Spatiotemporal Dynamics: Consider both spatial and temporal information to determine frame significance. ​
    - Retain only keyframes where:
        - The object undergoes significant deformation
        - The object's appearance changes drastically due to change in shape/size
        - The movement introduces major scene variations


#### 3. Memory Optimization
- Goal: Reduce memory footprint by optimizing how frames are stored.
- Approach:
    - Store only keyframe embeddings in the memory bank.
    - Use compressed representations for less significant frames:
        - Lower resolution feature maps (downsample to reduce memory size).
        - Reduced feature dimension (apply PCA or feature pruning).
    - Store a rolling buffer of the last `N` most significant frames instead of retaining all past frames.

#### 4. Need to Research
- Best metric to measure object feature change across frames:
    - What metric is most robust to shape and scale variations?
    - Does a combination (e.g., IoU + Optical Flow) provide better significance detection?

- What is an appropriate threshold to identify significant frames?
    - How much change should trigger frame storage?
    - Adaptive thresholding: Can a dynamic threshold (based on motion complexity) improve efficiency?

- Determine which method is the best for identifying significant frames:
    - Compare different significance metrics in real-world tracking scenarios.
    - Conduct experiments on various datasets to analyze accuracy and efficiency.
    - Evaluate trade-offs between computational cost and segmentation quality.


#### 5. Potential Implementation in SAM2
- Modify `_prepare_memory_conditioned_features()` to limit memory storage:
    ```python
    if self.use_obj_ptrs_in_encoder:
        # Only keep keyframes based on significance score
        pos_and_ptrs = [
            (t_diff, out["obj_ptr"]) for t, out in selected_cond_outputs.items()
            if significance_score(out["obj_ptr"]) > threshold
        ]
    ```
- Apply spatial compression before storing memory:
    ```python
    compressed_mem = downsample_features(maskmem_features)
    ```