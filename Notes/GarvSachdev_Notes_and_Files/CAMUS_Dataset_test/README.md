# Implementation of TransUNet Segmentation model on CAMUS Dataset

- to run directly on your system, please change the data path from "preprocessed_data" in test.py or train.py to the path to your Dataset.

- the data (after running preprocessing script.py) is stored in the following manner:
  ```markdown
  preprocessed_data
  |-> test
  |-> train
  |-> val
  ```
  change the directories in test.py, train.py accordingly to suit your own data path.
  
- this folder does NOT include the dataset due to size concerns.
  
- Download Google pre-trained ViT models
  * [Get models in this link](https://console.cloud.google.com/storage/vit_models/): R50-ViT-B_16, ViT-B_16, ViT-L_16...
  ```bash
  wget https://storage.googleapis.com/vit_models/imagenet21k/{MODEL_NAME}.npz &&
  mkdir ../model/vit_checkpoint/imagenet21k &&
  mv {MODEL_NAME}.npz ../model/vit_checkpoint/imagenet21k/{MODEL_NAME}.npz
  ```
  I used the R50-ViT-B_16 model.
