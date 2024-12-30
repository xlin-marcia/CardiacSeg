# Implementation of TransUNet Segmentation model on ACDC Dataset

- to run directly on your system, please change the data path from "./data/ACDC" in test.py or train.py to the path to your Dataset.
  
- this folder does NOT include the dataset due to size reasons.
  
- Download Google pre-trained ViT models
* [Get models in this link](https://console.cloud.google.com/storage/vit_models/): R50-ViT-B_16, ViT-B_16, ViT-L_16...
```bash
wget https://storage.googleapis.com/vit_models/imagenet21k/{MODEL_NAME}.npz &&
mkdir ../model/vit_checkpoint/imagenet21k &&
mv {MODEL_NAME}.npz ../model/vit_checkpoint/imagenet21k/{MODEL_NAME}.npz
```
I used the R50_ViT-B_16 model

