import os
import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim
from torch.utils.data import DataLoader
from dataset_camus import CAMUSDataset, RandomGenerator
from networks.vit_seg_modeling import VisionTransformer as ViT_seg
from networks.vit_seg_modeling import CONFIGS as CONFIGS_ViT_seg
from trainer import train_one_epoch, evaluate

def main():
    # Configurations
    data_dir = "preprocessed_data"  
    output_dir = "./output"
    vit_model_name = "R50-ViT-B_16" 
    output_size = (224, 224)
    num_epochs = 100
    num_classes = 4
    batch_size = 16
    learning_rate = 1e-4
    img_size = 224
    vit_patches_size = 16

    os.makedirs(output_dir, exist_ok=True)

    train_dataset = CAMUSDataset(data_dir, split="train", transform=RandomGenerator(output_size))
    val_dataset = CAMUSDataset(data_dir, split="val", transform=RandomGenerator(output_size))

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    # Load Model
    config_vit = CONFIGS_ViT_seg[vit_model_name]
    config_vit.n_classes = num_classes
    if vit_model_name.find('R50') != -1:
        config_vit.patches.grid = (int(img_size / vit_patches_size), int(img_size / vit_patches_size))
    model = ViT_seg(config_vit, img_size=224, num_classes=num_classes)
    
    #net.load_from(weights=np.load(config_vit.pretrained_path))
    #model = ViT_seg(img_size=224, num_classes=4, vit_name=vit_model_name)

    vit_path = f"../model/vit_checkpoint/imagenet21k/{vit_model_name}.npz"
    #model.load_from(weights=vit_path)
    model.load_from(weights=np.load(config_vit.pretrained_path))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Training
    best_val_dice = 0.0
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")

        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        print(f"Train Loss: {train_loss:.4f}")

        val_loss, val_dice, val_iou, val_hausdorff_95 = evaluate(model, val_loader, criterion, device)
        print(f"Validation Loss: {val_loss:.4f}, DICE: {val_dice:.4f}, IoU: {val_iou:.4f}, Hausdorff95: {val_hausdorff_95:.4f}")

        if val_dice > best_val_dice:
            best_val_dice = val_dice
            torch.save(model.state_dict(), os.path.join(output_dir, "best_model.pth"))
            print("Saved best model!")

        if (epoch + 1) % 10 == 0:
            model_path = os.path.join(output_dir, f"model_epoch_{epoch + 1}.pth")
            torch.save(model.state_dict(), model_path)
            print(f"Saved model at epoch {epoch + 1}.")

    print("Training Complete. Best Validation DICE:", best_val_dice)

if __name__ == "__main__":
    main()
