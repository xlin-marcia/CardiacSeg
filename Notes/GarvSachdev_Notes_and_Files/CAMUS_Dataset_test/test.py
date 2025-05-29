import os
import torch
import numpy as np
from dataset_camus import CAMUSDataset, RandomGenerator
from networks.vit_seg_modeling import VisionTransformer as ViT_seg
from networks.vit_seg_modeling import CONFIGS as CONFIGS_ViT_seg
from trainer import evaluate
import matplotlib.pyplot as plt

def save_prediction(image, label, prediction, output_dir, case_name):
    """Save the input image, ground truth, and predicted segmentation."""
    os.makedirs(output_dir, exist_ok=True)

    # Save as .npy
    np.save(os.path.join(output_dir, f"{case_name}_image.npy"), image)
    np.save(os.path.join(output_dir, f"{case_name}_label.npy"), label)
    np.save(os.path.join(output_dir, f"{case_name}_prediction.npy"), prediction)

    # Save visualizations
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 3, 1)
    plt.title("Image")
    plt.imshow(image[0], cmap="gray")
    plt.subplot(1, 3, 2)
    plt.title("Ground Truth")
    plt.imshow(label, cmap="jet")
    plt.subplot(1, 3, 3)
    plt.title("Prediction")
    plt.imshow(prediction, cmap="jet")
    plt.savefig(os.path.join(output_dir, f"{case_name}_visualization.png"))
    plt.close()

def main():
    # Configurations
    model_path = "./output/best_model.pth" 
    vit_model_name = "R50-ViT-B_16" 
    output_size = (224, 224)
    batch_size = 16
    data_dir = "preprocessed_data" 
    num_epochs = 100
    num_classes = 4
    img_size = 224
    vit_patches_size = 16
    prediction_output_dir = "./predictions"

    test_dataset = CAMUSDataset(data_dir, split="test", transform=RandomGenerator(output_size))
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    config_vit = CONFIGS_ViT_seg[vit_model_name]
    config_vit.n_classes = num_classes
    if vit_model_name.find('R50') != -1:
        config_vit.patches.grid = (int(img_size / vit_patches_size), int(img_size / vit_patches_size))  
    #net.load_from(weights=np.load(config_vit.pretrained_path))
    #model = ViT_seg(img_size=224, num_classes=4, vit_name=vit_model_name)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ViT_seg(config_vit, img_size=224, num_classes=num_classes)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)

    # Evaluation
    criterion = torch.nn.CrossEntropyLoss()
    test_loss, test_dice, test_iou, test_hausdorff_95 = evaluate(model, test_loader, criterion, device)
    print(f"Test Loss: {test_loss:.4f}, DICE: {test_dice:.4f}, IoU: {test_iou:.4f}, Hausdorff95: {test_hausdorff_95:.4f}")

    with torch.no_grad():
            for batch in test_loader:
                images, labels = batch['image'].to(device), batch['label'].to(device)
                outputs = model(images)
                predictions = torch.argmax(outputs, dim=1).cpu().numpy()

                case_names = batch.get('case_name', ["unknown_case"] * len(predictions))  # Fallback for missing case_name
    
                for i in range(len(predictions)):
                    case_name = batch['case_name'][i]
                    save_prediction(
                        image=images[i].cpu().numpy(),
                        label=labels[i].cpu().numpy(),
                        prediction=predictions[i],
                        output_dir=prediction_output_dir,
                        case_name=case_name,
                    )
    print(f"Predictions saved in {prediction_output_dir}")

if __name__ == "__main__":
    main()
