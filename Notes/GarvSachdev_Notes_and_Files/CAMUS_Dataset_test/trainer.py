import torch
import numpy as np
from scipy.ndimage import distance_transform_edt
from scipy.spatial.distance import directed_hausdorff

def train_one_epoch(model, dataloader, criterion, optimizer, device):
    """
    Train the model for one epoch.
    """
    model.train()
    total_loss = 0.0

    for batch in dataloader:
        images = batch["image"].to(device)
        labels = batch["label"].to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)

    return total_loss / len(dataloader.dataset)


def evaluate(model, dataloader, criterion, device):
    """
    Evaluate the model on the validation or test dataset.
    """
    model.eval()
    total_loss, total_dice, total_iou, total_hausdorff_95 = 0.0, 0.0, 0.0, 0.0
    num_samples = 0

    with torch.no_grad():
        for batch in dataloader:
            images = batch["image"].to(device)
            labels = batch["label"].to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item() * images.size(0)

            preds = torch.argmax(torch.softmax(outputs, dim=1), dim=1).cpu().numpy()
            labels = labels.cpu().numpy()

            for pred, label in zip(preds, labels):
                total_dice += dice_coeff(pred, label)
                total_iou += iou_score(pred, label)
                total_hausdorff_95 += hausdorff_distance(label, pred)
                num_samples += 1

    return (total_loss / num_samples,
            total_dice / num_samples,
            total_iou / num_samples,
            total_hausdorff_95 / num_samples)

def dice_coeff(pred, target, epsilon=1e-6):
    """
    Compute the DICE coefficient for multi-class segmentation.
    """
    num_classes = max(pred.max(), target.max()) + 1
    dice_scores = []
    for c in range(num_classes):
        pred_c = (pred == c).astype(np.uint8)
        target_c = (target == c).astype(np.uint8)

        intersection = (pred_c * target_c).sum()
        union = pred_c.sum() + target_c.sum()
        dice_scores.append((2.0 * intersection + epsilon) / (union + epsilon))
    return np.mean(dice_scores)

def iou_score(pred, target, epsilon=1e-6):
    """
    Compute the Intersection over Union (IoU) score for multi-class segmentation.
    """
    num_classes = max(pred.max(), target.max()) + 1
    iou_scores = []
    for c in range(num_classes):
        pred_c = (pred == c).astype(np.uint8)
        target_c = (target == c).astype(np.uint8)

        intersection = (pred_c * target_c).sum()
        union = pred_c.sum() + target_c.sum() - intersection
        iou_scores.append((intersection + epsilon) / (union + epsilon))
    return np.mean(iou_scores)

def hausdorff_distance(label, pred):
    """
    Compute the 95th percentile of the Hausdorff Distance between the label and prediction.
    """
    label_points = np.argwhere(label > 0)
    pred_points = np.argwhere(pred > 0)

    if label_points.size == 0 or pred_points.size == 0:
        return np.inf

    # Compute distance transform for both label and prediction
    label_distances = distance_transform_edt(1 - label)[pred_points[:, 0], pred_points[:, 1]]
    pred_distances = distance_transform_edt(1 - pred)[label_points[:, 0], label_points[:, 1]]

    # Combine distances and take the 95th percentile
    all_distances = np.concatenate([label_distances, pred_distances])
    return np.percentile(all_distances, 95)
