import os
import numpy as np
import torch
from torch.utils.data import Dataset
from scipy.ndimage import zoom, rotate
import random

class RandomGenerator(object):
    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        image, label = sample['image'], sample['label']

        # Random rotation and flip
        if random.random() > 0.5:
            k = random.randint(0, 3)
            image = np.rot90(image, k)
            label = np.rot90(label, k)
        if random.random() > 0.5:
            axis = random.randint(0, 1)
            image = np.flip(image, axis=axis).copy()
            label = np.flip(label, axis=axis).copy()

        # Random rotate
        if random.random() > 0.5:
            angle = random.uniform(-20, 20)
            image = rotate(image, angle, reshape=False, order=1, mode='nearest')
            label = rotate(label, angle, reshape=False, order=0, mode='nearest')

        # Resize to output size
        x, y = image.shape
        if x != self.output_size[0] or y != self.output_size[1]:
            image = zoom(image, (self.output_size[0] / x, self.output_size[1] / y), order=3)
            label = zoom(label, (self.output_size[0] / x, self.output_size[1] / y), order=0)

        # Convert to PyTorch tensors
        # image = torch.from_numpy(image).float().unsqueeze(0)  # Add channel dimension
        image = torch.from_numpy(image.copy()).float().unsqueeze(0)  # Add channel dimension
        # label = torch.from_numpy(label).long()
        label = torch.from_numpy(label.copy()).long()  # Ensure no negative strides
        return {'image': image, 'label': label}

class CAMUSDataset(Dataset):
    def __init__(self, base_dir, split, transform=None):
        """
        Args:
            base_dir (str): Path to preprocessed data directory.
            split (str): One of 'train', 'val', 'test'.
            transform (callable, optional): Transformations to apply to the data.
        """
        self.base_dir = os.path.join(base_dir, split)
        self.transform = transform
        self.images = sorted([f for f in os.listdir(self.base_dir) if not f.endswith('_gt.npy')])
        self.labels = [f.replace('.npy', '_gt.npy') for f in self.images]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image_path = os.path.join(self.base_dir, self.images[idx])
        label_path = os.path.join(self.base_dir, self.labels[idx])
        case_name = os.path.basename(image_path).split('.')[0]

        # Load images and labels
        image = np.load(image_path).astype(np.float32)
        label = np.load(label_path).astype(np.uint8)

        sample = {'image': image, 'label': label}
        if self.transform:
            sample = self.transform(sample)
        sample["case_name"] = case_name

        return sample

if __name__ == "__main__":
    base_dir = "preprocessed_data"
    output_size = (224, 224)

    train_dataset = CAMUSDataset(base_dir, split='train', transform=RandomGenerator(output_size))
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=4)

    for batch in train_loader:
        images = batch['image']
        labels = batch['label']
        print("Batch size:", images.size(), labels.size())
        break
