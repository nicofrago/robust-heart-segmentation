import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from tqdm import tqdm

import torch
import random
from torchvision.transforms import functional as TF  # Correct import

from tqdm import tqdm
from attacks import gen_perturbed_image
from utils import get_data_min_max


# Configuration
NUM_CLASSES = 8
BATCH_SIZE = 16
EPOCH_COUNT = 15
INITIAL_LR = 1e-4


class MedicalScanDataset(Dataset):
    def __init__(self, data_directory):
        self.data_dir = data_directory
        self.scan_files = [f for f in os.listdir(data_directory) if f.endswith('.npz')]

    def __len__(self):
        return len(self.scan_files)

    def __getitem__(self, index):
        file_path = os.path.join(self.data_dir, self.scan_files[index])
        data = np.load(file_path)
        scan = data['image'][None, ...]  # Add channel dim
        mask = data['label']

        return (
            torch.from_numpy(scan.astype(np.float32)),
            torch.from_numpy(mask).long()
        )


class AugmentedDataset(MedicalScanDataset):
    def __getitem__(self, idx):
        image, label = super().__getitem__(idx)

        # Brightness adjustment
        if random.random() > 0.5:
            image = image * random.uniform(0.8, 1.2)
            image = torch.clamp(image, 0, 1)

        # Contrast adjustment
        if random.random() > 0.5:
            mean = image.mean()
            image = (image - mean) * random.uniform(0.8, 1.2) + mean
            image = torch.clamp(image, 0, 1)

        # Horizontal flip
        if random.random() > 0.5:
            image = torch.flip(image, [-1])
            label = torch.flip(label, [-1])

        # Rotation (simplified 90 degree multiples)
        if random.random() > 0.5:
            k = random.randint(0, 3)
            image = torch.rot90(image, k, [-2, -1])
            label = torch.rot90(label, k, [-2, -1])

        return image, label


# Loss Function
class DiceLoss(nn.Module):
    def __init__(self, n_classes, eps=1e-7):
        super(DiceLoss, self).__init__()
        self.n_classes = n_classes
        self.eps = eps

    def forward(self, logits, targets):
        probs = F.softmax(logits, dim=1)
        targets_one_hot = F.one_hot(targets, self.n_classes).permute(0, 3, 1, 2).float()

        dims = (0, 2, 3)
        intersection = torch.sum(probs * targets_one_hot, dims)
        cardinality = torch.sum(probs + targets_one_hot, dims)

        dice = (2. * intersection + self.eps) / (cardinality + self.eps)
        return 1. - dice.mean()

class CombinedLoss(nn.Module):
    def __init__(self, n_classes, ce_weight=1.0, dice_weight=1.0):
        super().__init__()
        self.ce = nn.CrossEntropyLoss()
        self.dice = DiceLoss(n_classes)
        self.ce_weight = ce_weight
        self.dice_weight = dice_weight

    def forward(self, logits, targets):
        ce_loss = self.ce(logits, targets)
        dice_loss = self.dice(logits, targets)
        return self.ce_weight * ce_loss + self.dice_weight * dice_loss

def compute_dice_score(probs, labels, num_classes):
    targets_one_hot = F.one_hot(labels, num_classes=num_classes).permute(0, 3, 1, 2).float()
    dims = (0, 2, 3)
    intersection = torch.sum(probs * targets_one_hot, dims)
    cardinality = torch.sum(probs + targets_one_hot, dims)
    dice = (2. * intersection + 1e-7) / (cardinality + 1e-7)
    return dice.mean().item()

def train(
        model,
        loss_function,
        optimizer,
        train_loader,
        valid_loader,
        test_loader,
        epochs,
        num_classes,
        p_adversial=0.0,
        output_attention:bool=False
):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    minv, maxv = get_data_min_max(train_loader.dataset)
    max_epsilon = 0.03
    max_epsilon = 0.2

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0

        for scans, masks in train_loader:
            scans = scans.to(device)
            b, c, h, w = scans.shape
            masks = masks.to(device)
            adv = False
            if random.random() < p_adversial:
                clean_scans = scans[:b//2, :, :, :].clone()
                epsilon = max_epsilon * random.random()
                # scans_perturbed = gen_perturbed_image(scans[:b//2, :, :, :], masks[:b//2, :, :], loss_function, model, minv, maxv, epsilon=epsilon)
                # adversarial = random.choice(['FSGM', 'DAG'])
                adversarial = 'DAG'
                adv_iterations = 5
                scans_perturbed = gen_perturbed_image(scans[:b//2, :, :, :], masks[:b//2, :, :], loss_function, model, minv, maxv, method=adversarial, epsilon=epsilon, num_classes=NUM_CLASSES, iterations=adv_iterations)
                scans[:b//2, :, :, :] = scans_perturbed
                adv = True

            optimizer.zero_grad()
            if output_attention:
                outputs, attns = model(scans)
            else:
                outputs = model(scans)

            loss = loss_function(outputs, masks)

            if adv and output_attention:
                output_clean, attns_clean = model(clean_scans)
                consistency_loss = F.mse_loss(attns[:b//2], attns_clean)
                loss+= 0.7 * consistency_loss

            loss.backward()
            optimizer.step()

            epoch_loss += loss.item() * scans.size(0)
            print(f"Epoch {epoch+1}/{epochs}, Batch Loss: {loss.item():.4f}")

        epoch_loss /= len(train_data)

        model.eval()
        val_loss = 0.0
        metric_total = 0.0

        with torch.no_grad():
            for scans, masks in valid_loader:
                scans = scans.to(device)
                masks = masks.to(device)

                if output_attention:
                    outputs, attns = model(scans)
                else:
                    outputs = model(scans)

                val_loss += loss_function(outputs, masks).item() * scans.size(0)

                probs = F.softmax(outputs, dim=1)
                encoded_masks = F.one_hot(masks, num_classes=num_classes).permute(0, 3, 1, 2).float()

                dims = (0, 2, 3)
                intersection = torch.sum(probs * encoded_masks, dims)
                union = torch.sum(probs + encoded_masks, dims)
                dice = (2. * intersection + 1e-7) / (union + 1e-7)
                metric_total += dice.mean().item() * scans.size(0)

        val_loss /= len(valid_data)
        val_metric = metric_total / len(valid_data)

        print(f'Epoch {epoch+1}/{epochs}')
        print(f'Training Loss: {epoch_loss:.4f} | Validation Loss: {val_loss:.4f} | Validation Metric: {val_metric:.4f}')
        print('-'*40)

    model.eval()
    test_loss = 0.0
    with torch.no_grad():
        for scans, masks in test_loader:

            if output_attention:
                outputs, attns = model(scans)
            else:
                outputs = model(scans)

            scans = scans.to(device)
            masks = masks.to(device)
            outputs = model(scans)
            test_loss += loss_function(outputs, masks).item() * scans.size(0)

    test_loss /= 484#len(test_data)
    print(f'Final Test Loss: {test_loss:.4f}')
    return model


if __name__ == "__main__":
    # Training Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data_dir = "pack/processed_data"
    train_data = MedicalScanDataset(f'{data_dir}/ct_256/train/npz/')

    valid_data = MedicalScanDataset(f'{data_dir}/ct_256/val/npz/')
    test_data = MedicalScanDataset(f'{data_dir}/ct_256/test/npz/')

    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    valid_loader = DataLoader(valid_data, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)