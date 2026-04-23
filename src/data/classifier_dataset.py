import os
import torch
import pandas as pd
import numpy as np
import cv2
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import kornia.augmentation as K
import torch.nn as nn


# ==========================================
# 1. PYTORCH DATASET CLASS
# ==========================================
class LanguageClassifierDataset(Dataset):

    def __init__(self, manifest_path: str, transform=None):
        if not os.path.exists(manifest_path):
            raise FileNotFoundError(f"Manifest not found: {manifest_path}")

        self.df = pd.read_csv(manifest_path)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = row['file_path']
        label = row['lang_label']

        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            image = Image.new('RGB', (224, 224), color='white')

        if self.transform:
            image = self.transform(image)

        return image, torch.tensor(label, dtype=torch.long)


# ==========================================
# 2. CUSTOM AUGMENTATIONS
# ==========================================

import random
from PIL import ImageDraw

class RandomRuledLines:
    def __init__(self, p=0.05, max_lines=3, dotted_prob=0.5):
        self.p = p
        self.max_lines = max_lines
        self.dotted_prob = dotted_prob

    def __call__(self, img):
        if random.random() > self.p:
            return img

        draw = ImageDraw.Draw(img)
        width, height = img.size
        num_lines = random.randint(1, self.max_lines)

        for _ in range(num_lines):
            y = random.randint(0, height-1)
            thickness = random.randint(1, 2)

            if random.random() < self.dotted_prob:
                # draw dotted line
                step = 5
                for x in range(0, width, step*2):
                    draw.line([(x, y), (x+step, y)], fill=(200,200,200), width=thickness)
            else:
                # draw solid line
                draw.line([(0, y), (width, y)], fill=(200,200,200), width=thickness)

        return img
class RandomBinarize:
    def __init__(self, p=0.10):
        self.p = p

    def __call__(self, img):
        if np.random.rand() > self.p:
            return img

        # 1. Convert PIL RGB to OpenCV Grayscale
        img_np = np.array(img)
        gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)

        # 2. Apply Adaptive Thresholding
        # blockSize=21 and C=10 work well for finding text against uneven backgrounds
        binary = cv2.adaptiveThreshold(
            gray, 255, 
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY, 
            21, 10
        )

        # 3. Convert back to RGB so it matches the expected 3-channel input
        binary_rgb = cv2.cvtColor(binary, cv2.COLOR_GRAY2RGB)
        
        return Image.fromarray(binary_rgb)

class KorniaAugmentation(nn.Module):
    """
    Augmentations that must operate on tensors (motion blur, noise).
    """
    def __init__(self):
        super().__init__()
        self.aug = nn.Sequential(

            
            K.RandomMotionBlur(
                kernel_size=7,
                angle=35.0,
                direction=0.5,
                p=0.3
            ),

            
            K.RandomGaussianNoise(
                mean=0.0,
                std=0.02,
                p=0.2
            )
        )

    def forward(self, x):
        return self.aug(x)

class SqueezeKorniaBatch:
    """Removes the dummy batch dimension added by Kornia so torchvision can read it."""
    def __call__(self, tensor):
        if tensor.dim() == 4:
            return tensor.squeeze(0)
        return tensor

def get_classifier_transforms(is_train: bool = True):
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )

    if is_train:
        return transforms.Compose([
            transforms.Resize((224,224)),
            
            
            transforms.Grayscale(num_output_channels=3),
            RandomRuledLines(p=0.03),
            
            RandomBinarize(p=0.7),

            # Slight paper warping
            transforms.RandomApply([
                transforms.RandomRotation(degrees=[-15, 15], fill=255),
            ], p=0.15),
            
            transforms.RandomPerspective(distortion_scale=0.2, p=0.2, fill=255),
            transforms.RandomAffine(
                degrees=3, # Rotation handled above
                translate=(0.05, 0.05),
                scale=(0.8, 1.2),
                shear=10, 
                fill=255
            ),


            transforms.ColorJitter(
                brightness=0.45,
                contrast=0.65,
                saturation=0.3
            ),

            transforms.ToTensor(),
            transforms.RandomApply([
                transforms.RandomChoice([
                    transforms.GaussianBlur(kernel_size=(1, 7), sigma=(0.1, 1.2)), # Horizontal shake
                    transforms.GaussianBlur(kernel_size=(7, 1), sigma=(0.1, 1.2))  # Vertical shake
                ])
            ], p=0.25),

            
            transforms.RandomApply(
                [KorniaAugmentation()], 
                p=0.9
            ),
            
            SqueezeKorniaBatch(),

            transforms.RandomErasing(
                p=0.01,
                scale=(0.01, 0.02),
                ratio=(0.3, 3.0),
                value=1.0
            ),

            normalize
        ])

    else:
        # Deterministic for validation/testing
        return transforms.Compose([
            transforms.Resize((224,224)),
            transforms.Grayscale(num_output_channels=3),
            # RandomBinarize(p=0),
            transforms.ToTensor(),
            normalize
        ])

# ==========================================
# 3. DATALOADER FACTORY
# ==========================================
def get_classifier_dataloaders(
    train_csv: str,
    test_csv: str,
    batch_size: int = 32,
    num_workers: int = 4
):

    train_transform = get_classifier_transforms(is_train=True)
    test_transform = get_classifier_transforms(is_train=False)

    train_dataset = LanguageClassifierDataset(train_csv, transform=train_transform)
    test_dataset = LanguageClassifierDataset(test_csv, transform=test_transform)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    print(f"Train batches: {len(train_loader)}")
    print(f"Test batches: {len(test_loader)}")

    return train_loader, test_loader