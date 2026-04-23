import random
import re
from typing import Optional

import pandas as pd
from PIL import Image, ImageFilter, ImageOps
from torch.utils.data import Dataset
import torchvision.transforms as T


def normalize_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", "", text)   # keep spaces
    text = re.sub(r"\s+", " ", text).strip()
    return text


class SlightGaussianBlur:
    def __init__(self, p: float = 0.25, radius_min: float = 0.3, radius_max: float = 0.8):
        self.p = p
        self.radius_min = radius_min
        self.radius_max = radius_max

    def __call__(self, img: Image.Image) -> Image.Image:
        if random.random() < self.p:
            radius = random.uniform(self.radius_min, self.radius_max)
            return img.filter(ImageFilter.GaussianBlur(radius=radius))
        return img


class RandomGrayscaleRGB:
    
    def __init__(self, p: float = 0.2):
        self.p = p

    def __call__(self, img: Image.Image) -> Image.Image:
        if random.random() < self.p:
            img = ImageOps.grayscale(img).convert("RGB")
        return img


class RandomAutoContrast:
    def __init__(self, p: float = 0.2):
        self.p = p

    def __call__(self, img: Image.Image) -> Image.Image:
        if random.random() < self.p:
            img = ImageOps.autocontrast(img)
        return img


class RandomEqualize:
    def __init__(self, p: float = 0.1):
        self.p = p

    def __call__(self, img: Image.Image) -> Image.Image:
        if random.random() < self.p:
            img = ImageOps.equalize(img)
        return img


def build_real_train_transform():
    """
    Augmentation for real prescription images only.
    """
    return T.Compose([
        T.RandomAffine(
            degrees=3,
            translate=(0.02, 0.02),
            scale=(0.95, 1.05),
            shear=2,
            interpolation=T.InterpolationMode.BILINEAR,
            fill=255,
        ),
        T.ColorJitter(
            brightness=0.12,
            contrast=0.12,
        ),
        RandomGrayscaleRGB(p=0.20),
        RandomAutoContrast(p=0.20),
        RandomEqualize(p=0.10),
        SlightGaussianBlur(p=0.25, radius_min=0.3, radius_max=0.8),
    ])


class EnglishOCRDataset(Dataset):
    """
    Dataset for TrOCR OCR training/evaluation.

    """
    def __init__(
        self,
        manifest_path: str,
        processor,
        max_target_length: int = 32,
        augment_real_train: bool = True,
    ):
        self.df = pd.read_csv(manifest_path)
        self.processor = processor
        self.max_target_length = max_target_length
        self.real_train_transform = build_real_train_transform() if augment_real_train else None

        required_cols = {"file_path", "text"}
        missing = required_cols - set(self.df.columns)
        if missing:
            raise ValueError(f"Manifest is missing required columns: {missing}")

    def __len__(self):
        return len(self.df)

    def _load_image(self, image_path: str) -> Image.Image:
        return Image.open(image_path).convert("RGB")

    def _maybe_augment(self, image: Image.Image, source: str) -> Image.Image:
        if source == "real_train" and self.real_train_transform is not None:
            image = self.real_train_transform(image)
        return image

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]

        image_path = row["file_path"]
        text = normalize_text(str(row["text"]))
        source = row["source"] if "source" in row else "unknown"

        image = self._load_image(image_path)
        image = self._maybe_augment(image, source)

        pixel_values = self.processor(
            images=image,
            return_tensors="pt"
        ).pixel_values.squeeze(0)

        labels = self.processor.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.max_target_length,
            return_tensors="pt"
        ).input_ids.squeeze(0)

        labels[labels == self.processor.tokenizer.pad_token_id] = -100

        return {
            "pixel_values": pixel_values,
            "labels": labels,
            "text": text,
            "source": source,
            "file_path": image_path,
        }