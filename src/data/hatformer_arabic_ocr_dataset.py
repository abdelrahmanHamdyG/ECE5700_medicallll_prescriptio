import random
import re
from typing import Optional

import pandas as pd
from PIL import Image, ImageFilter, ImageOps
from torch.utils.data import Dataset
import torchvision.transforms as T
from transformers import TrOCRProcessor


# ======================================================
# TEXT NORMALISATION
# ======================================================
def normalize_arabic_text(text: str) -> str:
    text = str(text).strip()
    text = re.sub(r"\s+", " ", text)

    # Arabic character normalisation — same as arabic_ocr_dataset.py
    text = re.sub(r"[إأآا]", "ا", text)
    text = text.replace("ى", "ي")
    text = text.replace("ؤ", "و")
    text = text.replace("ئ", "ي")
    text = text.replace("ة", "ه")

    return text


# ======================================================
# AUGMENTATION HELPERS   
# ======================================================
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
    """Convert to grayscale then back to RGB """
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


def build_real_train_transform() -> T.Compose:
    """
    Augmentation for real Arabic prescription crop images.
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
        T.ColorJitter(brightness=0.12, contrast=0.12),
        RandomGrayscaleRGB(p=0.05),
        RandomEqualize(p=0.10),
        SlightGaussianBlur(p=0.25, radius_min=0.3, radius_max=0.8),
    ])


# ======================================================
# HATFORMER IMAGE PREPROCESSING
# ======================================================
def hatformer_preprocess(img: Image.Image, processor: TrOCRProcessor):
    
    img = img.convert("RGB")

    original_width, original_height = img.size
    new_height = 64
    aspect_ratio = original_width / original_height
    new_width = int(new_height * aspect_ratio)

    resized = img.resize((new_width, new_height), Image.BILINEAR)

    # Arabic flows right-to-left 
    resized = resized.transpose(Image.FLIP_LEFT_RIGHT)

    # Wrap into 384×384 black canvas
    final_width, final_height = 384, 384
    canvas = Image.new("RGB", (final_width, final_height), (0, 0, 0))

    if resized.width <= final_width:
        canvas.paste(resized, (0, 0))
    else:
        # split into 384-wide segments and stack vertically
        segment_width = final_width
        num_segments = (resized.width + segment_width - 1) // segment_width
        for i in range(num_segments):
            left = i * segment_width
            right = min(left + segment_width, resized.width)
            segment = resized.crop((left, 0, right, new_height))
            canvas.paste(segment, (0, i * new_height))

    pixel_values = processor(images=canvas, return_tensors="pt").pixel_values.squeeze(0)
    return pixel_values


# ======================================================
# DATASET CLASS
# ======================================================
class HATFormerArabicOCRDataset(Dataset):

    def __init__(
        self,
        manifest_path: str,
        processor: TrOCRProcessor,
        tokenizer,
        max_target_length: int = 200,
        augment_real_train: bool = True,
    ):
        self.df = pd.read_csv(manifest_path)
        self.processor = processor
        self.tokenizer = tokenizer
        self.max_target_length = max_target_length
        self.real_train_transform = build_real_train_transform() if augment_real_train else None

        required_cols = {"file_path", "text"}
        missing = required_cols - set(self.df.columns)
        if missing:
            raise ValueError(f"Manifest is missing required columns: {missing}")

        print(
            f"[HATFormerArabicOCRDataset] Loaded {len(self.df)} rows "
            f"from {manifest_path}"
        )
        if "source" in self.df.columns:
            print(self.df["source"].value_counts().to_string())

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
        text = normalize_arabic_text(row["text"])
        source = row["source"] if "source" in row else "unknown"

        # ----- image -----
        image = self._load_image(image_path)
        image = self._maybe_augment(image, source)

        # HATFormer preprocessing
        pixel_values = hatformer_preprocess(image, self.processor)

        # ----- text -----
        labels = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.max_target_length,
        ).input_ids

        # Replace padding token id with -100 so loss ignores it
        labels = [
            tok if tok != self.tokenizer.pad_token_id else -100
            for tok in labels
        ]

        import torch
        return {
            "pixel_values": pixel_values,
            "labels": torch.tensor(labels, dtype=torch.long),
            "text": text,
            "source": source,
            "file_path": image_path,
        }
