import os
import time
import random
from typing import List

import torch
from torch.utils.data import DataLoader, WeightedRandomSampler
from torch.optim import AdamW
from tqdm import tqdm

from transformers import (
    TrOCRProcessor,
    VisionEncoderDecoderModel,
    get_cosine_schedule_with_warmup
)

from peft import LoraConfig, get_peft_model

from src.data.english_ocr_dataset import EnglishOCRDataset
from src.config import DATA_PATHS


# ======================================================
# CONFIG
# ======================================================
SEED = 42
MODEL_NAME = "microsoft/trocr-large-handwritten"

TRAIN_MANIFEST = DATA_PATHS["TRAIN_ENGLISH_OCR_MANIFEST"]
EVAL_MANIFEST = DATA_PATHS["EVAL_ENGLISH_OCR_MANIFEST"]

CHECKPOINT_DIR = os.path.join("checkpoints", "ocr", "trocr_large_lora_r16_synth_1000_ill_20_augment")
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

BATCH_SIZE = 4   
NUM_EPOCHS = 10
LEARNING_RATE = 5e-5
WEIGHT_DECAY = 1e-4
NUM_WORKERS = 4
MAX_TARGET_LENGTH = 32

REAL_WEIGHT = 20.0
SYNTHETIC_WEIGHT = 1.0

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ======================================================
# UTILS
# ======================================================
import re

def normalize_text(text: str) -> str:
    text = text.lower()                     # lowercase
    text = re.sub(r"[^a-z0-9\s]", "", text) # remove symbols, keep spaces
    text = re.sub(r"\s+", " ", text).strip()# clean extra spaces
    return text

def set_seed(seed=42):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ======================================================
# SAMPLER
# ======================================================
def build_sampler(dataset):
    weights = []

    for _, row in dataset.df.iterrows():
        source = str(row["source"]).lower()

        if source == "real_train":
            weights.append(REAL_WEIGHT)
        else:
            weights.append(SYNTHETIC_WEIGHT)

    return WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)


class OCRCollator:
    def __call__(self, batch):
        return {
            "pixel_values": torch.stack([b["pixel_values"] for b in batch]),
            "labels": torch.stack([b["labels"] for b in batch]),
        }


# ======================================================
# DATALOADERS
# ======================================================
def get_loaders(processor):
    train_dataset = EnglishOCRDataset(
        manifest_path=TRAIN_MANIFEST,
        processor=processor,
        max_target_length=MAX_TARGET_LENGTH,
        augment_real_train=True
    )

    collator = OCRCollator()

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        collate_fn=collator
    )

    return train_dataset, train_loader


# ======================================================
# TRAIN
# ======================================================
def train():
    set_seed(SEED)

    print("Loading model...")


    processor = TrOCRProcessor.from_pretrained("microsoft/trocr-large-handwritten")

    
    model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-large-handwritten").to(DEVICE)

    # LoRA on decoder    
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.1,
        bias="none",
    )

    model.decoder = get_peft_model(model.decoder, lora_config)


    model.decoder.print_trainable_parameters()

    # Freeze encoder
    for param in model.encoder.parameters():
        param.requires_grad = False

    # =========================
    # Config
    # =========================
    model.config.decoder_start_token_id = processor.tokenizer.cls_token_id
    model.config.pad_token_id = processor.tokenizer.pad_token_id
    model.config.eos_token_id = processor.tokenizer.sep_token_id
    model.config.max_length = MAX_TARGET_LENGTH
    model.config.num_beams = 1

    # =========================
    # Data
    # =========================
    train_ds, train_loader = get_loaders(processor)

    print(f"Train size: {len(train_ds)}")

    # =========================
    # Optimizer
    # =========================
    optimizer = AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY
    )

    total_steps = len(train_loader) * NUM_EPOCHS
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.1 * total_steps),
        num_training_steps=total_steps
    )

    # =========================
    # LOOP
    # =========================
    for epoch in range(NUM_EPOCHS):
        model.train()
        total_loss = 0

        steps_per_epoch = len(train_loader)
        train_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}", ascii=True)

        for step, batch in enumerate(train_bar):
            pixel_values = batch["pixel_values"].to(DEVICE)
            labels = batch["labels"].to(DEVICE)

            optimizer.zero_grad()

            outputs = model(pixel_values=pixel_values, labels=labels)
            loss = outputs.loss

            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            optimizer.step()
            scheduler.step()

            total_loss += loss.item()

            train_bar.set_postfix({'loss': f"{loss.item():.4f}"})

        print(f"Epoch {epoch+1} Complete | Train Loss: {total_loss / len(train_loader):.4f}")
        
        # Save model after each epoch
        save_path = os.path.join(CHECKPOINT_DIR, f"epoch_{epoch+1}")
        os.makedirs(save_path, exist_ok=True)
        model.save_pretrained(save_path)
        processor.save_pretrained(save_path)
        print(f"Saved model to {save_path}")


if __name__ == "__main__":
    train()