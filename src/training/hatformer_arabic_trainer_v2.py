import os
import random
import re
import shutil
import warnings
from contextlib import nullcontext
from typing import List

import torch
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm import tqdm

import torchvision
torchvision.disable_beta_transforms_warning()

from transformers import (
    TrOCRProcessor,
    VisionEncoderDecoderModel,
    PreTrainedTokenizerFast,
    get_inverse_sqrt_schedule,
    GenerationConfig
)

from src.data.hatformer_arabic_ocr_dataset import HATFormerArabicOCRDataset
from src.config import DATA_PATHS


# ======================================================
# CONFIG
# ======================================================
SEED = 42

HATFORMER_MODEL_PATH = "hatformer-muharaf"
HATFORMER_PROCESSOR_PATH = "microsoft/trocr-base-handwritten"
HATFORMER_TOKENIZER_PATH = "arabic_tokenizer_clean/tokenizer.json"

TRAIN_MANIFEST = DATA_PATHS["TRAIN_ARABIC_OCR_MANIFEST"]
EVAL_MANIFEST = DATA_PATHS["EVAL_ARABIC_OCR_MANIFEST"]

CHECKPOINT_DIR = os.path.join(
    "checkpoints",
    "ocr",
    "hatformer_full_finetune_partial_encoder_partial_decoder_khatt"
)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

BATCH_SIZE = 1
NUM_EPOCHS = 10
NUM_WORKERS = 4
MAX_TARGET_LENGTH = 200

GRAD_ACCUM_STEPS = 4
MAX_GRAD_NORM = 1.0

UNFREEZE_LAST_N_ENCODER_BLOCKS = 4  # Unfreeze last N ViT blocks 
UNFREEZE_LAST_N_DECODER_LAYERS = 4  # Unfreeze last N text layers
USE_ADAFACTOR = True    

LR = 2e-5
WEIGHT_DECAY = 0.01
WARMUP_RATIO = 0.05

USE_AMP = True

REAL_WEIGHT = 30.0
SYNTHETIC_WEIGHT = 1.0

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ======================================================
# UTILS
# ======================================================
def set_seed(seed: int = 42):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def normalize_text(text: str) -> str:
    text = str(text).strip()
    text = re.sub(r"\s+", " ", text)
    return text


def decode_labels(label_ids, tokenizer) -> List[str]:
    labels = label_ids.clone()
    labels[labels == -100] = tokenizer.pad_token_id
    return tokenizer.batch_decode(labels, skip_special_tokens=True)


def optimizer_steps_per_epoch(num_batches: int, grad_accum_steps: int) -> int:
    return (num_batches + grad_accum_steps - 1) // grad_accum_steps


def build_optimizer(model, lr: float):
    # Full finetuning: pass all parameters
    params = [p for p in model.parameters() if p.requires_grad]

    if USE_ADAFACTOR:
        from transformers import Adafactor
        return Adafactor(
            params,
            lr=lr,
            scale_parameter=False,
            relative_step=False,
            warmup_init=False,
            weight_decay=WEIGHT_DECAY,
        )
    else:
        return torch.optim.AdamW(
            params,
            lr=lr,
            weight_decay=WEIGHT_DECAY,
        )


def build_scheduler(optimizer, total_optimizer_steps: int):
    # original HATFormer uses Inverse Sqrt Schedule
    warmup_steps = max(1, int(WARMUP_RATIO * total_optimizer_steps))
    return get_inverse_sqrt_schedule(
        optimizer,
        num_warmup_steps=warmup_steps,
    )


def maybe_autocast():
    if USE_AMP and DEVICE.type == "cuda":
        return torch.cuda.amp.autocast()
    return nullcontext()


def get_encoder_blocks(model):
    enc = model.encoder
    if hasattr(enc, "encoder") and hasattr(enc.encoder, "layer"): return enc.encoder.layer
    if hasattr(enc, "layer"): return enc.layer
    if hasattr(enc, "layers"): return enc.layers
    return []


def get_decoder_layers(model):
    dec = model.decoder
    if hasattr(dec, "model") and hasattr(dec.model, "decoder") and hasattr(dec.model.decoder, "layers"):
        return dec.model.decoder.layers
    if hasattr(dec, "decoder") and hasattr(dec.decoder, "layers"): return dec.decoder.layers
    if hasattr(dec, "layers"): return dec.layers
    return []


# ======================================================
# DATA
# ======================================================
def build_sampler(dataset: HATFormerArabicOCRDataset) -> WeightedRandomSampler:
    weights = []
    for _, row in dataset.df.iterrows():
        source = str(row.get("source", "synthetic")).lower()
        weights.append(REAL_WEIGHT if source == "real_train" else SYNTHETIC_WEIGHT)

    return WeightedRandomSampler(
        weights,
        num_samples=len(weights),
        replacement=True,
    )


class OCRCollator:
    def __call__(self, batch):
        return {
            "pixel_values": torch.stack([b["pixel_values"] for b in batch]),
            "labels": torch.stack([b["labels"] for b in batch]),
        }


def get_loaders(processor, tokenizer):
    train_dataset = HATFormerArabicOCRDataset(
        manifest_path=TRAIN_MANIFEST,
        processor=processor,
        tokenizer=tokenizer,
        max_target_length=MAX_TARGET_LENGTH,
        augment_real_train=True,
    )

    sampler = build_sampler(train_dataset)
    collator = OCRCollator()

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        sampler=sampler,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        persistent_workers=(NUM_WORKERS > 0),
        collate_fn=collator,
    )

    return train_dataset, train_loader


# ======================================================
# CHECKPOINT
# ======================================================
def save_checkpoint(model, tokenizer, processor, tag: str, optimizer=None, scheduler=None):
    save_path = os.path.join(CHECKPOINT_DIR, tag)
    os.makedirs(save_path, exist_ok=True)

    model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)
    processor.save_pretrained(save_path)
    
    if optimizer is not None:
        torch.save(optimizer.state_dict(), os.path.join(save_path, "optimizer.pt"))
    if scheduler is not None:
        torch.save(scheduler.state_dict(), os.path.join(save_path, "scheduler.pt"))

    print(f"Saved -> {save_path}")


def find_latest_checkpoint(checkpoint_dir: str):
    if not os.path.exists(checkpoint_dir):
        return None, 0
    
    epochs = []
    for d in os.listdir(checkpoint_dir):
        if d.startswith("epoch_"):
            try:
                ep = int(d.split("_")[1])
                epochs.append(ep)
            except ValueError:
                pass
    
    if epochs:
        latest = max(epochs)
        return os.path.join(checkpoint_dir, f"epoch_{latest}"), latest
    return None, 0


# ======================================================
# TRAIN
# ======================================================
def train():
    set_seed(SEED)

    if DEVICE.type == "cuda":
        torch.backends.cudnn.benchmark = True

    latest_ckpt_path, start_epoch = find_latest_checkpoint(CHECKPOINT_DIR)

    if latest_ckpt_path is not None:
        print(f" Resuming from checkpoint: {latest_ckpt_path}")
        processor = TrOCRProcessor.from_pretrained(latest_ckpt_path)
        tokenizer = PreTrainedTokenizerFast.from_pretrained(latest_ckpt_path)
        model = VisionEncoderDecoderModel.from_pretrained(latest_ckpt_path)
        
    else:
        if not os.path.exists(HATFORMER_MODEL_PATH) or not os.path.exists("arabic_tokenizer_clean"):
            print(" HATFormer model/tokenizer not found locally! Downloading from GDrive...")
            import gdown
            import zipfile
            file_id = "1Cd28c6oIS5O1GV0aZFZTQBb7E-ub5lyo"
            zip_path = "hatformer.zip"
            gdown.download(id=file_id, output=zip_path, quiet=False)
            print(" Extracting hatformer.zip...")
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(".")
            os.remove(zip_path)
            print(" Extracted successfully!")

        print(" Loading HATFormer processor and tokenizer...")
        processor = TrOCRProcessor.from_pretrained(HATFORMER_PROCESSOR_PATH)

        tokenizer = PreTrainedTokenizerFast(tokenizer_file=HATFORMER_TOKENIZER_PATH)
        tokenizer.add_special_tokens({
            "pad_token": "<pad>",
            "eos_token": "</s>",
            "cls_token": "<s>",
            "bos_token": "<s>",
        })

        print(f" Loading base model from: {HATFORMER_MODEL_PATH}")
        model = VisionEncoderDecoderModel.from_pretrained(HATFORMER_MODEL_PATH)
        
        
        model.decoder.resize_token_embeddings(len(tokenizer))

        
        gen_config = GenerationConfig.from_model_config(model.config)
        gen_config.decoder_start_token_id = tokenizer.bos_token_id
        gen_config.pad_token_id = tokenizer.pad_token_id
        

        if hasattr(tokenizer, "sep_token_id") and tokenizer.sep_token_id is not None:
            gen_config.eos_token_id = tokenizer.sep_token_id
        else:
            gen_config.eos_token_id = tokenizer.eos_token_id
            
        model.generation_config = gen_config

    
    model.config.use_cache = False

    #  Freeze everything 
    for p in model.parameters():
        p.requires_grad = False

    #  Unfreeze last N blocks of the Encoder 
    encoder_blocks = get_encoder_blocks(model)
    n_enc = min(UNFREEZE_LAST_N_ENCODER_BLOCKS, len(encoder_blocks) if encoder_blocks else 0)
    for block in list(encoder_blocks)[-n_enc:]:
        for p in block.parameters():
            p.requires_grad = True

    #  Unfreeze last N layers of the Decoder 
    decoder_layers = get_decoder_layers(model)
    n_dec = min(UNFREEZE_LAST_N_DECODER_LAYERS, len(decoder_layers) if decoder_layers else 0)
    for layer in list(decoder_layers)[-n_dec:]:
        for p in layer.parameters():
            p.requires_grad = True

    
    if hasattr(model.decoder, 'lm_head'):
        for p in model.decoder.lm_head.parameters():
            p.requires_grad = True
            
    if hasattr(model.decoder, "final_logits_bias") and isinstance(model.decoder.final_logits_bias, torch.nn.Parameter):
        model.decoder.final_logits_bias.requires_grad = True

    print(f"\n[Stage] Balanced Fine-Tuning: {n_enc} Encoder Blocks & {n_dec} Decoder Layers")

    # PRINT SUMMARY
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable params: {total_params:,}")

    model.to(DEVICE)

    train_ds, train_loader = get_loaders(processor, tokenizer)
    print(f"Train size : {len(train_ds)}")

    steps_per_epoch = optimizer_steps_per_epoch(len(train_loader), GRAD_ACCUM_STEPS)
    total_optimizer_steps = steps_per_epoch * NUM_EPOCHS

    optimizer = build_optimizer(model, lr=LR)
    scheduler = build_scheduler(optimizer, total_optimizer_steps=total_optimizer_steps)

    if latest_ckpt_path is not None:
        opt_path = os.path.join(latest_ckpt_path, "optimizer.pt")
        sched_path = os.path.join(latest_ckpt_path, "scheduler.pt")
        if os.path.exists(opt_path):
            optimizer.load_state_dict(torch.load(opt_path, map_location=DEVICE))
            print("Loaded optimizer state.")
        if os.path.exists(sched_path):
            scheduler.load_state_dict(torch.load(sched_path, map_location=DEVICE))
            print("Loaded scheduler state.")

    scaler = torch.cuda.amp.GradScaler(enabled=(USE_AMP and DEVICE.type == "cuda"))

    for epoch in range(start_epoch, NUM_EPOCHS):
        model.train()
        optimizer.zero_grad(set_to_none=True)

        running_loss = 0.0

        train_bar = tqdm(
            enumerate(train_loader),
            total=len(train_loader),
            desc=f"Epoch {epoch + 1}/{NUM_EPOCHS}",
            ascii=True
        )

        for step, batch in train_bar:
            pixel_values = batch["pixel_values"].to(DEVICE, non_blocking=True)
            labels = batch["labels"].to(DEVICE, non_blocking=True)

            with maybe_autocast():
                
                outputs = model(
                    pixel_values=pixel_values, 
                    labels=labels,
                    interpolate_pos_encoding=True
                )
                loss = outputs.loss / GRAD_ACCUM_STEPS

            scaler.scale(loss).backward()

            should_step = ((step + 1) % GRAD_ACCUM_STEPS == 0) or ((step + 1) == len(train_loader))

            if should_step:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad],
                    MAX_GRAD_NORM
                )
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()

            running_loss += loss.item() * GRAD_ACCUM_STEPS
            train_bar.set_postfix(loss=f"{loss.item() * GRAD_ACCUM_STEPS:.4f}")

        avg_train_loss = running_loss / max(len(train_loader), 1)

        print(f"\nEpoch {epoch + 1} | Train Loss: {avg_train_loss:.4f}")

        
        last_epoch_path = os.path.join(CHECKPOINT_DIR, f"epoch_{epoch}")
        if os.path.exists(last_epoch_path):
            shutil.rmtree(last_epoch_path)

        save_checkpoint(model, tokenizer, processor, tag=f"epoch_{epoch + 1}", optimizer=optimizer, scheduler=scheduler)

    print(f"\n Training complete.")


if __name__ == "__main__":
    train()
