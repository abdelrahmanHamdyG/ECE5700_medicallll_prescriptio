import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

from src.config import DATA_PATHS
from src.data.classifier_dataset import get_classifier_dataloaders
from src.models.lang_classifier import LanguageClassifier

# ==========================================
# HYPERPARAMETERS & CONFIGURATION
# ==========================================
BATCH_SIZE = 32
NUM_EPOCHS = 40  
LEARNING_RATE = 7e-5  
WEIGHT_DECAY = 7e-2   
NUM_WORKERS = 4
MODEL_SAVE_DIR = "checkpoints/classifier/"

# ==========================================
# MAIN TRAINING PIPELINE
# ==========================================
def train_model():
    print("="*50)
    print(" Initializing Language Classifier Training Pipeline")
    print("="*50)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using Device: {device.type.upper()}")

    os.makedirs(MODEL_SAVE_DIR, exist_ok=True)

    train_csv = DATA_PATHS.get("TRAIN_CLASSIFIER_MANIFEST", "lang_classifier_manifest.csv")
    test_csv = DATA_PATHS.get("TEST_CLASSIFIER_MANIFEST", "lang_classifier_test_manifest.csv")
    
    print("[INFO] Loading Datasets...")
    train_loader, val_loader = get_classifier_dataloaders(
        train_csv=train_csv, 
        test_csv=test_csv, 
        batch_size=BATCH_SIZE, 
        num_workers=NUM_WORKERS
    )

    print("[INFO] Initializing MobileNetV3 Model...")
    model = LanguageClassifier(num_classes=2, pretrained=True).to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    

    scheduler = CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS, eta_min=1e-6)

    start_time = time.time()

    print("\n" + "="*50)
    print("Starting Training Loop")
    print("="*50)

    for epoch in range(1, NUM_EPOCHS + 1):
        # -------------------
        # Training Phase
        # -------------------
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0

        train_bar = tqdm(train_loader, desc=f"Epoch [{epoch}/{NUM_EPOCHS}] Train", leave=False)
        

        for i, (images, labels) in enumerate(train_bar):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()

            train_bar.set_postfix({'loss': f"{loss.item():.4f}"})

        epoch_train_loss = running_loss / total_train
        epoch_train_acc = 100. * correct_train / total_train
        

        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]

        print(f"Epoch [{epoch:02d}/{NUM_EPOCHS:02d}] | LR: {current_lr:.1e} | "
              f"Train Loss: {epoch_train_loss:.4f} | Train Acc: {epoch_train_acc:.2f}%")

        # Checkpoint Saving
        save_path = os.path.join(MODEL_SAVE_DIR, "best_classifier.pth")
        torch.save(model.state_dict(), save_path)
        print(f" Model saved to {save_path}")

    total_time = time.time() - start_time
    print("\n" + "="*50)
    print(" Training Complete!")
    print(f"  Total Time: {total_time // 60:.0f}m {total_time % 60:.0f}s")
    print("="*50)

if __name__ == "__main__":
    train_model()