import os
import glob
import random
import pandas as pd
import zipfile
import urllib.request
from typing import List, Dict


try:
    from src.config import DATA_PATHS
except ImportError:
    # Fallback paths if script run directly without correct working dir
    DATA_PATHS = {
        "SYNTH_ARABIC_CLASSIFER": os.path.join("data", "synthetic", "ara_only_class_2"),
        "SYNTH_ENGLISH_CLASSIFER": os.path.join("data", "synthetic", "eng_only_class_2"),
        "TRAIN_CLASSIFIER_MANIFEST": os.path.join("data", "processed", "CLASSIFIER", "train.csv")
    }

# ======================================================================
# DATASET PATHS 
# ======================================================================

KHATT_REAL_PATH = os.path.join("data", "raw", "real_dataset", "khatt_crops_prescription")
IAM_REAL_PATH = os.path.join("data", "raw", "real_dataset", "Iam_crop")

# Links for automatic downloads
KHATT_GDRIVE_ID = "1rmrGnQKrcNS8F7fg1wKez9e-GJ21sIoC"
IAM_GDRIVE_ID = "1rbo0PsBZTOZU843BWIAJ8dKNdcCN8Kk9"


def download_and_extract_gdrive(file_id: str, dest_folder: str, zip_filename: str):
    """
    Downloads a zip file from Google Drive 
    
    """
    if os.path.exists(dest_folder) and len(os.listdir(dest_folder)) > 0:
        return  

    print(f"\n[INFO] Target folder '{dest_folder}' is missing or empty.")
    print(f"[INFO] Automatically downloading dataset from Google Drive...")

    try:
        import gdown
    except ImportError:
        raise ImportError(
            "The 'gdown' package is required to download the datasets automatically.\n"
            "Please run: pip install gdown"
        )

    url = f'https://drive.google.com/uc?id={file_id}'
    os.makedirs(os.path.dirname(dest_folder), exist_ok=True)
    zip_path = os.path.join(os.path.dirname(dest_folder), zip_filename)

    gdown.download(url, zip_path, quiet=False)

    print(f"[INFO] Extracting {zip_filename}...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(os.path.dirname(dest_folder))
    
    print(f"[SUCCESS] Dataset successfully extracted to {dest_folder}\n")


def load_images_from_folder(folder_path: str, lang_name: str, lang_label: int, limit: int = None) -> List[Dict]:
    
    if not os.path.exists(folder_path):
        print(f"[WARNING] Directory not found: {folder_path}")
        return []

    
    valid_exts = ("*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG")
    image_paths = []
    for ext in valid_exts:
        image_paths.extend(glob.glob(os.path.join(folder_path, ext)))

    
    if not image_paths:
        for ext in valid_exts:
            image_paths.extend(glob.glob(os.path.join(folder_path, "**", ext), recursive=True))

    if not image_paths:
        print(f"[WARNING] No images found inside {folder_path}")
        return []

    # Shuffle 
    random.shuffle(image_paths)
    if limit is not None:
        image_paths = image_paths[:limit]

    data = []
    for img_path in image_paths:
        data.append({
            "file_path": img_path,
            "lang": lang_name,
            "lang_label": lang_label,
            "source": os.path.basename(folder_path)
        })

    return data


def main():
    print("="*60)
    print("Building Refined Balanced Language Classifier Dataset...")
    print("="*60)

    # Automatically fetch and extract datasets if they are missing
    download_and_extract_gdrive(KHATT_GDRIVE_ID, KHATT_REAL_PATH, "khatt_crops_prescription.zip")
    download_and_extract_gdrive(IAM_GDRIVE_ID, IAM_REAL_PATH, "Iam_crop.zip")

    
    random.seed(42)
    classifier_train_data = []

    
    # 1. ARABIC DATA 
    
    print("\n--- Loading ARABIC Data (Label 0) ---")
    
    # Synthetic Arabic 
    synth_ar_path = DATA_PATHS["SYNTH_ARABIC_CLASSIFER"]
    synth_ar_data = load_images_from_folder(
        folder_path=synth_ar_path, 
        lang_name="arabic", 
        lang_label=0, 
        limit=2000
    )
    print(f"Loaded {len(synth_ar_data)} Synthetic Arabic crops.")

    
    real_ar_data = load_images_from_folder(
        folder_path=KHATT_REAL_PATH, 
        lang_name="arabic", 
        lang_label=0, 
        limit=500
    )
    print(f"Loaded {len(real_ar_data)} Real KHATT Arabic crops.")

    arabic_total = synth_ar_data + real_ar_data
    classifier_train_data.extend(arabic_total)
    print(f"-> Total Arabic: {len(arabic_total)}")


    
    # 2. ENGLISH DATA 
    
    print("\n--- Loading ENGLISH Data (Label 1) ---")
    
    
    synth_en_path = DATA_PATHS["SYNTH_ENGLISH_CLASSIFER"]
    synth_en_data = load_images_from_folder(
        folder_path=synth_en_path, 
        lang_name="english", 
        lang_label=1, 
        limit=2000
    )
    print(f"Loaded {len(synth_en_data)} Synthetic English crops.")

    
    real_en_data = load_images_from_folder(
        folder_path=IAM_REAL_PATH, 
        lang_name="english", 
        lang_label=1, 
        limit=500
    )
    print(f"Loaded {len(real_en_data)} Real IAM English crops.")

    english_total = synth_en_data + real_en_data
    classifier_train_data.extend(english_total)
    print(f"-> Total English: {len(english_total)}")


    
    if not classifier_train_data:
        print("\n[ERROR] No data found at all! Dataset was NOT created.")
        return

    df_all = pd.DataFrame(classifier_train_data)
    
    
    df_all = df_all.sample(frac=1, random_state=42).reset_index(drop=True)

    
    split_idx = int(len(df_all) * 0.9)
    df_train = df_all.iloc[:split_idx]
    df_test = df_all.iloc[split_idx:]

    
    train_save_path = DATA_PATHS.get("TRAIN_CLASSIFIER_MANIFEST", "data/processed/CLASSIFIER/train.csv")
    test_save_path = DATA_PATHS.get("TEST_CLASSIFIER_MANIFEST", "data/processed/CLASSIFIER/test.csv")
    os.makedirs(os.path.dirname(train_save_path), exist_ok=True)
    os.makedirs(os.path.dirname(test_save_path), exist_ok=True)

    df_train.to_csv(train_save_path, index=False)
    df_test.to_csv(test_save_path, index=False)

    print("\n" + "="*60)
    print(f" Successfully compiled & split into TRAIN & TEST manifests!")
    print("\n[Train Distribution by Source]")
    print(df_train['source'].value_counts().to_string())
    print("\n[Train Distribution by Language]")
    print(df_train['lang'].value_counts().to_string())
    print("="*60)


if __name__ == "__main__":
    main()
