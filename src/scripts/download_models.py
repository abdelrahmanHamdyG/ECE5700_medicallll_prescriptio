import os
import zipfile
import gdown

print("="*60)
print(" Downloading Production Models into new Repository...")
print("="*60)

# Links mapping
# 1. Classifier Best path: 1ZfSNAFuSRvsSzPqOajNLs5TMPfD52bHC -> checkpoints/classifier/best_classifier.pth
# 2. Arabic OCR: 11OCAvd-Z_iypsWhvJaObd7yUd1Kl38sL -> checkpoints/ocr/arabic_best/
# 3. English OCR: 1va0LMFcJWfVbiYIl1XMw-NMwG0YRSYVJ -> checkpoints/ocr/english_best/

CLASSIFIER_ID = "1ZfSNAFuSRvsSzPqOajNLs5TMPfD52bHC"
ARABIC_ID = "11OCAvd-Z_iypsWhvJaObd7yUd1Kl38sL"
ENGLISH_ID = "1va0LMFcJWfVbiYIl1XMw-NMwG0YRSYVJ"

CLASSIFIER_DIR = os.path.join("checkpoints", "classifier")
OCR_DIR = os.path.join("checkpoints", "ocr")

os.makedirs(CLASSIFIER_DIR, exist_ok=True)
os.makedirs(OCR_DIR, exist_ok=True)

# 1. DOWNLOAD CLASSIFIER
classifier_path = os.path.join(CLASSIFIER_DIR, "best_classifier.pth")
if not os.path.exists(classifier_path):
    print("\n Downloading Language Classifier weights...")
    gdown.download(id=CLASSIFIER_ID, output=classifier_path, quiet=False)
else:
    print(f" Classifier already exists at {classifier_path}")

# Helper to download and extract ZIP models
def load_zip_model(file_id, folder_name):
    target_extract_dir = os.path.join(OCR_DIR, folder_name)
    zip_path = os.path.join(OCR_DIR, f"{folder_name}.zip")
    
    if os.path.exists(target_extract_dir) and os.path.exists(os.path.join(target_extract_dir, "config.json")):
        print(f" Model {folder_name} already exists.")
        return
        
    print(f"\n Downloading OCR model -> {folder_name}...")
    gdown.download(id=file_id, output=zip_path, quiet=False)
    
    print(f" Extracting {zip_path}...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(target_extract_dir)
    os.remove(zip_path)
    
    print(f" Extracted {folder_name} successfully.")

# 2. DOWNLOAD ARABIC OCR
load_zip_model(ARABIC_ID, "arabic_best")

# 3. DOWNLOAD ENGLISH OCR
load_zip_model(ENGLISH_ID, "english_best")

# 4. DOWNLOAD HATFORMER BASE (For Tokenizer)
def load_hatformer_base():
    if not os.path.exists("arabic_tokenizer_clean"):
        print("\n downloading HATFormer base & Tokenizer...")
        gdown.download(id="1Cd28c6oIS5O1GV0aZFZTQBb7E-ub5lyo", output="hatformer.zip", quiet=False)
        print(" Extracting hatformer.zip...")
        with zipfile.ZipFile("hatformer.zip", 'r') as zip_ref:
            zip_ref.extractall(".")
        os.remove("hatformer.zip")
        print(" Extracted base HATFormer successfully.")
    else:
        print(" Tokenizer exists locally.")

load_hatformer_base()

print("\n" + "="*60)
print(" All Models Downloaded Structurally!")
