import os
import random
import pandas as pd

try:
    from src.config import DATA_PATHS
except ImportError:
    
    DATA_PATHS = {
        "TRAIN_ARABIC_OCR_MANIFEST": os.path.join("data", "processed", "ARABIC", "train.csv"),
        "EVAL_ARABIC_OCR_MANIFEST": os.path.join("data", "processed", "ARABIC", "test.csv")
    }


SYNTH_ARA_DIR = os.path.join("data", "synthetic", "ara_ocr_prescription")
SYNTH_ARA_LABELS = os.path.join(SYNTH_ARA_DIR, "labels.csv")

def main():
    print("="*60)
    print(" Building HATFormer Arabic OCR Dataset (Synthetic Only)")
    print("="*60)

    random.seed(42)

    if not os.path.exists(SYNTH_ARA_LABELS):
        print(f"[ERROR] Synthetic labels not found at: {SYNTH_ARA_LABELS}")
        print("Please run the Arabic OCR generator first!")
        return

    
    df_synth = pd.read_csv(SYNTH_ARA_LABELS)
    
    
    if "filename" not in df_synth.columns or "label" not in df_synth.columns:
        if "text" in df_synth.columns: 
            pass 
        else:
            print("[ERROR] CSV must contain a label or text column.")
            return

    
    ocr_data = []
    for _, row in df_synth.iterrows():
        # Get filename and ground truth
        fname = str(row["filename"]).strip() if "filename" in row else str(row.get("image_path", ""))
        text = str(row["label"]).strip() if "label" in row else str(row.get("text", ""))
        
        full_path = os.path.join(SYNTH_ARA_DIR, fname)
        if not os.path.exists(full_path):
            continue  

        ocr_data.append({
            "file_path": full_path,
            "text": text,
            "source": "synth_arabic"
        })

    print(f"\n[INFO] Loaded {len(ocr_data)} valid synthetic image-text pairs.")

    if not ocr_data:
        print("[ERROR] No valid data found to process.")
        return

    
    df_all = pd.DataFrame(ocr_data)
    df_all = df_all.sample(frac=1, random_state=42).reset_index(drop=True)

    
    split_idx = int(len(df_all) * 0.9)
    df_train = df_all.iloc[:split_idx].copy()
    df_eval = df_all.iloc[split_idx:].copy()

    df_train["split"] = "train"
    df_eval["split"] = "eval"

    
    train_save_path = DATA_PATHS.get("TRAIN_ARABIC_OCR_MANIFEST", os.path.join("data", "processed", "ARABIC", "train.csv"))
    eval_save_path = DATA_PATHS.get("EVAL_ARABIC_OCR_MANIFEST", os.path.join("data", "processed", "ARABIC", "test.csv"))

    os.makedirs(os.path.dirname(train_save_path), exist_ok=True)
    os.makedirs(os.path.dirname(eval_save_path), exist_ok=True)

    df_train.to_csv(train_save_path, index=False)
    df_eval.to_csv(eval_save_path, index=False)

    print("\n" + "="*60)
    print(f" Successfully exported Train/Eval split!")
    print("="*60)

if __name__ == "__main__":
    main()
