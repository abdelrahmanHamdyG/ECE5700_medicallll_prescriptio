import os


GENERATOR_CONFIG = {
    # ----------------------------------------
    # English-Only Generators
    # ----------------------------------------
    "ENGLISH": {
        "OUTPUT_DIR": os.path.join("data", "synthetic", "eng_only"),
        "FONTS_DIR": os.path.join("resources", "fonts", "english_fonts"),
        "FINAL_SIZE": (600, 80),
        "BASE_FONT_SIZE": 58,
        "SAMPLE_SIZE": 6500,
        "PROBS": {
            "ROTATE": 0.5,
            "BLUR": 0.2,
            "SALT_PEPPER": 0.15,
            "MORPHOLOGY": 0.25,
            "INK_NOISE": 0.20,
            "CHAR_DROPOUT": 0.1,
        },
        "WORD_COUNTS": [2, 1, "all"],
        "WORD_WEIGHTS": [0.80, 0.15, 0.05],
    },
    
    "ENGLISH_CLASS": {
        "OUTPUT_DIR": os.path.join("data", "synthetic", "eng_only_class_2"),
        "FONTS_DIR": os.path.join("resources", "fonts", "english_fonts"),
        "FINAL_SIZE": (384, 128),
        "BASE_FONT_SIZE": 58,
        "SAMPLE_SIZE": 6500,
        "PROBS": {
            "ROTATE": 0.5,
            "BLUR": 0.2,
            "SALT_PEPPER": 0.15,
            "MORPHOLOGY": 0.25,
            "INK_NOISE": 0.20,
            "CHAR_DROPOUT": 0.1,
        },
        "WORD_COUNTS": [1, 2, "all"],
        "WORD_WEIGHTS": [0.85, 0.1, 0.05],
    },

    # ----------------------------------------
    # Arabic-Only Generators
    # ----------------------------------------
    "ARABIC": {
        "CSV_FILE": os.path.join("resources", "lexicons", "egyptian_medical_instructions.csv"),
        "OUTPUT_DIR": os.path.join("data", "synthetic", "ara_only"),
        "FINAL_SIZE": (500, 100),
        "BASE_FONT_SIZE": 65,
        "SAMPLE_SIZE": 3500,
        "PROBS": {
            "ROTATE": 0.6,
            "BLUR": 0.2,
            "SALT_PEPPER": 0.15,
            "MORPHOLOGY": 0.9,
            "INK_NOISE": 0.9,
            "WAVE_WARP": 0.05,
        },
    },
    
    "ARABIC_CLASS": {
        "CSV_FILE": os.path.join("resources", "lexicons", "egyptian_medical_instructions.csv"),
        "OUTPUT_DIR": os.path.join("data", "synthetic", "ara_only_class_2"),
        "FINAL_SIZE": (384, 124),
        "BASE_FONT_SIZE": 65,
        "SAMPLE_SIZE": 6500,
        "PROBS": {
            "ROTATE": 0.6,
            "BLUR": 0.2,
            "SALT_PEPPER": 0.15,
            "MORPHOLOGY": 0.9,
            "INK_NOISE": 0.9,
            "WAVE_WARP": 0.05,
        },
    },

    

    # ----------------------------------------
    # Global Asset Configurations
    # ----------------------------------------
    "GLOBAL": {
        "MEDICINES": os.path.join("resources", "lexicons", "medicine_names.csv"),
        "INSTRUCTIONS": os.path.join("resources", "lexicons", "egyptian_medical_instructions.csv"),
        "ENG_FONTS_DIR": os.path.join("resources", "fonts", "english_fonts"),
        "ARA_FONTS_DIR": os.path.join("resources", "fonts", "arabic_fonts"),
        "DATA_ROOT": "data/",
        "SYNTH_ROOT": "data/synthetic/",
        "RAW_ROOT": "data/raw/",
        "TEXT_COLORS": [
            (0, 0, 90),     
            (20, 20, 20),   
            (130, 0, 0),    
        ],
        "SEED": 42
    }
}


# ==========================================
# FILE SYSTEM PATHS FOR TRAINING/EVALUATION
# ==========================================
DATA_PATHS = {
    # --- Synthetic Output Datasets ---
    "SYNTH_ARABIC": os.path.join("data", "synthetic", "ara_ocr_prescription"),
    "SYNTH_ENGLISH": os.path.join("data", "synthetic", "eng_ocr_prescription"),
    "SYNTH_ARABIC_CLASSIFER": os.path.join("data", "synthetic", "ara_only_class_2"),
    "SYNTH_ENGLISH_CLASSIFER": os.path.join("data", "synthetic", "eng_only_class_2"),
    
    # --- Real World Datasets ---
    "KHATT_CROPS_IMGS": os.path.join("data", "raw", "real_dataset", "khatt_words"),
    "IAM_IMGS_ROOT": os.path.join("data", "raw", "real_dataset", "iam", "iam_words", "words"),
    
    # --- Processed CSV Manifests ---
    "TRAIN_ENGLISH_OCR_MANIFEST": os.path.join("data", "processed", "ENGLISH", "train.csv"),
    "EVAL_ENGLISH_OCR_MANIFEST": os.path.join("data", "processed", "ENGLISH", "test.csv"),

    "TRAIN_ARABIC_OCR_MANIFEST": os.path.join("data", "processed", "ARABIC", "train.csv"),
    "EVAL_ARABIC_OCR_MANIFEST": os.path.join("data", "processed", "ARABIC", "test.csv"),
    
    "TRAIN_CLASSIFIER_MANIFEST": os.path.join("data", "processed", "CLASSIFIER", "train.csv"),
    "TEST_CLASSIFIER_MANIFEST": os.path.join("data", "processed", "CLASSIFIER", "test.csv"),
}

