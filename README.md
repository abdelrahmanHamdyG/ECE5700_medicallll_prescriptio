# Egyptian Medical Prescription OCR System

An OCR pipeline that reads handwritten Egyptian medical prescriptions and returns structured JSON pairing each medicine name (English) with its Arabic instruction.

## How It Works

1. **CRAFT** — detects text bounding boxes
2. **MobileNetV3** — classifies each word crop as English or Arabic
3. **TrOCR + LoRA** — reads English medicine names
4. **HATFormer** — reads Arabic instructions
5. **Gemini 1.5 Flash** — pairs medicines with instructions using spatial layout, outputs JSON

## Project Structure

```
src/
├── config.py              # Global paths, hyperparameters, and constants
├── generators/            # Synthetic prescription image generation (Arabic and English)
├── models/
│   └── lang_classifier.py # CNN architecture for English/Arabic word classification
├── scripts/
│   ├── build_*_dataset.py # Builds train/test CSV splits for each model
│   └── download_models.py # Downloads model checkpoints from Google Drive
├── training/
│   ├── lang_classifier_trainer.py       # Trains the language classifier
│   ├── english_ocr_trainer.py           # Fine-tunes TrOCR with LoRA
│   └── hatformer_arabic_trainer_v2.py   # Fine-tunes HATFormer for Arabic OCR
└── utils/
    └── craft_batch_extractor.py         # Runs CRAFT detection and crops text regions
```

> The HATFormer training script is inspired by the original HATFormer repository.

## Requirements

- Windows 10/11
- [Miniconda](https://docs.conda.io/en/latest/miniconda.html)
- Gemini API key — free at [Google AI Studio](https://aistudio.google.com/app/apikey)
- NVIDIA GPU recommended for training, not required for inference

## Installation

Our automated startup scripts natively handle creating a dedicated Conda environment (`ocr310`), dynamically probing whether your system supports CUDA to install the correct PyTorch bindings, and installing all OCR dependencies perfectly without manual headaches!

**1. Clone the repo**
```bash
git clone https://github.com/abdelrahmanHamdyG/ECE5700_medicallll_prescriptio
cd egyptian_prescription_final
```

**2. Run the installer**
```bash
# Windows (Command Prompt / PowerShell)
install.bat

# Linux / Mac (Git Bash)
bash install.sh
```

**3. Add your Gemini API key**
Create a `.env` file in the project root:
```text
GEMINI_API_KEY=your_api_key_here
```

**4. Download model weights**
Now that the installation successfully built the `ocr310` environment, activate it and grab the heavy model weights directly from the cloud natively!
```bash
conda activate ocr310
python -m src.scripts.download_models
```

## Running the Pipeline

```bash
conda activate ocr310
python test_pipeline.py path/to/image.jpeg
```

Three sample images are included for testing: `10.jpeg`, `38.jpeg`, `51.jpeg`

Output format:
```json
[
  { "medicine": "Amoxicillin", "instruction": "حبة كل 8 ساعات لمدة 7 أيام" },
  { "medicine": "Paracetamol", "instruction": "حبتين عند الألم" }
]
```

## Training

Requires an NVIDIA GPU. Trains entirely on synthetic data.

```bash
# Generate synthetic data
python -m src.generators.generate_english_ocr_prescription
python -m src.generators.generate_arabic_ocr_prescription

# Build CSV splits
python -m src.scripts.build_classifier_dataset
python -m src.scripts.build_english_ocr_dataset
python -m src.scripts.build_hatformer_arabic_dataset

# Train
python -m src.training.lang_classifier_trainer
python -m src.training.english_ocr_trainer
python -m src.training.hatformer_arabic_trainer_v2
```

Checkpoints are saved to `checkpoints/` at the end of each epoch.

**PyTorch install fails** — install manually:
```bash
conda activate ocr310
pip install torch==2.5.1 torchvision==0.20.1
pip install -r requirements.txt
```
