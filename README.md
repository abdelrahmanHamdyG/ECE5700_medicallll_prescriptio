# Egyptian Medical Prescription OCR System

An OCR pipeline that reads handwritten Egyptian medical prescriptions and returns structured JSON pairing each medicine name (English) with its Arabic instruction.

## How It Works

1. **CRAFT** — detects text bounding boxes
2. **MobileNetV3** — classifies each word crop as English or Arabic
3. **TrOCR + LoRA** — reads English medicine names
4. **HATFormer** — reads Arabic instructions
5. **Gemini 1.5 Flash** — pairs medicines with instructions using spatial layout, outputs JSON

## Requirements

- Windows 10/11
- [Miniconda](https://docs.conda.io/en/latest/miniconda.html)
- Gemini API key — free at [Google AI Studio](https://aistudio.google.com/app/apikey)
- NVIDIA GPU recommended for training, not required for inference

## Installation

**1. Clone the repo**
```bash
git clone <your-repo-url>
cd egyptian_prescription_final
```

**2. Run the installer**
```bash
# Command Prompt / PowerShell
install.bat

# Git Bash
bash install.sh
```

**3. Add your Gemini API key**

Create a `.env` file in the project root:
```
GEMINI_API_KEY=your_api_key_here
```

**4. Download model weights**
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

