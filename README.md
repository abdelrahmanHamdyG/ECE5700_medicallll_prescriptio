# Egyptian Medical Prescription OCR System

This project is an OCR pipeline built specifically to handle handwritten Egyptian medical prescriptions. It extracts the English medicine names and their corresponding Arabic instructions, and structures them into a clean JSON format.

Since medical prescriptions can be highly unstructured and noisy (doctor signatures, overlapping text, bad lighting, etc.), we use a multi-step approach combining different specialized models to tackle the problem piece by piece.

## How It Works

1. **Text Detection (CRAFT)**: Finds all the text bounding boxes on the page.
2. **Language Classification (MobileNetV3)**: Looks at each cropped word and figures out if it's English or Arabic.
3. **Medical English OCR (TrOCR + LoRA)**: Reads the English medicine names.
4. **Arabic Instructions OCR (HATFormer)**: Reads the Arabic medical instructions (like dosages and timing).
5. **Structuring (Gemini 1.5 Flash)**: Takes all the scattered English and Arabic text, looks at where they were placed on the page, and smartly pairs each medicine with its instruction while filtering out the noise.

## Project Structure

```text
egyptian_prescription_final/
├── checkpoints/              # Model weights go here
├── data/
│   ├── processed/            # Pandas CSV sheets for train/test splits
│   ├── raw/                  # Real datasets, crops, and evaluation benchmarks
│   └── synthetic/            # Output folders for our synthetic generators
├── resources/
│   ├── fonts/                # Arabic and English fonts used for data generation
│   └── lexicons/             # Lists of medicines and common Arabic instructions
├── src/
│   ├── config.py             # Main configuration for paths and generator settings
│   ├── data/                 # PyTorch datasets 
│   ├── generators/           # Scripts to generate synthetic prescription crops
│   ├── scripts/              # Helper scripts to build CSVs and download models
│   ├── training/             # Clean, stripped-down training scripts
│   └── utils/                # Extras like CRAFT batch utilities
├── test_pipeline.py          # The main script you run to process an image
└── requirements.txt          # Python packages you need
```

## Setup & Installation

I recommend using a Conda environment (like `ocr310`) to keep things clean and avoid conflicts.

**1. Install packages:**
```bash
pip install -r requirements.txt
```

**2. Setup Gemini:**
To use the LLM structuring step, you need a Gemini API key. Just create a `.env` file in the main folder and add:
```text
GEMINI_API_KEY=your_api_key_here
```

**3. Download the Models:**
The repo doesn't include the heavy model weights by default. We have a handy script that downloads the best trained models directly from Google Drive into the `checkpoints/` folder.
```bash
python -m src.scripts.download_models
```

## Running the Pipeline

Once the models are downloaded, you can test the system on an image:

```bash
python test_pipeline.py path/to/your/image.jpeg
```
This will run the detection, OCR, and Gemini steps, printing out the structured JSON at the end!

## Training the Models

If you ever want to retrain the models, the codebase is setup to train entirely on synthetic data. We've removed validation steps during training to keep things fast, straightforward, and memory-efficient.

**Step 1: Generate synthetic images**
```bash
python -m src.generators.generate_english_ocr_prescription
python -m src.generators.generate_arabic_ocr_prescription
```

**Step 2: Build train/test CSVs**
```bash
python -m src.scripts.build_classifier_dataset
python -m src.scripts.build_english_ocr_dataset
python -m src.scripts.build_hatformer_arabic_dataset
```

**Step 3: Run the trainers**
(Checkpoints will automatically save to the `checkpoints/` folder at the end of each epoch).
```bash
python -m src.training.lang_classifier_trainer
python -m src.training.english_ocr_trainer
python -m src.training.hatformer_arabic_trainer_v2
```
