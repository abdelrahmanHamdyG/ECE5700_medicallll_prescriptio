import os
import cv2
import re
import numpy as np
import torch
import warnings
from dotenv import load_dotenv
from PIL import Image
from torchvision import transforms


warnings.filterwarnings("ignore", category=FutureWarning, module="torch")
warnings.filterwarnings("ignore", category=FutureWarning, module="craft_text_detector")


if "SSL_CERT_FILE" in os.environ and not os.path.exists(os.environ["SSL_CERT_FILE"]):
    del os.environ["SSL_CERT_FILE"]




import torchvision.models.vgg as tv_vgg
try:
    from torchvision.models import VGG16_BN_Weights, VGG16_Weights
    if not hasattr(tv_vgg, "model_urls"):
        tv_vgg.model_urls = {
            "vgg16_bn": VGG16_BN_Weights.IMAGENET1K_V1.url,
            "vgg16": VGG16_Weights.IMAGENET1K_V1.url,
        }
except Exception:
    pass

from craft_text_detector import Craft
import craft_text_detector.craft_utils as craft_utils
from transformers import TrOCRProcessor, VisionEncoderDecoderModel, PreTrainedTokenizerFast
from peft import LoraConfig, get_peft_model
import google.generativeai as genai
import json
from src.models.lang_classifier import LanguageClassifier
from src.data.hatformer_arabic_ocr_dataset import hatformer_preprocess


def patched_adjustResultCoordinates(polys, ratio_w, ratio_h, ratio_net=2):
    if len(polys) > 0:
        for k in range(len(polys)):
            if polys[k] is not None:
                polys[k] = np.array(polys[k]) * np.array([ratio_w * ratio_net, ratio_h * ratio_net])
    return polys
craft_utils.adjustResultCoordinates = patched_adjustResultCoordinates


import craft_text_detector.predict as craft_predict
_original_array = craft_predict.np.array

def safe_array(obj, *args, **kwargs):
    try:
        return _original_array(obj, *args, **kwargs)
    except ValueError as e:
        if "inhomogeneous" in str(e):
            kwargs['dtype'] = object
            return _original_array(obj, *args, **kwargs)
        raise
craft_predict.np.array = safe_array

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def normalize_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"\s+", " ", text).strip()
    return text

def find_latest_checkpoint(checkpoint_dir: str):
    if not os.path.exists(checkpoint_dir):
        return None
    epochs = []
    for d in os.listdir(checkpoint_dir):
        if d.startswith("epoch_"):
            try:
                ep = int(d.split("_")[1])
                epochs.append((ep, d))
            except ValueError:
                pass
    if epochs:
        epochs.sort()
        return os.path.join(checkpoint_dir, epochs[-1][1])
    
    best_path = os.path.join(checkpoint_dir, "best")
    if os.path.exists(best_path):
        return best_path
    return None

class MedicalPrescriptionPipeline:
    def __init__(self):
        print("========= Initializing Pipeline =========")
        self.load_craft()
        self.load_classifier()
        self.load_english_ocr()
        self.load_arabic_ocr()
        self.load_lexicons()
        self.setup_gemini()
        print("========= Pipeline Ready ===============")

    def setup_gemini(self):
        load_dotenv()
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            print("[WARNING] GEMINI_API_KEY not found in .env. Gemini step will fail.")
        else:
            genai.configure(api_key=api_key)
            available = [m.name for m in genai.list_models() if 'generateContent' in m.supported_generation_methods]
            print(f"[INFO] Discovered accessible Gemini models: {available}")
            chosen_model = next((m for m in available if "flash" in m), None) or (available[0] if available else "gemini-1.5-flash")
            chosen_model = chosen_model.replace("models/", "")
            print(f"[INFO] Choosing Gemini Model: {chosen_model}")
            self.gemini_model = genai.GenerativeModel(chosen_model)
            
            

    def load_lexicons(self):
        print("[INFO] Loading lexicons for fuzzy matching...")
        pop_path = os.path.join("resources", "lexicons", "most_popular_medicines.csv")
        all_path = os.path.join("resources", "lexicons", "medicine_names.csv")
        
        def read_csv(path):
            words = set()
            if not os.path.exists(path): return list(words)
            import csv
            
            for enc in ["utf-8-sig", "utf-8", "windows-1256", "latin-1"]:
                try:
                    words.clear()
                    with open(path, "r", encoding=enc, errors="replace" if enc == "latin-1" else "strict") as f:
                        for row in csv.reader(f):
                            for col in row:
                                val = col.strip()
                                if val and val.lower() not in ["name", "medicine"]:
                                    first_word = val.split()[0]
                                    words.add(first_word)
                    return list(words)
                except UnicodeDecodeError:
                    continue
                    
            return list(words)

        popular = read_csv(pop_path)
        all_meds = read_csv(all_path)
        
        # Optimize for difflib
        self.pop_upper = {m.upper(): m for m in popular}
        self.all_upper = {m.upper(): m for m in all_meds}
        print(f"[INFO] Loaded {len(popular)} popular, {len(all_meds)} total medicines.")

    def load_craft(self):
        print("[INFO] Loading CRAFT model...")
        self.temp_junk_dir = "temp_craft_junk"
        self.craft = Craft(
            output_dir=self.temp_junk_dir,
            crop_type="box",
            cuda=(DEVICE.type == "cuda"),
            refiner=True,
            text_threshold=0.6,
            link_threshold=0.335,
            low_text=0.335
        )

    def load_classifier(self):
        print("[INFO] Loading Language Classifier...")
        self.classifier = LanguageClassifier(num_classes=2, pretrained=False).to(DEVICE)
        ckpt_path = os.path.join("checkpoints", "classifier", "best_classifier.pth")
        if os.path.exists(ckpt_path):
            self.classifier.load_state_dict(torch.load(ckpt_path, map_location=DEVICE, weights_only=True))
        else:
            print(f"[WARNING] Classifier weights not found at {ckpt_path}")
        self.classifier.eval()
        
        self.classifier_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def load_english_ocr(self):
        print("[INFO] Loading English OCR (TrOCR + LoRA)...")
        model_name = "microsoft/trocr-large-handwritten"
        checkpoint_path = os.path.join("checkpoints", "ocr", "english_best", "best")
        
        # Load Processor
        self.eng_processor = TrOCRProcessor.from_pretrained(checkpoint_path if os.path.exists(checkpoint_path) else model_name)
        
        # Load Base Model
        self.eng_model = VisionEncoderDecoderModel.from_pretrained(model_name)
        
        # Apply LoRA
        lora_config = LoraConfig(
            r=8,
            lora_alpha=16,
            target_modules=["q_proj", "v_proj"],
            lora_dropout=0.1,
            bias="none",
        )
        self.eng_model.decoder = get_peft_model(self.eng_model.decoder, lora_config)
        
        # Load  LoRA weights
        if os.path.exists(checkpoint_path):
            bin_path = os.path.join(checkpoint_path, "pytorch_model.bin")
            safe_path = os.path.join(checkpoint_path, "model.safetensors")
            if os.path.exists(bin_path):
                self.eng_model.load_state_dict(torch.load(bin_path, map_location="cpu", weights_only=True), strict=False)
            elif os.path.exists(safe_path):
                from safetensors.torch import load_file
                self.eng_model.load_state_dict(load_file(safe_path), strict=False)
        else:
            print(f"[WARNING] English LoRA weights not found at {checkpoint_path}")
            
        self.eng_model.config.decoder_start_token_id = self.eng_processor.tokenizer.cls_token_id
        self.eng_model.config.pad_token_id = self.eng_processor.tokenizer.pad_token_id
        self.eng_model.config.eos_token_id = self.eng_processor.tokenizer.sep_token_id
        self.eng_model.config.max_length = 32
        self.eng_model.config.num_beams = 4
        
        self.eng_model.to(DEVICE)
        self.eng_model.eval()

    def load_arabic_ocr(self):
        print("[INFO] Loading Arabic OCR (HATFormer)...")
        latest_ckpt = os.path.join("checkpoints", "ocr", "arabic_best", "best")
        
        if os.path.exists(latest_ckpt):
            print(f"       -> Checkpoint found: {latest_ckpt}")
            self.ara_processor = TrOCRProcessor.from_pretrained(latest_ckpt)
            
            
            self.ara_tokenizer = PreTrainedTokenizerFast(tokenizer_file="arabic_tokenizer_clean/tokenizer.json")
            self.ara_tokenizer.add_special_tokens({
                "pad_token": "<pad>",
                "eos_token": "</s>",
                "cls_token": "<s>",
                "bos_token": "<s>",
            })
            
            self.ara_model = VisionEncoderDecoderModel.from_pretrained(latest_ckpt)
            
            if hasattr(self.ara_model, "generation_config"):
                self.ara_model.generation_config.pad_token_id = self.ara_tokenizer.pad_token_id
                self.ara_model.generation_config.decoder_start_token_id = self.ara_tokenizer.bos_token_id
                if hasattr(self.ara_tokenizer, "sep_token_id") and self.ara_tokenizer.sep_token_id is not None:
                    self.ara_model.generation_config.eos_token_id = self.ara_tokenizer.sep_token_id
                else:
                    self.ara_model.generation_config.eos_token_id = self.ara_tokenizer.eos_token_id
        else:
            print(f"[WARNING] Arabic checkpoint not found in {latest_ckpt}, loading base...")
            self.ara_processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-handwritten")
            self.ara_tokenizer = PreTrainedTokenizerFast(tokenizer_file="arabic_tokenizer_clean/tokenizer.json")
            self.ara_model = VisionEncoderDecoderModel.from_pretrained("hatformer-muharaf")
            self.ara_model.decoder.resize_token_embeddings(len(self.ara_tokenizer))

        self.ara_model.to(DEVICE)
        self.ara_model.eval()

    def get_crops(self, image_path, image, padding=6):
        prediction_result = self.craft.detect_text(image_path)
        boxes = prediction_result["boxes"]
        img_h, img_w = image.shape[:2]
        crop_data = []

        for i, box in enumerate(boxes):
            box_coords = np.array(box).astype(np.int32)
            x, y, w, h = cv2.boundingRect(box_coords)

            x_min = max(0, x - padding)
            y_min = max(0, y - padding)
            x_max = min(img_w, x + w + padding)
            y_max = min(img_h, y + h + padding)

            crop = image[y_min:y_max, x_min:x_max]
            if crop.size == 0:
                continue
                
            crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            crop_pil = Image.fromarray(crop_rgb)
            crop_data.append({
                "id": i,
                "box": [int(x_min), int(y_min), int(x_max), int(y_max)],
                "image": crop_pil
            })
        return crop_data

    def classify_language(self, crop_pil):
        input_tensor = self.classifier_transform(crop_pil).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            outputs = self.classifier(input_tensor)
            probs = torch.nn.functional.softmax(outputs.data, dim=1)
            confidence, predicted = torch.max(probs, 1)
        lang = "Arabic" if predicted.item() == 0 else "English"
        return lang, confidence.item()

    def run_english_ocr(self, crop_pil):
        pixel_values = self.eng_processor(crop_pil, return_tensors="pt").pixel_values.to(DEVICE)
        with torch.no_grad():
            generated_ids = self.eng_model.generate(
                pixel_values,
                max_length=32,
                num_beams=2,
                do_sample=False
            )
        text = self.eng_processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return normalize_text(text)

    def run_arabic_ocr(self, crop_pil):
        pixel_values = hatformer_preprocess(crop_pil, self.ara_processor).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            kwargs = {}
            if "interpolate_pos_encoding" in self.ara_model.forward.__code__.co_varnames:
                kwargs["interpolate_pos_encoding"] = True
                
            generated_ids = self.ara_model.generate(
                pixel_values,
                generation_config=self.ara_model.generation_config,
                max_new_tokens=200,
                num_beams=1,
                length_penalty=0.0,
                **kwargs
            )
        text = self.ara_tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return normalize_text(text)

    def run_gemini(self, text_snippets, medicine_candidates):
        prompt = (
            "You are a medical prescription parser.\n"
            "You are given noisy OCR results and a list of candidate words extracted via fuzzy matching.\n"
            "\n"
            "IMPORTANT RULES:\n"
            "- The medicine_candidates are NOT a strict dictionary, but they are the strongest hints for correcting noisy English OCR medicine names.\n"
            "- You may use medical knowledge to correct obvious OCR mistakes.\n"
            "- Do NOT invent medicine names from meaningless or heavily corrupted text.\n"
            "\n"
            "LAYOUT & SPATIAL RULES:\n"
            "- Use the provided bounding box coordinates [x_min, y_min, x_max, y_max] to understand layout.\n"
            "- In Egyptian medical prescriptions, the usual pattern is:\n"
            "  1) medicine name in English\n"
            "  2) Arabic dosage/instruction directly below it\n"
            "- Prefer pairing an English medicine block with the nearest plausible Arabic instruction below it.\n"
            "- Text at the very top of the page is often doctor/clinic/header information.\n"
            "- Text at the very bottom of the page is often address, dates, phone numbers, or clinic timings.\n"
            "- Very high or very low text is usually NOT part of the prescription unless it is clearly a medicine-instruction pair.\n"
            "\n"
            "ARABIC INSTRUCTION PATTERNS:\n"
            "- Arabic medical instructions usually consist of words like:\n"
            "  قرص، قرصين، كيس، كيسين، كبسولة، كبسولات، حقنة، شراب، نقط، بخاخ  \n"
            "  قبل، بعد، مع، بدون  \n"
            "  فطار، غداء، عشاء، النوم، صباحًا، مساءً، العصر  \n"
            "  يوميًا، يوم بعد يوم  \n"
            "  مرة، مرتين، ثلاث مرات، كل  \n"
            "  ساعة، ساعات، كل ٨ ساعات، كل ١٢ ساعة  \n"
            "  لمدة، أيام، أسبوع  \n"
            "  عند اللزوم  \n"
            "\n"
            "- Abbreviations and short forms may appear:\n"
            "  ق = قرص  \n"
            "  ك = كبسولة أو كيس  \n"
            "  ن = نقط  \n"
            "\n"
            "- These words may appear in many combinations such as:\n"
            "  - قرص قبل الأكل\n"
            "  - كبسولة بعد العشاء\n"
            "  - نقط كل ٨ ساعات\n"
            "  - مرة يوميًا\n"
            "  - قرص صباحًا ومساءً\n"
            "  - لمدة ٥ أيام\n"
            "\n"
            "- If the Arabic text does NOT resemble these patterns or combinations, it is likely NOT a valid instruction and should be ignored.\n"
            "\n"
            "MEDICINE VALIDATION RULES:\n"
            "- A medicine name should usually be:\n"
            "  1) a close correction of one of the medicine_candidates, OR\n"
            "  2) a clearly recognizable real medicine name.\n"
            "- Do NOT force every English-looking crop to become a medicine.\n"
            "\n"
            "INSTRUCTION VALIDATION RULES:\n"
            "- Only keep a pair if the Arabic text looks like an actual medical instruction using the patterns above.\n"
            "- Ignore Arabic text that looks like:\n"
            "  - header/footer text\n"
            "  - patient info\n"
            "  - dates\n"
            "  - clinic address\n"
            "  - appointment times\n"
            "  - unrelated notes\n"
            "\n"
            "CRITICAL FILTERING RULES:\n"
            "- NOT every crop corresponds to a medicine.\n"
            "- It is completely valid to return fewer medicines than the number of crops.\n"
            "- It is completely valid to return an empty list [].\n"
            "- DO NOT rely blindly on medicine_candidates.\n"
            "\n"
            "Goal:\n"
            "For each valid medicine found, construct the cleanest, most accurate medical instruction by combining:\n"
            "- corrected medicine name\n"
            "- corrected Arabic dosage/instruction\n"
            "\n"
            "Output format (JSON only):\n"
            "[\n"
            "  {\n"
            '    "medicine": "...",\n'
            '    "instruction": "..."\n'
            "  }\n"
            "]\n"
            "\n"
            "Do not explain your reasoning. Output JSON only. Do not output doubtful items.\n"
            "\n"
            "---\n"
            "medicine_candidates: " + str(medicine_candidates) + "\n"
            "Input OCR Data (Crops):\n"
            "```json\n"
            f"{text_snippets}\n"
            "```\n"
        )
        
        print("[INFO] Calling Gemini...")
        
        response_text = ""
        if not hasattr(self, 'gemini_model'):
            response_text = "Error: Gemini model not configured (check your GEMINI_API_KEY in .env)."
        else:
            try:
                response = self.gemini_model.generate_content(prompt)
                response_text = response.text
            except Exception as e:
                response_text = f"Error calling Gemini: {e}"
            
        # Write prompt and response to an external file
        output_file_path = "gemini_output_log.txt"
        with open(output_file_path, "w", encoding="utf-8") as f:
            f.write("================ PLANNING PROMPT ================\n")
            f.write(prompt)
            f.write("\n=================================================\n\n")
            f.write("================ GEMINI OUTPUT ================\n")
            f.write(response_text)
            f.write("\n====================================================\n")
            
        print(f"[INFO] Prompt and output saved to: {output_file_path}")
        
        return response_text

    def process_image(self, image_path):
        print(f"\n======================================")
        print(f"Processing Image: {image_path}")
        print(f"======================================")
        
        image = cv2.imread(image_path)
        if image is None:
            print(f"[ERROR] Could not read {image_path}")
            return
            
        crops = self.get_crops(image_path, image)
        print(f"[INFO] Extracted {len(crops)} crops using CRAFT...")
        print("[INFO] Running Classifier and OCR on crops...")
        output_dir = "pipeline_output_crops"
        os.makedirs(output_dir, exist_ok=True)
        
        results = []
        global_medicine_candidates = set()
        
        for i, crop_data in enumerate(crops):
            crop_pil = crop_data["image"]
            
            # Save the crop
            crop_path = os.path.join(output_dir, f"crop_{i:02d}.jpg")
            crop_pil.save(crop_path)
            
            lang, conf = self.classify_language(crop_pil)
            
            eng_text = None
            ara_text = None

            if 0.4 <= conf <= 0.6:
                # Borderline confidence: run both
                eng_text = self.run_english_ocr(crop_pil)
                ara_text = self.run_arabic_ocr(crop_pil)
            elif lang == "English":
                eng_text = self.run_english_ocr(crop_pil)
            else:
                ara_text = self.run_arabic_ocr(crop_pil)
            
            result_item = {
                "box": crop_data["box"],
                "predicted_language": lang,
                "classifier_confidence": round(conf, 4),
                "saved_path": crop_path
            }
            if eng_text is not None:
                result_item["english_ocr"] = eng_text
            if ara_text is not None:
                result_item["arabic_ocr"] = ara_text

            results.append(result_item)
            
            print(f"  -> Crop {i:02d} | Box {crop_data['box']} | Class: {lang:7s} ({conf:.2f}) | Eng: {eng_text} | Ara: {ara_text}")
            
            if eng_text:
                import difflib
                pop_matches = difflib.get_close_matches(eng_text.upper(), self.pop_upper.keys(), n=5, cutoff=0.3)
                all_matches = difflib.get_close_matches(eng_text.upper(), self.all_upper.keys(), n=5, cutoff=0.3)
                for m in pop_matches: global_medicine_candidates.add(self.pop_upper[m])
                for m in all_matches: global_medicine_candidates.add(self.all_upper[m])
            
        print("\n[INFO] Releasing VRAM for Gemini...")
        for attr in ['eng_model', 'ara_model', 'classifier']:
            if hasattr(self, attr):
                delattr(self, attr)
        import torch, gc
        torch.cuda.empty_cache()
        gc.collect()
        
        print("[INFO] Calling Gemini for structured extraction...")
        import json
        formatted_snippets = json.dumps(results, indent=2, ensure_ascii=False)
        
        final_result = self.run_gemini(formatted_snippets, list(global_medicine_candidates))
        print("\n================ LLM OUTPUT ================\n")
        print(final_result)
        print("\n============================================\n")
        return final_result


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        img_path = sys.argv[1]
    else:
        img_path = "51.jpeg" 
        
    pipeline = MedicalPrescriptionPipeline()
    pipeline.process_image(img_path)
