import os
import csv
import random
import re
from glob import glob

import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont


from src.config import GENERATOR_CONFIG
GLOBAL_CONFIG = GENERATOR_CONFIG["GLOBAL"]
CONFIG = GENERATOR_CONFIG["ENGLISH"]


CONFIG["MEDICINES_CSV"] = os.path.join("resources", "lexicons", "most_popular_medicines.csv")
CONFIG["CSV_COLUMN_NAME"] = "medicine_name"
CONFIG["LABELS_CSV"] = "labels.csv"
CONFIG["MIN_FONT_SIZE"] = 32
CONFIG["FORM_OPTIONS"] = ["Tab", "Cap", "Syp", "capsules", "syrup"]
CONFIG["STRENGTH_OPTIONS"] = ["10", "50", "100", "25", "50", "400", "500", "200"]
CONFIG["TEXT_COLORS"] = GLOBAL_CONFIG["TEXT_COLORS"]
CONFIG["BG_COLORS"] = [
    (255, 255, 255),
    (252, 252, 250),
    (250, 250, 245)
]
CONFIG["FINAL_W"], CONFIG["FINAL_H"] = CONFIG["FINAL_SIZE"]


if "ADD_FORM" not in CONFIG["PROBS"]:
    CONFIG["PROBS"]["ADD_FORM"] = 0.10
if "ADD_STRENGTH" not in CONFIG["PROBS"]:
    CONFIG["PROBS"]["ADD_STRENGTH"] = 0.10
if "NOTEBOOK_LINES" not in CONFIG["PROBS"]:
    CONFIG["PROBS"]["NOTEBOOK_LINES"] = 0.05


def get_true_text_size(draw, text, font):
    
    bbox = draw.textbbox((0, 0), text, font=font)
    x0, y0, x1, y1 = bbox
    width  = x1 - x0
    height = y1 - y0
    return width, height, x0, y0


class PrescriptionGenerator:
    def __init__(self, config):
        self.config = config
        self.fonts = glob(
            os.path.join(GLOBAL_CONFIG["ENG_FONTS_DIR"], "**", "*.[to][t]f"),
            recursive=True
        )

        if not self.fonts:
            raise ValueError(
                f"No fonts found in: {GLOBAL_CONFIG['ENG_FONTS_DIR']}"
            )

        os.makedirs(self.config["OUTPUT_DIR"], exist_ok=True)
        print(f"Loaded {len(self.fonts)} fonts.")

    # =====================================================
    # CSV / DATA LOADING
    # =====================================================
    def _load_medicines(self):
        csv_path = self.config["MEDICINES_CSV"]
        col_name = self.config["CSV_COLUMN_NAME"]

        encodings_to_try = ["utf-8-sig", "utf-8", "cp1252", "latin-1", "windows-1256"]
        last_error = None

        for enc in encodings_to_try:
            try:
                with open(csv_path, "r", encoding=enc, newline="") as f:
                    reader = csv.DictReader(f)

                    if reader.fieldnames is None:
                        raise ValueError("CSV is empty or has no header.")

                    stripped_to_actual = {}
                    clean_fieldnames = []

                    for name in reader.fieldnames:
                        clean_name = name.strip() if name else name
                        clean_fieldnames.append(clean_name)
                        if clean_name is not None:
                            stripped_to_actual[clean_name] = name

                    if col_name not in clean_fieldnames:
                        raise ValueError(
                            f"Column '{col_name}' not found in CSV.\n"
                            f"Found columns: {clean_fieldnames}"
                        )

                    actual_col = stripped_to_actual[col_name]

                    medicines = []
                    for row in reader:
                        value = row.get(actual_col, "")
                        med = str(value).strip()
                        if med:
                            medicines.append(med)

                if not medicines:
                    raise ValueError("No medicines found in the CSV.")

                print(f"Loaded medicines CSV using encoding: {enc}")
                print(f"Total medicines found: {len(medicines)}")
                return medicines

            except Exception as e:
                last_error = e

        raise ValueError(
            f"Could not read medicines CSV: {csv_path}\n"
            f"Last error: {last_error}"
        )

    def _build_dataset_entries(self, medicines, sample_size):
        n = len(medicines)

        if sample_size <= n:
            return random.sample(medicines, sample_size)

        full_cycles = sample_size // n
        remainder   = sample_size % n

        entries = []
        for _ in range(full_cycles):
            cycle = medicines.copy()
            random.shuffle(cycle)
            entries.extend(cycle)

        if remainder > 0:
            entries.extend(random.sample(medicines, remainder))

        return entries

    # =====================================================
    # TEXT HELPERS
    # =====================================================
    def _clean_text(self, text):
        return re.sub(r"[^a-zA-Z0-9\s\.\-,]", "", str(text)).strip()

    def _augment_medicine_text(self, medicine_name):
        text = self._clean_text(medicine_name)
        if not text:
            return ""

        parts = [text]

        if random.random() < self.config["PROBS"]["ADD_FORM"]:
            parts.append(random.choice(self.config["FORM_OPTIONS"]))

        if random.random() < self.config["PROBS"]["ADD_STRENGTH"]:
            parts.append(random.choice(self.config["STRENGTH_OPTIONS"]))

        return " ".join(parts)

    def _get_text(self, raw_line):
        clean_line = self._augment_medicine_text(raw_line)
        words = clean_line.split()

        if not words:
            return ""

        choice = random.choices(
            self.config["WORD_COUNTS"],
            weights=self.config["WORD_WEIGHTS"]
        )[0]

        selected = words if choice == "all" else words[:choice]
        text = " ".join(selected)

        if random.random() < 0.3:
            text = text.capitalize()
        else:
            text = text.lower()

        if len(text) > 3 and random.random() < self.config["PROBS"]["CHAR_DROPOUT"]:
            chars = list(text)
            idx = random.randint(1, len(chars) - 1)
            chars.pop(idx)
            text = "".join(chars)

        return text

    # =====================================================
    # AUGMENTATIONS
    # =====================================================
    def _apply_morphology(self, img_np):
        kernel = np.ones((2, 2), np.uint8)
        img_inv = cv2.bitwise_not(img_np)

        if random.random() < 0.6:
            img_inv = cv2.dilate(img_inv, kernel, iterations=1)
        else:
            img_inv = cv2.erode(img_inv, kernel, iterations=1)

        return cv2.bitwise_not(img_inv)

    def _apply_ink_texture(self, img_np):
        hsv = cv2.cvtColor(img_np, cv2.COLOR_RGB2HSV)
        h, s, v = cv2.split(hsv)

        noise = np.random.randint(
            0, 100, (img_np.shape[0], img_np.shape[1]), dtype=np.uint8
        )

        ink_mask  = (v < 160)
        fade_mask = ink_mask & (noise > 90)
        v[fade_mask] = np.clip(v[fade_mask] + 20, 0, 255)

        final_hsv = cv2.merge((h, s, v))
        return cv2.cvtColor(final_hsv, cv2.COLOR_HSV2RGB)

    def _add_salt_pepper(self, img_np):
        noise = np.random.rand(img_np.shape[0], img_np.shape[1])
        img_np[noise < 0.003] = 0
        img_np[noise > 0.997] = 255
        return img_np

    def _draw_dotted_notebook_lines(self, draw, width, height):
        line_color = random.choice([
            (173, 216, 230),
            (200, 200, 200),
            (180, 180, 190)
        ])
        spacing = random.randint(32, 48)
        start_y = random.randint(10, min(25, spacing))

        for y in range(start_y, height, spacing):
            for x in range(0, width, 8):
                draw.ellipse([(x, y), (x + 2, y + 2)], fill=line_color)

    # =====================================================
    # FONT / LAYOUT  —  FIXED
    # =====================================================
    def _fit_font_and_lines(self, text, font_path, draw, canvas_w, canvas_h):
        
        # Leave margin
        margin    = 20
        max_text_w = canvas_w - margin * 2
        max_text_h = canvas_h - margin * 2
        line_gap   = 6

        font_size = self.config["BASE_FONT_SIZE"]

        while font_size >= self.config["MIN_FONT_SIZE"]:
            font = ImageFont.truetype(font_path, font_size)

            # Try single line 
            w, h, _, _ = get_true_text_size(draw, text, font)
            if w <= max_text_w and h <= max_text_h:
                return font, [text], line_gap

            #  Try splitting into at most 2 lines
            words = text.split()
            if len(words) >= 2:
                best_lines = None
                best_max_w = None

                for split_idx in range(1, len(words)):
                    line1 = " ".join(words[:split_idx])
                    line2 = " ".join(words[split_idx:])

                    w1, h1, _, _ = get_true_text_size(draw, line1, font)
                    w2, h2, _, _ = get_true_text_size(draw, line2, font)

                    total_h = h1 + h2 + line_gap
                    cur_max_w = max(w1, w2)

                    if cur_max_w <= max_text_w and total_h <= max_text_h:
                        if best_max_w is None or cur_max_w < best_max_w:
                            best_max_w = cur_max_w
                            best_lines = [line1, line2]

                if best_lines is not None:
                    return font, best_lines, line_gap

            font_size -= 1

        # Fallback: minimum font, single line (clipped is better than crash)
        font = ImageFont.truetype(font_path, self.config["MIN_FONT_SIZE"])
        return font, [text], line_gap

    def _draw_centered_text(self, draw, canvas_w, canvas_h, lines, font, fill):
        
        line_gap = 6
        line_metrics = []   # (width, height,x0,y0) 

        for line in lines:
            w, h, x0, y0 = get_true_text_size(draw, line, font)
            line_metrics.append((w, h, x0, y0))

        total_h = sum(m[1] for m in line_metrics) + line_gap * max(0, len(lines) - 1)

        
        current_y = max(0, (canvas_h - total_h) // 2)

        for line, (w, h, x0, y0) in zip(lines, line_metrics):
            draw_x = max(0, (canvas_w - w) // 2)

            
            draw.text((draw_x - x0, current_y - y0), line, font=font, fill=fill)

            current_y += h + line_gap

    # =====================================================
    # IMAGE GENERATION
    # =====================================================
    def generate_image(self, raw_entry):
        text = self._get_text(raw_entry)
        if not text:
            return None, None, None

        final_w = self.config["FINAL_W"]
        final_h = self.config["FINAL_H"]

        # Extra padding 
        pad_w    = 60
        pad_h    = 40
        canvas_w = final_w + pad_w
        canvas_h = final_h + pad_h

        bg_color   = random.choice(self.config["BG_COLORS"])
        text_color = random.choice(self.config["TEXT_COLORS"])

        font_path = random.choice(self.fonts)
        font_name = os.path.basename(font_path)

        img  = Image.new("RGB", (canvas_w, canvas_h), bg_color)
        draw = ImageDraw.Draw(img)

        if random.random() < self.config["PROBS"]["NOTEBOOK_LINES"]:
            self._draw_dotted_notebook_lines(draw, canvas_w, canvas_h)

        font, lines, _ = self._fit_font_and_lines(
            text=text,
            font_path=font_path,
            draw=draw,
            canvas_w=canvas_w,
            canvas_h=canvas_h
        )

        self._draw_centered_text(
            draw=draw,
            canvas_w=canvas_w,
            canvas_h=canvas_h,
            lines=lines,
            font=font,
            fill=text_color
        )

        #  light rotation 
        if random.random() < self.config["PROBS"]["ROTATE"]:
            angle = random.uniform(-2, 2)
            img   = img.rotate(angle, resample=Image.BICUBIC, fillcolor=bg_color)

        # Centre-crop 
        left   = (canvas_w - final_w) // 2
        top    = (canvas_h - final_h) // 2
        right  = left + final_w
        bottom = top  + final_h
        img    = img.crop((left, top, right, bottom))

        img_np = np.array(img)

        if random.random() < self.config["PROBS"]["MORPHOLOGY"]:
            img_np = self._apply_morphology(img_np)

        if random.random() < self.config["PROBS"]["INK_NOISE"]:
            img_np = self._apply_ink_texture(img_np)

        if random.random() < self.config["PROBS"]["SALT_PEPPER"]:
            img_np = self._add_salt_pepper(img_np)

        if random.random() < self.config["PROBS"]["BLUR"]:
            img_np = cv2.GaussianBlur(img_np, (3, 3), 0.5)

        return Image.fromarray(img_np), text, font_name

    # =====================================================
    # MAIN LOOP
    # =====================================================
    def run(self):
        medicines     = self._load_medicines()
        dataset_entries = self._build_dataset_entries(
            medicines,
            self.config["SAMPLE_SIZE"]
        )

        print(f"Total output samples to generate: {len(dataset_entries)}")

        out_csv = os.path.join(self.config["OUTPUT_DIR"], self.config["LABELS_CSV"])

        with open(out_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["filename", "text", "font", "original_medicine"])

            for i, entry in enumerate(dataset_entries):
                try:
                    img, label, font_used = self.generate_image(entry)
                    if img is None:
                        continue

                    fname     = f"presc_{i:06d}.png"
                    save_path = os.path.join(self.config["OUTPUT_DIR"], fname)
                    img.save(save_path)

                    writer.writerow([fname, label, font_used, entry])

                    if i > 0 and i % 200 == 0:
                        print(f"Generated {i}/{len(dataset_entries)} images...")

                except Exception as e:
                    print(f"Error on index {i}: {e}")

        print("Dataset generation completed successfully!")


if __name__ == "__main__":
    generator = PrescriptionGenerator(CONFIG)
    generator.run()