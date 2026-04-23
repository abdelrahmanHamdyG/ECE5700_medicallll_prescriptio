import os
import csv
import random
import math
import re
import numpy as np
import cv2
import textwrap 
from glob import glob
from PIL import Image, ImageDraw, ImageFont
from src.config import GENERATOR_CONFIG

# =========================
# CONFIGURATION
# =========================

GLOBAL_CONFIG = GENERATOR_CONFIG["GLOBAL"]
CONFIG = GENERATOR_CONFIG["ENGLISH_CLASS"]

class PrescriptionGenerator:
    def __init__(self):
        self.fonts = glob(os.path.join(GLOBAL_CONFIG["ENG_FONTS_DIR"], "**", "*.[to][t]f"), recursive=True)
        os.makedirs(CONFIG["OUTPUT_DIR"], exist_ok=True)
        print(f"Loaded {len(self.fonts)} fonts.")

    def _clean_text(self, text):
        return re.sub(r'[^a-zA-Z0-9\s\.\-,]', '', text)

    def _get_text(self, raw_line):
        clean_line = self._clean_text(raw_line)
        words = clean_line.split()
        if not words: return ""

        choice = random.choices(CONFIG["WORD_COUNTS"], weights=CONFIG["WORD_WEIGHTS"])[0]
        selected = words if choice == "all" else words[:choice]
        text = " ".join(selected)

        if random.random() < 0.3:
            text = text.capitalize()    
        else:
            text = text.lower()

        if len(text) > 3 and random.random() < CONFIG["PROBS"]["CHAR_DROPOUT"]:
            chars = list(text)
            idx = random.randint(1, len(chars)-1)
            chars.pop(idx)
            text = "".join(chars)
            
        return text

    def _apply_morphology(self, img_np):
        kernel = np.ones((2,2), np.uint8) 
        img_inv = cv2.bitwise_not(img_np)
        if random.random() < 0.5:
            img_inv = cv2.dilate(img_inv, kernel, iterations=1)
        else:
            img_inv = cv2.erode(img_inv, kernel, iterations=1)
        return cv2.bitwise_not(img_inv)

    def _apply_ink_texture(self, img_np):
        hsv = cv2.cvtColor(img_np, cv2.COLOR_RGB2HSV)
        h, s, v = cv2.split(hsv)
        noise = np.random.randint(0, 100, (img_np.shape[0], img_np.shape[1]), dtype=np.uint8)
        ink_mask = (v < 200) 
        v[ink_mask & (noise > 70)] += 60 
        final_hsv = cv2.merge((h, s, v))
        return cv2.cvtColor(final_hsv, cv2.COLOR_HSV2RGB)

    def _add_salt_pepper(self, img_np):
        noise = np.random.rand(img_np.shape[0], img_np.shape[1])
        img_np[noise < 0.005] = 0   
        img_np[noise > 0.995] = 255 
        return img_np

    def _draw_dotted_notebook_lines(self, draw, width, height):
        
        line_color = random.choice([(173, 216, 230), (200, 200, 200), (180, 180, 190)])
        spacing = random.randint(30, 50)
        start_y = random.randint(10, spacing)
        
        for y in range(start_y, height, spacing):
            
            for x in range(0, width, 8):
                draw.ellipse([(x, y), (x+2, y+2)], fill=line_color)

    def generate_image(self, raw_entry):
        text = self._get_text(raw_entry)
        if not text: return None, None, None

        
        FINAL_W, FINAL_H = 384, 128
        
        
        pad = 5
        W, H = FINAL_W + pad, FINAL_H + pad
        
        bg_color = random.choice([(255, 255, 255), (252, 252, 250), (250, 250, 245)])
        text_color = random.choice(GLOBAL_CONFIG["TEXT_COLORS"])
        
        font_path = random.choice(self.fonts)
        font_name = os.path.basename(font_path)
        
        
        font_size = CONFIG.get("BASE_FONT_SIZE", 38)
        font = ImageFont.truetype(font_path, font_size)
        
        
        max_text_width = FINAL_W - 20 
        
        longest_word = max(text.split(), key=len) if text.split() else text
        while font.getlength(longest_word) > max_text_width and font_size > 14:
            font_size -= 2
            font = ImageFont.truetype(font_path, font_size)

        char_width = max(1, int(max_text_width / (font_size * 0.5))) 
        wrapped_lines = textwrap.wrap(text, width=char_width)

        
        img = Image.new("RGB", (W, H), bg_color)
        draw = ImageDraw.Draw(img)
        
        # 5% CHANCE 
        if random.random() < 0.05:
            self._draw_dotted_notebook_lines(draw, W, H)
        
        line_spacing = 6
        total_text_height = sum([draw.textbbox((0, 0), line, font=font)[3] for line in wrapped_lines]) + (line_spacing * (len(wrapped_lines) - 1))
        
        current_y = (H - total_text_height) // 2
        
        for line in wrapped_lines:
            draw.text((W // 2, current_y), line, font=font, fill=text_color, anchor="ma")
            current_y += draw.textbbox((0, 0), line, font=font)[3] + line_spacing

        #  ROTATION 
        if random.random() < CONFIG["PROBS"]["ROTATE"]:
            angle = random.uniform(-3, 3) 
            img = img.rotate(angle, resample=Image.BICUBIC, fillcolor=bg_color)

        
        left = (W - FINAL_W) // 2
        top = (H - FINAL_H) // 2
        right = left + FINAL_W
        bottom = top + FINAL_H
        
        img = img.crop((left, top, right, bottom))

        
        img_np = np.array(img)

        if random.random() < CONFIG["PROBS"]["MORPHOLOGY"]:
            img_np = self._apply_morphology(img_np)

        if random.random() < CONFIG["PROBS"]["INK_NOISE"]:
            img_np = self._apply_ink_texture(img_np)

        if random.random() < CONFIG["PROBS"]["SALT_PEPPER"]:
            img_np = self._add_salt_pepper(img_np)

        if random.random() < CONFIG["PROBS"]["BLUR"]:
            img_np = cv2.GaussianBlur(img_np, (3, 3), 0)

        return Image.fromarray(img_np), text, font_name

    def run(self):
        with open(GLOBAL_CONFIG["MEDICINES"], "r", encoding="utf-8") as f:
            reader = csv.reader(f)
            all_data = [row[0].strip() for row in reader if row]
            
        print(f"Total medicines found: {len(all_data)}")
        
        if len(all_data) < CONFIG["SAMPLE_SIZE"]:
            subset_data = all_data
        else:
            subset_data = random.sample(all_data, CONFIG["SAMPLE_SIZE"])

        out_csv = os.path.join(CONFIG["OUTPUT_DIR"], "labels.csv")
        with open(out_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["filename", "text", "font"])

            for i, entry in enumerate(subset_data):
                try:
                    img, label, font_used = self.generate_image(entry)
                    if img is None: continue

                    fname = f"presc_{i:06d}.png"
                    save_path = os.path.join(CONFIG["OUTPUT_DIR"], fname)
                    img.save(save_path)
                    
                    writer.writerow([fname, label, font_used])

                    if i > 0 and i % 500 == 0:
                        print(f"Generated {i}/{len(subset_data)} images...")
                
                except Exception as e:
                    print(f"Error on index {i}: {e}")

        print("Dataset generation completed successfully!")

if __name__ == "__main__":
    PrescriptionGenerator().run()