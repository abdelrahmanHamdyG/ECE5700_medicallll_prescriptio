import os
import csv
import random
import math
import numpy as np
import cv2
import textwrap # INTEGRATED: For wrapping text
from glob import glob
from PIL import Image, ImageDraw, ImageFont

import arabic_reshaper
from bidi.algorithm import get_display
from src.config import GENERATOR_CONFIG

# =========================
# CONFIGURATION
# =========================
GLOBAL_CONFIG = GENERATOR_CONFIG["GLOBAL"]
CONFIG = GENERATOR_CONFIG["ARABIC_CLASS"]

class ArabicPrescriptionGenerator:
    def __init__(self):
        #  Load Fonts
        self.fonts = glob(os.path.join(GLOBAL_CONFIG["ARA_FONTS_DIR"], "**", "*.[to][t]f"), recursive=True)
        
        #  Load Dataset
        self.dataset = []
        if not os.path.exists(GLOBAL_CONFIG["INSTRUCTIONS"]):
            raise ValueError(f"Missing {GLOBAL_CONFIG['INSTRUCTIONS']}!")
            
        with open(GLOBAL_CONFIG["INSTRUCTIONS"], "r", encoding="utf-8") as f:
            reader = csv.reader(f)
            header = next(reader, None)
            
            for row in reader:
                if row:
                    self.dataset.append(row[0].strip())
        
        if not self.dataset:
            raise ValueError("CSV file is empty or formatted incorrectly.")

        os.makedirs(CONFIG["OUTPUT_DIR"], exist_ok=True)

    def _fix_arabic(self, text):

        configuration = {
            'delete_harakat': False,  
            'support_ligatures': True,
            'RIAL SIGN': False,
        }
        reshaped_text = arabic_reshaper.reshape(text) 
        bidi_text = get_display(reshaped_text)        
        return bidi_text

    def _to_indian_numbers(self, text):
        translation = str.maketrans("0123456789", "٠١٢٣٤٥٦٧٨٩")
        return text.translate(translation)

    def _generate_text_logic(self):
        
        base_text = random.choice(self.dataset)
        words = base_text.split()
        rand_val = random.random()
        
        if rand_val < 0.30:
            final_text = base_text
        elif rand_val < 0.80:
            final_text = random.choice(words) if words else base_text
        else:
            if len(words) >= 2:
                start_idx = random.randint(0, len(words) - 2)
                final_text = " ".join(words[start_idx : start_idx + 2])
            elif words:
                final_text = words[0]
            else:
                final_text = base_text

        return self._to_indian_numbers(final_text) 

    def _draw_dotted_notebook_lines(self, draw, width, height):
        
        line_color = random.choice([(173, 216, 230), (200, 200, 200), (180, 180, 190)])
        spacing = random.randint(30, 50)
        start_y = random.randint(10, spacing)
        
        for y in range(start_y, height, spacing):
            # Draw a dot every 8 pixels across the width
            for x in range(0, width, 8):
                draw.ellipse([(x, y), (x+2, y+2)], fill=line_color)

    def generate_image(self):
        #  Get raw text
        raw_label = self._generate_text_logic()
        
        # Setup Canvas 
        FINAL_W, FINAL_H = 384, 128
        
        # Reduced padding from 40 to 16
        pad = 16 
        W, H = FINAL_W + pad, FINAL_H + pad
        
        bg_color = random.choice([(255, 255, 255), (252, 252, 250), (250, 250, 245)])
        text_color = random.choice(GLOBAL_CONFIG["TEXT_COLORS"])

        available_fonts = self.fonts
        if "ء" in raw_label:
            excluded_fonts = {"a-bad-khat.ttf", "ghalam-1.ttf", "b-shekari.ttf"}
            available_fonts = [f for f in self.fonts if os.path.basename(f) not in excluded_fonts]
        
        font_path = random.choice(available_fonts)
        font_name = os.path.basename(font_path)
        
        #  Fit Font Size
        font_size = CONFIG.get("BASE_FONT_SIZE", 38)
        font = ImageFont.truetype(font_path, font_size)
        
        # Adjusted max width to fit new padding
        max_text_width = FINAL_W - 20 
        
        # Check longest word using fixed Arabic to measure accurately
        words = raw_label.split()
        longest_word = max(words, key=len) if words else raw_label
        fixed_longest = self._fix_arabic(longest_word)

        while font.getlength(fixed_longest) > max_text_width and font_size > 14:
            font_size -= 2
            font = ImageFont.truetype(font_path, font_size)

        
        char_width = max(1, int(max_text_width / (font_size * 0.5)))
        raw_lines = textwrap.wrap(raw_label, width=char_width)
        
        
        display_lines = [self._fix_arabic(line) for line in raw_lines]

        # 5. Draw Text
        img = Image.new("RGB", (W, H), bg_color)
        draw = ImageDraw.Draw(img)
        
        # 5% CHANCE TO DRAW DOTTED NOTEBOOK LINES ---
        if random.random() < 0.05:
            self._draw_dotted_notebook_lines(draw, W, H)
        
        line_spacing = 6
        total_text_height = sum([draw.textbbox((0, 0), line, font=font)[3] for line in display_lines]) + (line_spacing * (len(display_lines) - 1))
        current_y = (H - total_text_height) // 2
        
        for line in display_lines:
            draw.text((W // 2, current_y), line, font=font, fill=text_color, anchor="ma")
            current_y += draw.textbbox((0, 0), line, font=font)[3] + line_spacing

        # Rotate
        if random.random() < CONFIG["PROBS"]["ROTATE"]:
            angle = random.uniform(-3, 3) 
            img = img.rotate(angle, resample=Image.BICUBIC, fillcolor=bg_color)

        #  Crop to Exact Final Size
        left = (W - FINAL_W) // 2
        top = (H - FINAL_H) // 2
        img = img.crop((left, top, left + FINAL_W, top + FINAL_H))

        # Noise Pipeline
        img_np = np.array(img)
        
        # Morphology
        if random.random() < CONFIG["PROBS"]["MORPHOLOGY"]:
            kernel = np.ones((2,2), np.uint8) 
            img_inv = cv2.bitwise_not(img_np)
            morphed = cv2.dilate(img_inv, kernel, iterations=1)
            img_np = cv2.bitwise_not(cv2.addWeighted(img_inv, 0.7, morphed, 0.3, 0))

        # Ink Noise
        if random.random() < CONFIG["PROBS"]["INK_NOISE"]:
            hsv = cv2.cvtColor(img_np, cv2.COLOR_RGB2HSV)
            h, s, v = cv2.split(hsv)
            noise = np.random.randint(0, 100, (img_np.shape[0], img_np.shape[1]), dtype=np.uint8)
            v[(v < 200) & (noise > 70)] += 50 
            img_np = cv2.cvtColor(cv2.merge((h, s, v)), cv2.COLOR_HSV2RGB)

        # Salt & Pepper
        if random.random() < CONFIG["PROBS"]["SALT_PEPPER"]:
            noise = np.random.rand(img_np.shape[0], img_np.shape[1])
            img_np[noise < 0.001] = 0; img_np[noise > 0.999] = 255

        # Blur
        img_np = cv2.GaussianBlur(img_np, (3, 3), 0)

        return Image.fromarray(img_np), raw_label, font_name

    def run(self):
        out_csv = os.path.join(CONFIG["OUTPUT_DIR"], "labels.csv")
        
        with open(out_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["filename", "text", "font"])

            print(f"Generating {CONFIG['SAMPLE_SIZE']} Arabic samples from {len(self.dataset)} unique instructions...")

            for i in range(CONFIG["SAMPLE_SIZE"]):
                try:
                    img, label, font_used = self.generate_image()
                    if img is None: continue
                    
                    fname = f"ara_{i:06d}.png"
                    img.save(os.path.join(CONFIG["OUTPUT_DIR"], fname))
                    writer.writerow([fname, label, font_used])
                    
                    if i % 500 == 0 and i > 0: 
                        print(f"Progress: {i}/{CONFIG['SAMPLE_SIZE']}")
                except Exception as e:
                    print(f"Error skipping sample {i}: {e}")
            
            print("Done! Check folder:", CONFIG["OUTPUT_DIR"])

if __name__ == "__main__":
    ArabicPrescriptionGenerator().run()