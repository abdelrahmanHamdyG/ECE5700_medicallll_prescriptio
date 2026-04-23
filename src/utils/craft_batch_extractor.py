import cv2
import os
import glob
import shutil
import numpy as np


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


# Fixes incompatibility with numpy

def patched_adjustResultCoordinates(polys, ratio_w, ratio_h, ratio_net=2):
    if len(polys) > 0:
        for k in range(len(polys)):
            if polys[k] is not None:
                polys[k] = np.array(polys[k]) * np.array([ratio_w * ratio_net, ratio_h * ratio_net])
    return polys

craft_utils.adjustResultCoordinates = patched_adjustResultCoordinates


class CraftBatchExtractor:
    
    def __init__(self, use_cuda=True, temp_dir="temp_craft_junk"):
        self.temp_dir = temp_dir
        print("[INFO] Loading CRAFT text detection model...")
        self.craft = Craft(
            output_dir=self.temp_dir,
            crop_type="box",
            cuda=use_cuda,
            refiner=True,
            text_threshold=0.6,
            link_threshold=0.335,
            low_text=0.335
        )

    def process_directory(self, input_dir: str, output_dir: str, padding: int = 10):
        """Processes all valid images in a directory and exports distinct crops."""
        os.makedirs(output_dir, exist_ok=True)

        
        image_paths = []
        for ext in ("*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG"):
            image_paths.extend(glob.glob(os.path.join(input_dir, ext)))

        if not image_paths:
            print(f"[WARNING] No images found in '{input_dir}'")
            return

        print(f"[INFO] Found {len(image_paths)} images. Proceeding with crop extraction...\n")
        total_crops = 0

        for img_path in image_paths:
            base_name = os.path.splitext(os.path.basename(img_path))[0]
            image = cv2.imread(img_path)
            
            if image is None:
                print(f"[ERROR] Unable to read image: {img_path}")
                continue

            img_h, img_w = image.shape[:2]
            prediction_result = self.craft.detect_text(img_path)
            boxes = prediction_result["boxes"]

            for i, box in enumerate(boxes):
                box_coords = np.array(box).astype(np.int32)
                x, y, w, h = cv2.boundingRect(box_coords)

                
                x_min = max(0, x - padding)
                y_min = max(0, y - padding)
                x_max = min(img_w, x + w + padding)
                y_max = min(img_h, y + h + padding)

                crop = image[y_min:y_max, x_min:x_max]
                crop_filename = f"{base_name}_crop_{i:02d}.jpg"
                cv2.imwrite(os.path.join(output_dir, crop_filename), crop)
                total_crops += 1

        print(f"\n[SUCCESS] Batch processing complete!")
        print(f"[SUCCESS] Exported {total_crops} distinct crops to '{output_dir}/'")

    def cleanup(self):
        
        try:
            self.craft.unload_craftnet_model()
            self.craft.unload_refinenet_model()
        except Exception:
            pass
            
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)


if __name__ == "__main__":
    
    INPUT_FOLDER = "data/raw/eval/new_data_test"
    FLAT_OUTPUT_FOLDER = "data/processed/prescriptions_filtered_crops_new_data_test"
    
    
    BOX_PADDING = 6 

    print(f"--- Starting Bulk Extraction Job ---")
    extractor = CraftBatchExtractor(use_cuda=True)
    extractor.process_directory(
        input_dir=INPUT_FOLDER, 
        output_dir=FLAT_OUTPUT_FOLDER, 
        padding=BOX_PADDING
    )
    extractor.cleanup()
    print("--- Job Finished safely ---")
