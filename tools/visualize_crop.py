import os
import sys
import cv2
import numpy as np
import torch
from PIL import Image
from glob import glob

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from init.mediapipe_handler import MediaPipeHandler
from function import crop_face_without_forehead

# Difficult samples to visualize
DIFFICULT_SAMPLES = [
    "08;48;56_蔡準庭_C84_Z1.84.jpg",
    "10;55;48_王碩賢_C76_Z1.63.jpg",
    "14;14;47_黃唯明_C73_Z1.90_W463.jpg",
    "15;03;23_廖家琳_C77_Z1.61_W689.jpg",
    "15;21;35_夏有生_C73_Z1.79_W453.jpg",
    "15;32;02_廖家琳_C77_Z1.76_W441.jpg",
    "18;10;22_李中仁_C74_Z1.90_W425.jpg",
    "18;10;45_黎文秀_C75_Z1.79_W632.jpg",
    "18;10;34_農文豐_C70_Z1.54_W507.jpg"
]

def main():
    TEST_DIR = "/mnt/c/Users/Yang Tan/Desktop/test"
    OUTPUT_DIR = "tools/crop_visuals"
    
    mp_handler = MediaPipeHandler(max_num_faces=1)
    
    print(f"Visualizing crops to {OUTPUT_DIR}...")
    
    # Try to find the files
    for fname in DIFFICULT_SAMPLES:
        fpath = os.path.join(TEST_DIR, fname)
        if not os.path.exists(fpath):
            # Try to match by suffix if exact name fails (some might have changed)
            candidates = glob(os.path.join(TEST_DIR, f"*{fname.split('_')[1]}*"))
            if candidates:
                fpath = candidates[0]
            else:
                print(f"Skipping {fname} (Not found)")
                continue
            
        print(f"Processing {os.path.basename(fpath)}...")
        img_bgr = cv2.imread(fpath)
        if img_bgr is None: continue
        
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        
        # Detect
        boxes, _, points = mp_handler.detect(img_rgb)
        if boxes is None: 
            print("  No face detected")
            continue
        
        # Crop using CURRENT function.py logic (which reads updated config.json)
        img_pil = Image.fromarray(img_rgb)
        
        # crop_face_without_forehead returns a tensor (Standardized -1 to 1)
        face_tensor = crop_face_without_forehead(img_pil, boxes[0], points[0])
        
        # Convert back to uint8 image
        # Tensor is (C, H, W), range [-1, 1]
        face_np = face_tensor.permute(1, 2, 0).numpy() # (H, W, C)
        
        # Denormalize: (x * 128.0) + 127.5
        face_np = (face_np * 128.0 + 127.5).clip(0, 255).astype(np.uint8)
        
        # Save
        out_name = os.path.basename(fpath).replace(";", "-") # Sanitize filename
        out_path = os.path.join(OUTPUT_DIR, f"crop_{out_name}")
        
        Image.fromarray(face_np).save(out_path)
        print(f"  Saved: {out_path}")

if __name__ == "__main__":
    main()
