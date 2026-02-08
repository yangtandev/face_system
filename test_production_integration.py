
import os
import cv2
import numpy as np
import time
from ppe_classifier import PPEClassifier
try:
    from ultralytics import YOLOv10 as YOLO
except ImportError:
    from ultralytics import YOLO

def main():
    print("=== Production Integration Test ===")
    
    # 1. 初始化 (模擬系統啟動時的動作)
    print("[System] Initializing models...")
    start_time = time.time()
    yolo_model = YOLO("./models/best_cloth.pt")
    ppe_clf = PPEClassifier()
    print(f"[System] Initialization took {time.time() - start_time:.2f}s")
    
    # 2. 準備測試圖片 (使用之前最難搞的 17;33;06)
    target_img_path = "/mnt/c/Users/Yang Tan/Desktop/cloth2/17;33;06_黃邦和_C81_Z3.12_W386.jpg"
    if not os.path.exists(target_img_path):
        print("Test image not found, skipping specific test.")
        return

    print(f"\n[System] Processing image: {os.path.basename(target_img_path)}")
    img = cv2.imread(target_img_path)
    h_img, w_img = img.shape[:2]
    
    # 3. YOLO 偵測 (模擬串流中的 frame 處理)
    results = yolo_model(img, verbose=False, conf=0.15)[0]
    
    vest_box = None
    helmet_box = None
    best_helmet_conf = -1
    best_vest_conf = -1
    
    # 解析 YOLO 結果
    for box in results.boxes:
        cls_id = int(box.cls[0])
        cls_name = yolo_model.names[cls_id]
        conf = float(box.conf[0])
        xyxy = box.xyxy[0].cpu().numpy().astype(int)
        
        if 'vest' in cls_name.lower():
            if conf > best_vest_conf:
                best_vest_conf = conf
                vest_box = xyxy
        
        if 'helmet' in cls_name.lower():
            if conf > best_helmet_conf:
                best_helmet_conf = conf
                helmet_box = xyxy

    # 4. 裁切與 AI 推論
    
    # --- Buckle Check ---
    if helmet_box is not None:
        hx1, hy1, hx2, hy2 = helmet_box
        head_h = hy2 - hy1
        # 這裡使用 V10/V8 相同的裁切邏輯
        chin_y1 = int(hy2 - head_h * 0.25)
        chin_y2 = int(hy2 + head_h * 0.05) 
        chin_y2 = min(chin_y2, h_img) 
        chin_crop = img[chin_y1:chin_y2, hx1:hx2]
        
        # AI Predict
        t0 = time.time()
        res, conf = ppe_clf.predict_buckle(chin_crop)
        t1 = time.time()
        print(f"  > Buckle: {res} ({conf:.4f}) [Inference: {(t1-t0)*1000:.1f}ms]")
    else:
        print("  > Buckle: No Helmet Detected")

    # --- Vest Check ---
    vest_crop = None
    head_w = 0
    if helmet_box is not None:
        head_w = helmet_box[2] - helmet_box[0]

    # 選擇 Vest 來源
    if vest_box is not None:
        vx1, vy1, vx2, vy2 = vest_box
        vest_w = vx2 - vx1
        # 寬度檢查邏輯
        if head_w > 0 and vest_w < head_w * 1.8:
            vest_crop = None # 太窄，強迫轉向 Est
        else:
            vest_crop = img[vy1:vy2, vx1:vx2]
            print("  > Vest Source: YOLO Detection")
            
    if vest_crop is None and helmet_box is not None:
        hx1, hy1, hx2, hy2 = helmet_box
        head_w = hx2 - hx1
        head_h = hy2 - hy1
        head_cx = (hx1 + hx2) // 2
        est_w = head_w * 1.8 
        vx1 = max(0, int(head_cx - est_w / 2))
        vx2 = min(w_img, int(head_cx + est_w / 2))
        vy1 = int(hy2 + head_h * 0.1) 
        vy2 = min(h_img, int(hy2 + head_h * 2.6))
        if vy2 > vy1 and vx2 > vx1:
            vest_crop = img[vy1:vy2, vx1:vx2]
            print("  > Vest Source: Head Estimation")

    if vest_crop is not None:
        # AI Predict
        t0 = time.time()
        res, conf = ppe_clf.predict_vest(vest_crop)
        t1 = time.time()
        print(f"  > Vest:   {res} ({conf:.4f}) [Inference: {(t1-t0)*1000:.1f}ms]")
    else:
        print("  > Vest:   No Vest Crop Available")

if __name__ == "__main__":
    main()
