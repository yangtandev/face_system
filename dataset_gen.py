
import os
import cv2
import numpy as np
import shutil
try:
    from ultralytics import YOLOv10 as YOLO
except ImportError:
    from ultralytics import YOLO

# --- V10 演算法 (用於預標記) ---
VEST_DIFF_THRESHOLD = 0.50       
BUCKLE_SCORE_THRESHOLD = 0.28    
BUCKLE_EXEMPTION_THRESHOLD = 0.28 
HOG_VERT_RATIO_THRESHOLD = 1.15 

def check_valid_vest_color(img_hsv):
    lower = np.array([0, 60, 60])
    upper = np.array([40, 255, 255])
    mask = cv2.inRange(img_hsv, lower, upper)
    ratio = np.sum(mask > 0) / mask.size
    return ratio

def analyze_vest_crop(vest_crop):
    if vest_crop is None or vest_crop.size == 0:
        return 0.0, "ERR"
    h, w = vest_crop.shape[:2]
    center_start = int(w * 0.42)
    center_end = int(w * 0.58)
    img_center = vest_crop[:, center_start:center_end]
    img_sides = np.concatenate((vest_crop[:, 0:int(w*0.15)], vest_crop[:, int(w*0.85):w]), axis=1)
    try:
        vest_hsv = cv2.cvtColor(vest_crop, cv2.COLOR_BGR2HSV)
        if check_valid_vest_color(vest_hsv) < 0.15: 
            return 0.0, "PASS" # Invalid color -> Pass (safe)

        center_hsv = cv2.cvtColor(img_center, cv2.COLOR_BGR2HSV)
        sides_hsv = cv2.cvtColor(img_sides, cv2.COLOR_BGR2HSV)
        
        hist_center = cv2.calcHist([center_hsv], [0, 1], None, [30, 32], [0, 180, 0, 256])
        hist_sides = cv2.calcHist([sides_hsv], [0, 1], None, [30, 32], [0, 180, 0, 256])
        cv2.normalize(hist_center, hist_center, 0, 1, cv2.NORM_MINMAX)
        cv2.normalize(hist_sides, hist_sides, 0, 1, cv2.NORM_MINMAX)
        correlation = cv2.compareHist(hist_center, hist_sides, cv2.HISTCMP_CORREL)
        diff_score = 1.0 - correlation
        
        final_score = diff_score

        sat_center = np.mean(center_hsv[:, :, 1])
        sat_sides = np.mean(sides_hsv[:, :, 1])
        if sat_center / (sat_sides + 1e-5) < 0.85:
            final_score += 0.3 

        gray_center = cv2.cvtColor(img_center, cv2.COLOR_BGR2GRAY)
        gx = cv2.Sobel(gray_center, cv2.CV_32F, 1, 0, ksize=1)
        gy = cv2.Sobel(gray_center, cv2.CV_32F, 0, 1, ksize=1)
        sum_gx = np.sum(np.abs(gx))
        sum_gy = np.sum(np.abs(gy))
        vert_ratio = sum_gx / (sum_gy + 1e-5)
        
        if vert_ratio > HOG_VERT_RATIO_THRESHOLD:
            penalty = min((vert_ratio - HOG_VERT_RATIO_THRESHOLD) * 0.5, 0.4)
            final_score += penalty

        status = "PASS" if final_score < VEST_DIFF_THRESHOLD else "FAIL"
        return final_score, status
    except:
        return 0.0, "ERR"

def analyze_buckle_crop(chin_crop):
    if chin_crop is None or chin_crop.size == 0:
        return 0.0, 0.0, 0.0, "ERR"
    try:
        gray = cv2.cvtColor(chin_crop, cv2.COLOR_BGR2GRAY)
        dark_ratio = np.sum(gray < 80) / gray.size
        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.sum(edges) / gray.size
        norm_edge = min(edge_density / 40.0, 1.0)
        weighted_score = (dark_ratio * 0.6) + (norm_edge * 0.4)
        
        if dark_ratio > BUCKLE_EXEMPTION_THRESHOLD:
            status = "PASS"
        else:
            status = "PASS" if weighted_score > BUCKLE_SCORE_THRESHOLD else "FAIL"
        return dark_ratio, edge_density, weighted_score, status
    except:
        return 0.0, 0.0, 0.0, "ERR"

def main():
    model_path = "./models/best_cloth.pt"
    print(f"Loading YOLO model: {model_path}...")
    model = YOLO(model_path)
    
    # 來源目錄 (這裡只用 cloth2，如果有 cloth3 也可加入)
    source_dirs = ["/mnt/c/Users/Yang Tan/Desktop/cloth2"]
    
    # 資料集輸出目錄
    dataset_base = "/mnt/c/Users/Yang Tan/Desktop/ppe_dataset"
    
    # 清理並重建目錄結構
    if os.path.exists(dataset_base):
        shutil.rmtree(dataset_base)
    
    for category in ["vest", "buckle"]:
        for label in ["pass", "fail"]:
            os.makedirs(os.path.join(dataset_base, category, label), exist_ok=True)

    count = 0
    for source_dir in source_dirs:
        if not os.path.exists(source_dir): continue
        
        image_files = [f for f in os.listdir(source_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        image_files.sort()
        
        print(f"Processing {len(image_files)} images from {source_dir}...")

        for img_file in image_files:
            img_path = os.path.join(source_dir, img_file)
            img = cv2.imread(img_path)
            if img is None: continue
            
            h_img, w_img = img.shape[:2]
            results = model(img, verbose=False, conf=0.15)[0]
            
            vest_box = None
            helmet_box = None
            best_helmet_conf = -1
            best_vest_conf = -1
            
            for box in results.boxes:
                cls_id = int(box.cls[0])
                cls_name = model.names[cls_id]
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

            # --- Buckle Extraction & Pre-labeling ---
            head_w = 0
            if helmet_box is not None:
                hx1, hy1, hx2, hy2 = helmet_box
                head_h = hy2 - hy1
                head_w = hx2 - hx1
                chin_y1 = int(hy2 - head_h * 0.25)
                chin_y2 = int(hy2 + head_h * 0.05) 
                chin_y2 = min(chin_y2, h_img) 
                chin_crop = img[chin_y1:chin_y2, hx1:hx2]
                
                # 使用 V10 預測
                _, _, _, status = analyze_buckle_crop(chin_crop)
                
                # 存入對應資料夾
                label = status.lower() # pass or fail
                if label != "err":
                    fname = f"{img_file}_buckle.jpg"
                    cv2.imwrite(os.path.join(dataset_base, "buckle", label, fname), chin_crop)

            # --- Vest Extraction & Pre-labeling ---
            vest_crop = None
            use_est_anyway = False
            
            if vest_box is not None:
                vx1, vy1, vx2, vy2 = vest_box
                vest_w = vx2 - vx1
                if head_w > 0 and vest_w < head_w * 1.8:
                    use_est_anyway = True
                else:
                    vest_crop = img[vy1:vy2, vx1:vx2]
            
            if (vest_box is None or use_est_anyway) and helmet_box is not None:
                hx1, hy1, hx2, hy2 = helmet_box
                head_cx = (hx1 + hx2) // 2
                est_w = head_w * 1.8 
                vx1 = max(0, int(head_cx - est_w / 2))
                vx2 = min(w_img, int(head_cx + est_w / 2))
                vy1 = int(hy2 + head_h * 0.1) 
                vy2 = min(h_img, int(hy2 + head_h * 2.6))
                if vy2 > vy1 and vx2 > vx1:
                    vest_crop = img[vy1:vy2, vx1:vx2]
            
            if vest_crop is not None:
                # 使用 V10 預測
                _, status = analyze_vest_crop(vest_crop)
                
                # 存入對應資料夾
                label = status.lower()
                if label != "err":
                    fname = f"{img_file}_vest.jpg"
                    cv2.imwrite(os.path.join(dataset_base, "vest", label, fname), vest_crop)
            
            count += 1

    print(f"Dataset generation complete! Check: {dataset_base}")
    print("IMPORTANT: Please review images in 'pass' and 'fail' folders and move any misclassified images.")

if __name__ == "__main__":
    main()
