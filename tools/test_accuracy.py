import os
import sys
import cv2
import numpy as np
import torch
from PIL import Image
import json
from tqdm import tqdm
import re
from glob import glob
from collections import defaultdict

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models import inception_resnet_v1
from init.mediapipe_handler import MediaPipeHandler
from function import crop_face_without_forehead, get_parts_crop
from init.model import CAM_NAME_MAP

# --- Configuration Constants (Simulating GlobalState) ---
# Hardcoded to match config.json "inCamera" / "outCamera"
# MIN_FACE_THRESHOLD = 380
MIN_FACE_THRESHOLD = 0 # [2026-01-18 Disabled by User Request]
CLOSE_MODE = False
CLOSE_N = 8 if CLOSE_MODE else 6  # 6 means 1/6 margin (66% ROI)

CONFIDENCE_THRESHOLD = 0.7
POTENTIAL_MISS_RATIO = 0.8  # Not used for strict fail, but good for context
Z_SCORE_THRESHOLD = 1.5     # Not fully simulated as we don't calculate Z-Score in this simplified test, but we use Conf.

# Quality Thresholds (From Comparison.check_face_quality)
QUALITY_CENTER_LIMIT = 0.15 # 15% deviation
QUALITY_BLUR_LIMIT = 13.0
QUALITY_MARGIN = 5

def load_config():
    config_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "config.json")
    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)

def cosine_similarity(vec1, vec2):
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

def parse_enrollment_name(filename):
    # Format: JY22_2025_1214_0241_10_754932_夏有生.jpg
    base = os.path.splitext(os.path.basename(filename))[0]
    return base.split("_")[-1]

def parse_test_filename(filename):
    # Format: 00;08;55_Yang_C72_Z1.63_W559.jpg
    base = os.path.splitext(os.path.basename(filename))[0]
    parts = base.split("_")
    # Usually name is at index 1: HH;MM;SS_Name_...
    if len(parts) >= 2:
        return parts[1]
    return "Unknown"

def check_roi(box, img_w):
    """
    Simulate Detector.apply_mask logic (implicit ROI check)
    Detector filters faces based on center point.
    """
    x1, y1, x2, y2 = box
    center_x = (x1 + x2) / 2
    
    roi_x1 = img_w // CLOSE_N
    roi_x2 = (CLOSE_N - 1) * img_w // CLOSE_N
    
    if center_x < roi_x1 or center_x > roi_x2:
        return False, f"Out of ROI (Center:{center_x:.0f} not in {roi_x1}-{roi_x2})"
    return True, "Pass"

def check_face_quality_strict(box, points, img, gaze_status):
    """
    Simulate Comparison.check_face_quality strictly.
    Returns: quality_score (1.0 or 0.0), reason
    """
    h, w, _ = img.shape
    
    # 1. Center Alignment
    face_center_x = (box[0] + box[2]) / 2
    frame_center_x = w / 2
    offset = abs(face_center_x - frame_center_x)
    limit_offset = w * QUALITY_CENTER_LIMIT
    
    if offset > limit_offset:
        return 0.0, f"Not Centered (Offset {offset:.1f} > {limit_offset:.1f})"

    # 2. Integrity (Touching borders)
    for i, p in enumerate(points):
        if p[0] < QUALITY_MARGIN or p[0] > w - QUALITY_MARGIN or \
           p[1] < QUALITY_MARGIN or p[1] > h - QUALITY_MARGIN:
             return 0.0, f"Touching Border (Point {i})"

    # 3. Blur Detection (ROI)
    # [2026-01-18 Disabled by User Request]
    # try:
    #     x1, y1, x2, y2 = map(int, box)
    #     x1, y1 = max(0, x1), max(0, y1)
    #     x2, y2 = min(w, x2), min(h, y2)
    #     
    #     face_roi = img[y1:y2, x1:x2]
    #     if face_roi.size > 0:
    #         gray_roi = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
    #         blur_score = cv2.Laplacian(gray_roi, cv2.CV_64F).var()
    #         
    #         if blur_score < QUALITY_BLUR_LIMIT:
    #             return 0.0, f"Blurry (Score {blur_score:.1f} < {QUALITY_BLUR_LIMIT})"
    # except Exception:
    #     pass

    # 4. Gaze/Pose (From MediaPipeHandler)
    if gaze_status:
        is_looking, gaze_msg, _ = gaze_status
        if not is_looking:
            return 0.0, f"Gaze Fail: {gaze_msg}"
    else:
        return 0.0, "Gaze Status Missing"

    return 1.0, "Pass"

def main():
    # Setup paths
    ENROLL_DIR = "/mnt/c/Users/Yang Tan/Desktop/jenyi-xg"
    TEST_DIR = "/mnt/c/Users/Yang Tan/Desktop/test"
    
    if not os.path.exists(ENROLL_DIR) or not os.path.exists(TEST_DIR):
        print(f"Error: Directories not found.")
        return

    # Load Config & Model
    print("Loading models...")
    config = load_config() # Just to ensure environment is sane
    resnet = inception_resnet_v1.InceptionResnetV1(pretrained='vggface2').eval()
    
    # STRICT: max_num_faces=1 (Production Setting)
    mp_handler = MediaPipeHandler(max_num_faces=1) 
    
    # 1. Build Database
    print("\n=== Building Database from Enrollment Images ===")
    db_embeddings = {}
    db_parts = {} # [2026-01-19] Part features
    
    enroll_files = glob(os.path.join(ENROLL_DIR, "*.jpg"))
    for fpath in tqdm(enroll_files):
        name = parse_enrollment_name(fpath)
        try:
            img_bgr = cv2.imread(fpath)
            if img_bgr is None: continue
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            
            # Enrollment doesn't use strict ROI/Size filtering usually, just gets the face
            boxes, _, points = mp_handler.detect(img_rgb)
            if boxes is None or len(boxes) == 0:
                print(f"Warning: No face detected in {fpath}")
                continue
            
            # Enrollment typically takes largest face
            areas = [(b[2]-b[0]) * (b[3]-b[1]) for b in boxes]
            max_idx = np.argmax(areas)
            
            box = boxes[max_idx]
            lm = points[max_idx]
            
            img_pil = Image.fromarray(img_rgb)
            img_tensor = crop_face_without_forehead(img_pil, box, lm)
            
            with torch.no_grad():
                emb = resnet(img_tensor.unsqueeze(0)).detach().numpy()[0]
            
            if name not in db_embeddings: db_embeddings[name] = []
            db_embeddings[name].append(emb)
            
            # [2026-01-19 Feature] Generate Part Embeddings
            parts_tensors = get_parts_crop(img_pil, lm)
            parts_emb = {}
            for p_name, p_tensor in parts_tensors.items():
                with torch.no_grad():
                    emb = resnet(p_tensor.unsqueeze(0)).detach().numpy()[0]
                    parts_emb[p_name] = emb
            db_parts[name] = parts_emb
            
        except Exception as e:
            print(f"Error processing {fpath}: {e}")

    print(f"Database built. {len(db_embeddings)} people enrolled.")

    # 2. Run Strict Test
    print("\n=== Running Strict Production Test (Part Verification) ===")
    print(f"Config: Min Face={MIN_FACE_THRESHOLD}, ROI=1/{CLOSE_N}, Conf={CONFIDENCE_THRESHOLD}")
    
    test_files = glob(os.path.join(TEST_DIR, "*.jpg"))
    
    stats = defaultdict(int)
    results = []
    
    for fpath in tqdm(test_files):
        filename = os.path.basename(fpath)
        expected_name = parse_test_filename(filename)
        
        try:
            img_bgr = cv2.imread(fpath)
            if img_bgr is None:
                stats["Read Error"] += 1
                continue
            
            # Simulate 1080p frame (Test images might be different size, but logic applies to frame coordinate space)
            # Production uses original frame resolution.
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            h, w, _ = img_rgb.shape
            
            # 1. Detection (Strict: Max 1 Face)
            boxes, _, points = mp_handler.detect(img_rgb)
            
            if boxes is None or len(boxes) == 0:
                stats["No Face Detected"] += 1
                results.append({"file": filename, "status": "Filtered (No Face)", "reason": "MediaPipe Found Nothing"})
                continue
            
            # MediaPipe with max_num_faces=1 returns exactly one face (the most confident one)
            # We MUST process this face. We cannot "choose" another one.
            box = boxes[0]
            lm = points[0]
            
            # 2. ROI Filter
            roi_pass, roi_msg = check_roi(box, w)
            if not roi_pass:
                stats["Filtered (ROI)"] += 1
                results.append({"file": filename, "status": "Filtered (ROI)", "reason": roi_msg})
                continue
                
            # 3. Size Filter
            face_width = box[2] - box[0]
            if face_width < MIN_FACE_THRESHOLD:
                stats["Filtered (Size)"] += 1
                results.append({"file": filename, "status": "Filtered (Size)", "reason": f"Width {face_width} < {MIN_FACE_THRESHOLD}"})
                continue

            # 4. Gaze/Pose Check (Simulated)
            # In production, check_gaze is called on the specific face index
            gaze_result = mp_handler.check_gaze(0)
            if len(gaze_result) == 3:
                g_pass, g_msg, pose = gaze_result
            else:
                # Fallback for debugging or old version
                print(f"DEBUG: check_gaze returned {len(gaze_result)} items: {gaze_result}")
                g_pass, g_msg = gaze_result[0], gaze_result[1]
                pose = (0, 0, 0)
            
            gaze_status = (g_pass, g_msg, pose)

            # 5. Quality Filter
            q_score, q_msg = check_face_quality_strict(box, lm, img_bgr, gaze_status)
            
            if q_score == 0.0:
                stats["Filtered (Quality)"] += 1
                results.append({"file": filename, "status": "Filtered (Quality)", "reason": q_msg})
                continue
                
            # 6. Recognition
            img_pil = Image.fromarray(img_rgb)
            img_tensor = crop_face_without_forehead(img_pil, box, lm)
            
            with torch.no_grad():
                query_emb = resnet(img_tensor.unsqueeze(0)).detach().numpy()[0]
            
            best_score = -1.0
            best_name = "Unknown"
            
            for db_name, embs in db_embeddings.items():
                for db_emb in embs:
                    score = cosine_similarity(query_emb, db_emb)
                    if score > best_score:
                        best_score = score
                        best_name = db_name
            
            # Apply Quality Score (In this strict flow, Q is 1.0 if passed, but production multiplies)
            final_conf = best_score * q_score # q_score is 1.0 here if passed
            
            # Decision Logic
            status = "FAIL"
            reason = ""
            
            # [2026-01-19] Part-Based Verification Logic
            if 0.7 <= final_conf < 0.9:
                if best_name in db_parts:
                    current_parts = get_parts_crop(img_pil, lm)
                    target_parts = db_parts[best_name]
                    
                    veto_reasons = []
                    # Thresholds from experiment: Eye < 0.65 or Nose < 0.6 -> Reject
                    for p_name, threshold in [('eye', 0.65), ('nose', 0.6)]:
                        if p_name in current_parts and p_name in target_parts:
                            with torch.no_grad():
                                p_vec = resnet(current_parts[p_name].unsqueeze(0)).detach().numpy()[0]
                            sim = cosine_similarity(p_vec, target_parts[p_name])
                            if sim < threshold:
                                veto_reasons.append(f"{p_name}({sim:.2f})")
                    
                    if veto_reasons:
                        reason = f"Part Mismatch: {', '.join(veto_reasons)}"
                        final_conf = final_conf * 0.8 # Penalize
                        stats["Part Rejection"] += 1
            
            if final_conf >= CONFIDENCE_THRESHOLD:
                status = "PASS"
            else:
                if "Part Mismatch" in reason:
                    status = "REJECT (Part)"
                else:
                    reason = "Low Confidence"

            # Check correctness
            match_name = (best_name == expected_name)
            
            if status == "PASS":
                if match_name:
                    stats["Success"] += 1
                else:
                    stats["Recognition Fail"] += 1
                    status = "FAIL (MisID)"
            
            results.append({
                "file": filename,
                "expected": expected_name,
                "predicted": best_name,
                "score": final_conf,
                "status": status,
                "reason": reason if match_name else f"Exp: {expected_name}, Got: {best_name} [{reason}]"
            })
            
        except Exception as e:
            print(f"Error testing {fpath}: {e}")
            stats["Error"] += 1

    # 3. Report
    print("\n" + "="*50)
    print("STRICT PRODUCTION SIMULATION REPORT (PART VERIFICATION)")
    print("="*50)
    print(f"Total Images: {len(test_files)}")
    print("-" * 30)
    print(f"Success:           {stats['Success']}")
    print(f"Recognition Fail:  {stats['Recognition Fail']}")
    print(f"Part Rejection:    {stats['Part Rejection']}")
    print("-" * 30)
    print(f"Filtered (ROI):    {stats['Filtered (ROI)']}")
    print(f"Filtered (Size):   {stats['Filtered (Size)']}")
    print(f"Filtered (Quality):{stats['Filtered (Quality)']}")
    print(f"No Face Detected:  {stats['No Face Detected']}")
    print(f"Errors:            {stats['Error']}")
    print("-" * 30)
    
    total_valid = stats['Success'] + stats['Recognition Fail']
    acc = (stats['Success'] / total_valid * 100) if total_valid > 0 else 0
    print(f"Recognition Accuracy (on passed filters): {acc:.2f}%")
    print("="*50)
    
    print("\n--- Failure Details ---")
    for r in results:
        if r["status"] != "PASS":
            print(f"[{r['status']}] {r['file']}")
            print(f"   -> Reason: {r['reason']}")
            if "score" in r:
                print(f"   -> Score:  {r['score']:.2f}")
            print("-" * 20)

if __name__ == "__main__":
    main()
