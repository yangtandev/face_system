#!/usr/bin/env python3
"""
Offline Replay Test — 離線重測腳本
==================================
用與現場完全相同的管線重新辨識 miss 目錄中的照片：
  Track A: 從圖片重新跑整條管線 (MediaPipe → Crop → ResNet → FAISS)
  Track B: 用 JSON 中的 embedding 直接搜 FAISS
同時套用完整的品質過濾 (Gaze, Pose, V-Ratio, EAR, Gap Check)。
"""

import sys, os, json, glob, shutil, time
import numpy as np
import cv2
import torch
from PIL import Image
from collections import defaultdict
from pathlib import Path

# ── 路徑設定 ──────────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

INPUT_DIR  = "/mnt/c/Users/Yang Tan/Desktop/miss"
OUTPUT_DIR = "/mnt/c/Users/Yang Tan/Desktop/miss_retest3"
MISMATCH_IMG_DIR = os.path.join(OUTPUT_DIR, "mismatch_images")
if os.path.exists(MISMATCH_IMG_DIR):
    shutil.rmtree(MISMATCH_IMG_DIR)
os.makedirs(MISMATCH_IMG_DIR, exist_ok=True)

# ── 模型載入 ──────────────────────────────────────────────
print("[1/5] 載入模型...")
from models import inception_resnet_v1
from init.function import crop_face_without_forehead
from init.mediapipe_handler import MediaPipeHandler
from init.ann_index import AnnIndex

resnet = inception_resnet_v1.InceptionResnetV1(pretrained='vggface2').eval()

mp_handler = MediaPipeHandler(static_image_mode=True, max_num_faces=1, min_detection_confidence=0.4)

ann_index = AnnIndex(
    index_path=os.path.join(PROJECT_ROOT, "media", "faiss.index"),
    mapping_path=os.path.join(PROJECT_ROOT, "media", "faiss_map.json"),
)
ann_index.load()
print(f"   FAISS index loaded: {ann_index.index.ntotal} vectors, {len(set(ann_index.id_map))} people")

# 載入 id -> name 映射
features_dict = {}
desc_dir = os.path.join(PROJECT_ROOT, "media", "descriptors")
for f in os.listdir(desc_dir):
    if f.endswith('.npy'):
        bn = os.path.splitext(f)[0]
        parts = bn.split('_')
        cat = parts[0]
        name = parts[-1]
        features_dict.setdefault("id_name", {})[cat] = name
print(f"   Enrolled IDs: {len(features_dict.get('id_name', {}))} people")

# 載入 config
with open(os.path.join(PROJECT_ROOT, "config.json"), "r", encoding="utf-8") as f:
    CONFIG = json.load(f)

# ── 常數 ─────────────────────────────────────────────────
CONFIDENCE_THRESHOLD = 0.70
Z_SCORE_THRESHOLD    = 1.50
VISITOR_CONF_THRESHOLD = 0.50
id_name_map = features_dict.get("id_name", {})

# ── 品質檢查 ─────────────────────────────────────────────
def check_face_quality(box, points, frame_w, frame_h, gaze_status):
    """複製 model.py Comparison.check_face_quality 的完整邏輯"""
    face_center_x = (box[0] + box[2]) / 2
    frame_center_x = frame_w / 2
    offset = abs(face_center_x - frame_center_x)
    limit_offset = frame_w * 0.15
    margin = 5

    current_ear = 1.0

    metrics = {
        'center_offset_px': float(offset),
        'center_limit_px': float(limit_offset),
        'face_width_px': float(box[2] - box[0]),
        'visibility_margin': float(margin),
        'gaze_passed': False,
        'gaze_msg': 'Init',
        'ear': 1.0,
        'v_ratio': 0.0,
        'roll_angle': 0.0,
        'pitch_check': 'Pass',
        'yaw_check': 'Pass'
    }

    if offset > limit_offset:
        return 0.0, f"未置中 (偏離 {offset:.1f}px > 容許 {limit_offset:.1f}px)", metrics

    # 特徵點完整性
    for i, p in enumerate(points):
        if p[0] < margin or p[0] > frame_w - margin or \
           p[1] < margin or p[1] > frame_h - margin:
            return 0.0, f"特徵點被切除/遮擋 (點{i}座標 {p} 超出邊界)", metrics

    face_w = max(10, box[2] - box[0])

    # Gaze & Pose
    if face_w > 100:
        if gaze_status:
            is_looking = gaze_status[0]
            gaze_msg = gaze_status[1]
            current_ear = 1.0
            pose_tuple = (0, 0, 0)
            if len(gaze_status) >= 4:
                current_ear = gaze_status[3]
                pose_tuple = gaze_status[2]

            metrics['gaze_passed'] = is_looking
            metrics['gaze_msg'] = gaze_msg
            metrics['ear'] = float(current_ear)

            pitch, yaw, roll = pose_tuple
            metrics['pitch'] = float(pitch)
            metrics['yaw'] = float(yaw)
            metrics['roll_angle'] = float(roll)

            # [2026-04-15] 修正邏輯：只要任一角度過大就是不良姿態 (or)
            is_bad_pose = (abs(yaw) > 25 or abs(pitch) > 20) or abs(roll) > 20
            metrics['is_bad_pose'] = is_bad_pose

            if not is_looking:
                return 0.0, f"{gaze_msg}", metrics
        else:
            return 0.0, "Gaze Status Missing", metrics

    # V-Ratio
    eye_y = (points[0][1] + points[1][1]) / 2
    nose_y = points[2][1]
    mouth_y = (points[3][1] + points[4][1]) / 2
    eye_nose_dist = nose_y - eye_y
    nose_mouth_dist = mouth_y - nose_y

    if eye_nose_dist > 0:
        v_ratio = nose_mouth_dist / eye_nose_dist
        metrics['v_ratio'] = float(v_ratio)
        if v_ratio < 0.35:
            metrics['pitch_check'] = 'Fail (Extreme Low)'
            return 0.0, f"低頭 (V-Ratio: {v_ratio:.2f} < 0.35)", metrics
        if v_ratio < 0.42:
            if current_ear < 0.22:
                metrics['pitch_check'] = 'Fail (Combo Low+Cover)'
                return 0.0, f"低頭/遮眼 (V {v_ratio:.2f}<0.42 & EAR {current_ear:.2f}<0.22)", metrics

    # EAR
    if current_ear < 0.075:
        return 0.0, f"眼睛閉合 (EAR: {current_ear:.4f} < 0.075)", metrics

    # [2026-04-14] Continuous penalty logic
    quality_score = 1.0
    if 'yaw' in metrics and 'pitch' in metrics:
        yaw = metrics['yaw']
        pitch = metrics['pitch']
        roll = metrics.get('roll_angle', 0.0)
        
        yaw_penalty = max(0, abs(yaw) - 20) * 0.005
        pitch_penalty = max(0, abs(pitch) - 15) * 0.005
        roll_penalty = max(0, abs(roll) - 5) * 0.005
        
        total_penalty = min(0.20, yaw_penalty + pitch_penalty + roll_penalty)
        quality_score -= total_penalty

    return quality_score, "Pass", metrics


def run_faiss_search(embedding_vec, quality_score, quality_metrics):
    """用 embedding 向量搜 FAISS，回傳辨識結果"""
    if ann_index.index is None or ann_index.index.ntotal == 0:
        return {"predicted_id": "None", "predicted_name": "None", "confidence": 0.0,
                "z_score": 0.0, "top5": [], "decision": "NO_INDEX", "gap": 0.0}

    distances, person_ids = ann_index.search(embedding_vec, k=ann_index.index.ntotal)
    if person_ids is None or len(person_ids) == 0:
        return {"predicted_id": "None", "predicted_name": "None", "confidence": 0.0,
                "z_score": 0.0, "top5": [], "decision": "NO_RESULT", "gap": 0.0}

    top_k_sims = np.array(distances)
    mean_score = np.mean(top_k_sims) if len(top_k_sims) > 1 else 0
    std_dev = np.std(top_k_sims) if len(top_k_sims) > 1 else 0

    # Candidates
    candidates = []
    for i, pid in enumerate(person_ids):
        s_raw = distances[i]
        # [2026-04-15] 高分豁免機制
        current_qs = quality_score
        if s_raw >= 0.750 and current_qs < 0.99:
            current_qs = max(current_qs, 0.99)
        
        s_final = s_raw * current_qs
        z = (s_raw - mean_score) / std_dev if std_dev > 0 else 0

        # [2026-04-15] 四象限 Z-Score 動態門檻矩陣
        if quality_metrics.get('is_bad_pose', False):
            required_conf = 0.65 if z >= 2.5 else 0.85
        else:
            required_conf = 0.65 if z >= 2.5 else 0.70

        if s_final >= required_conf and z >= Z_SCORE_THRESHOLD:
            candidates.append({'id': pid, 'raw': s_raw, 'conf': s_final, 'z': z, 'required_conf': required_conf})

    best_id = person_ids[0]
    raw_conf = distances[0]
    
    final_qs = quality_score
    if raw_conf >= 0.750 and final_qs < 0.99:
        final_qs = max(final_qs, 0.99)
    confidence = raw_conf * final_qs
    z_score = (raw_conf - mean_score) / std_dev if std_dev > 0 else 0

    # Gap Check
    gap = 0.0
    if len(distances) > 1:
        gap = float(distances[0]) - float(distances[1])
    gap_threshold = 0.005 if confidence >= 0.75 or z_score >= 2.5 else 0.015

    top5 = []
    for i in range(min(5, len(person_ids))):
        pid = person_ids[i]
        s_raw = distances[i]
        z = (s_raw - mean_score) / std_dev if std_dev > 0 else 0
        top5.append({
            "rank": i + 1,
            "id": pid,
            "name": id_name_map.get(pid, "?"),
            "score": float(s_raw),
            "z_score": float(z)
        })

    # Decision
    decision = "REJECT_LOW_CONF"
    final_required = CONFIDENCE_THRESHOLD
    if quality_metrics.get('is_bad_pose', False):
        final_required = 0.65 if z_score >= 2.5 else 0.85
    else:
        final_required = 0.65 if z_score >= 2.5 else 0.70

    if gap < gap_threshold:
        decision = f"REJECT_GAP ({gap:.4f} < {gap_threshold})"
    elif candidates:
        winner = candidates[0]
        best_id = winner['id']
        raw_conf = winner['raw']
        confidence = winner['conf']
        z_score = winner['z']
        if confidence >= final_required and z_score >= Z_SCORE_THRESHOLD:
            decision = "ACCEPT"
    else:
        if confidence >= 0.58:
            decision = "VISITOR"

    return {
        "predicted_id": best_id,
        "predicted_name": id_name_map.get(best_id, "?"),
        "confidence": float(confidence),
        "raw_confidence": float(raw_conf),
        "z_score": float(z_score),
        "gap": float(gap),
        "gap_threshold": float(gap_threshold),
        "top5": top5,
        "decision": decision,
        "required_conf": float(final_required),
    }


# ── 主迴圈 ───────────────────────────────────────────────
print(f"\n[2/5] 掃描輸入目錄: {INPUT_DIR}")
jpg_files = sorted(glob.glob(os.path.join(INPUT_DIR, "*.jpg")))
print(f"   找到 {len(jpg_files)} 張照片")

os.makedirs(OUTPUT_DIR, exist_ok=True)

results = []
errors = []
total = len(jpg_files)

print(f"\n[3/5] 開始離線重測 ({total} 張)...\n")
for idx, jpg_path in enumerate(jpg_files):
    basename = os.path.splitext(os.path.basename(jpg_path))[0]
    json_path = os.path.join(INPUT_DIR, basename + ".json")

    # 解析現場結果 (from filename)
    parts = basename.split("_")
    # 格式: HH;MM;SS_In/Out_Name_C{conf}_Z{z}_W{width}
    field_direction = parts[1] if len(parts) > 1 else "?"
    field_name = parts[2] if len(parts) > 2 else "?"

    # 讀取現場 JSON
    field_json = None
    field_embedding = None
    field_predicted_id = None
    if os.path.exists(json_path):
        with open(json_path, 'r', encoding='utf-8') as f:
            field_json = json.load(f)
        field_predicted_id = field_json.get("predicted_id", "?")
        emb = field_json.get("embedding")
        if emb is not None:
            field_embedding = np.array(emb, dtype=np.float32)

    # 讀取圖片
    img_bgr = cv2.imread(jpg_path)
    if img_bgr is None:
        errors.append({"file": basename, "error": "Cannot read image"})
        continue

    frame_h, frame_w = img_bgr.shape[:2]
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    # ─── Track A: Full Pipeline ──────────────────────────
    track_a = {"track": "A_pipeline", "predicted_id": "None", "decision": "NO_FACE"}

    boxes, _, points_arr = mp_handler.detect(img_rgb)

    if boxes is not None and len(boxes) > 0:
        box = list(map(int, boxes[0]))
        points = points_arr[0].copy()
        face_width = box[2] - box[0]

        # Gaze Check (from last detect result)
        g_pass, g_msg, g_pose, g_ear = mp_handler.check_gaze(0)
        gaze_status = (g_pass, g_msg, g_pose, g_ear)

        # Quality Check
        quality_score, quality_msg, quality_metrics = check_face_quality(
            box, points, frame_w, frame_h, gaze_status)

        track_a["face_detected"] = True
        track_a["face_width"] = face_width
        track_a["quality_score"] = float(quality_score)
        track_a["quality_msg"] = quality_msg
        track_a["quality_metrics"] = quality_metrics

        if quality_score == 0.0:
            track_a["decision"] = f"QUALITY_FAIL: {quality_msg}"
            track_a["predicted_id"] = "None"
            track_a["predicted_name"] = "None"
        else:
            # Extract embedding
            try:
                frame_image = Image.fromarray(img_rgb)
                img_cropped = crop_face_without_forehead(frame_image, list(box), points.copy())
                face_emb = resnet(img_cropped.unsqueeze(0))
                emb_vec = face_emb[0].detach().numpy()

                search_result = run_faiss_search(emb_vec, quality_score, quality_metrics)
                track_a.update(search_result)
                track_a["embedding_norm"] = float(np.linalg.norm(emb_vec))
            except Exception as e:
                track_a["decision"] = f"EMBEDDING_FAIL: {e}"
                track_a["predicted_id"] = "None"
    else:
        track_a["face_detected"] = False

    # ─── Track B: JSON Embedding ─────────────────────────
    track_b = {"track": "B_json_embedding", "predicted_id": "None", "decision": "NO_EMBEDDING"}

    if field_embedding is not None:
        # 用 field JSON 裡的品質資料 (如果有的話)
        q_score_b = float(field_json.get("quality_score", 1.0)) if field_json else 1.0
        q_metrics_b = field_json.get("quality_metrics", {}) if field_json else {}

        search_result_b = run_faiss_search(field_embedding, q_score_b, q_metrics_b)
        track_b.update(search_result_b)

    # ─── 彙整 ────────────────────────────────────────────
    result_entry = {
        "file": basename,
        "field_direction": field_direction,
        "field_name": field_name,
        "field_predicted_id": field_predicted_id,
        "track_a": track_a,
        "track_b": track_b,
    }

    # 判斷 Track A 最終結果 (用於檔名)
    if track_a["decision"] == "ACCEPT":
        retest_name = id_name_map.get(track_a["predicted_id"], track_a["predicted_id"])
        retest_id = track_a["predicted_id"]
    elif track_a["decision"] == "VISITOR":
        retest_name = "VISITOR"
        retest_id = "VISITOR"
    else:
        retest_name = "REJECTED"
        retest_id = "None"

    result_entry["retest_name"] = retest_name
    result_entry["retest_id"] = retest_id

    # 是否匹配
    match_a = (track_a.get("predicted_id") == field_predicted_id) and track_a.get("decision") == "ACCEPT"
    match_b = (track_b.get("predicted_id") == field_predicted_id) and track_b.get("decision") == "ACCEPT"
    result_entry["match_field_a"] = match_a
    result_entry["match_field_b"] = match_b

    results.append(result_entry)

    # 進度
    status_a = f'{track_a["decision"][:20]:20s}'
    status_match = "✓" if match_a else "✗"
    if (idx + 1) % 10 == 0 or idx == 0 or idx == total - 1:
        print(f"  [{idx+1:3d}/{total}] {basename[:40]:40s} Field={field_name:6s} → A={retest_name:6s} {status_match} | {status_a}")

# ── 輸出結果 ──────────────────────────────────────────────
print(f"\n[4/5] 寫入結果到 {OUTPUT_DIR}...")

# 完整結果 JSON
full_result_path = os.path.join(OUTPUT_DIR, "retest_results.json")
def default_converter(o):
    if isinstance(o, np.integer): return int(o)
    if isinstance(o, np.floating): return float(o)
    if isinstance(o, np.ndarray): return o.tolist()
    if isinstance(o, np.bool_): return bool(o)
    return str(o)

with open(full_result_path, 'w', encoding='utf-8') as f:
    json.dump(results, f, ensure_ascii=False, indent=2, default=default_converter)

# 每張照片的個別 JSON (方便人工比對)
for r in results:
    per_file_path = os.path.join(OUTPUT_DIR, r["file"] + "_retest.json")
    with open(per_file_path, 'w', encoding='utf-8') as f:
        json.dump(r, f, ensure_ascii=False, indent=2, default=default_converter)

# ── 差異分析 ──────────────────────────────────────────────
print(f"\n[5/5] 差異分析...")

mismatches = []
quality_rejected = []
accepted_match = 0
accepted_mismatch = 0
total_quality_fail = 0

for r in results:
    ta = r["track_a"]
    if ta.get("decision", "").startswith("QUALITY_FAIL") or ta.get("decision") == "NO_FACE":
        total_quality_fail += 1
        quality_rejected.append({
            "file": r["file"],
            "field_name": r["field_name"],
            "field_id": r["field_predicted_id"],
            "reason": ta.get("decision", "?"),
            "quality_msg": ta.get("quality_msg", "?"),
        })
    elif ta.get("decision") == "ACCEPT":
        # Load ground truth to check if this is actually correct
        with open("/home/ubuntu/facial_recognition/tests/ground_truth.json", "r", encoding='utf-8') as gf:
            gt_data = json.load(gf)
            gt_dict = {item['file'].replace(".jpg", ""): item['ground_truth'] for item in gt_data}
            
        gt_val = gt_dict.get(r['file'])
        
        if r["retest_id"] == gt_val or (r["retest_id"] == "SH99" and gt_val == "VISITOR"):
            accepted_match += 1
        else:
            accepted_mismatch += 1
            m = {
                "file": r["file"],
                "field_name": r["field_name"],
                "field_id": r["field_predicted_id"],
                "retest_name": r["retest_name"],
                "retest_id": r["retest_id"],
                "confidence": ta.get("confidence", 0),
                "z_score": ta.get("z_score", 0),
                "reason": f"ID_MISMATCH (GT: {gt_val})",
            }
            mismatches.append(m)
            shutil.copy2(os.path.join(INPUT_DIR, r["file"] + ".jpg"), os.path.join(MISMATCH_IMG_DIR, f"GT_{gt_val}_TEST_{r['retest_id']}_{r['file']}.jpg"))
    elif ta.get("decision") == "VISITOR":
        # Load ground truth to check if this is actually correct
        with open("/home/ubuntu/facial_recognition/tests/ground_truth.json", "r", encoding='utf-8') as gf:
            gt_data = json.load(gf)
            gt_dict = {item['file'].replace(".jpg", ""): item['ground_truth'] for item in gt_data}
            
        gt_val = gt_dict.get(r['file'])
        
        if r["field_predicted_id"] == "VISITOR" or gt_val == "VISITOR":
             accepted_match += 1
        else:
             accepted_mismatch += 1
             m= {
                 "file": r["file"],
                 "field_name": r["field_name"],
                 "field_id": r["field_predicted_id"],
                 "retest_name": "VISITOR",
                 "retest_id": "VISITOR",
                 "confidence": ta.get("confidence", 0),
                 "z_score": ta.get("z_score", 0),
                 "reason": f"VISITOR_MISMATCH (GT: {gt_val})",
             }
             mismatches.append(m)
             shutil.copy2(os.path.join(INPUT_DIR, r["file"] + ".jpg"), os.path.join(MISMATCH_IMG_DIR, f"GT_{gt_val}_TEST_VISITOR_{r['file']}.jpg"))
    else:
        # Load ground truth to check if this is actually correct
        with open("/home/ubuntu/facial_recognition/tests/ground_truth.json", "r", encoding='utf-8') as gf:
            gt_data = json.load(gf)
            gt_dict = {item['file'].replace(".jpg", ""): item['ground_truth'] for item in gt_data}
            
        gt_val = gt_dict.get(r['file'])
        
        if gt_val == "FILTERED":
            accepted_match += 1 # correctly filtered
        else:
            m = {
                "file": r["file"],
                "field_name": r["field_name"],
                "field_id": r["field_predicted_id"],
                "retest_name": "REJECTED",
                "retest_id": "None",
                "confidence": ta.get("confidence", 0),
                "z_score": ta.get("z_score", 0),
                "reason": f"FIELD_ACCEPTED_BUT_RETEST_REJECTED ({ta.get('decision', '?')}) (GT: {gt_val})",
            }
            mismatches.append(m)
            shutil.copy2(os.path.join(INPUT_DIR, r["file"] + ".jpg"), os.path.join(MISMATCH_IMG_DIR, f"GT_{gt_val}_TEST_FILTERED_{r['file']}.jpg"))

# 輸出差異清單
mismatch_path = os.path.join(OUTPUT_DIR, "mismatches.json")
with open(mismatch_path, 'w', encoding='utf-8') as f:
    json.dump(mismatches, f, ensure_ascii=False, indent=2, default=default_converter)

quality_reject_path = os.path.join(OUTPUT_DIR, "quality_rejected.json")
with open(quality_reject_path, 'w', encoding='utf-8') as f:
    json.dump(quality_rejected, f, ensure_ascii=False, indent=2, default=default_converter)

# ── 摘要 ─────────────────────────────────────────────────
print("\n" + "=" * 70)
print("📊 離線重測摘要")
print("=" * 70)
print(f"  總照片數:           {total}")
print(f"  品質過濾攔截:       {total_quality_fail} ({total_quality_fail/total*100:.1f}%)")
print(f"  通過辨識 & 匹配:    {accepted_match}")
print(f"  通過辨識 & 不匹配:  {accepted_mismatch}")
print(f"  差異總數:           {len(mismatches)}")
print(f"")
print(f"  結果已存至:         {OUTPUT_DIR}")
print(f"  完整結果:           retest_results.json")
print(f"  差異清單:           mismatches.json ({len(mismatches)} 筆)")
print(f"  品質攔截清單:       quality_rejected.json ({total_quality_fail} 筆)")
print("=" * 70)

if mismatches:
    print("\n📋 差異清單:")
    print(f"{'#':>3s}  {'檔案':40s}  {'現場':8s}  {'重測':8s}  {'原因':30s}")
    print("-" * 95)
    for i, m in enumerate(mismatches):
        print(f"{i+1:3d}  {m['file'][:40]:40s}  {m['field_name']:8s}  {m['retest_name']:8s}  {m['reason'][:30]:30s}")

if quality_rejected:
    print(f"\n🔍 品質過濾攔截清單 (前20筆):")
    print(f"{'#':>3s}  {'檔案':40s}  {'現場':8s}  {'原因':40s}")
    print("-" * 95)
    for i, q in enumerate(quality_rejected[:20]):
        print(f"{i+1:3d}  {q['file'][:40]:40s}  {q['field_name']:8s}  {q['quality_msg'][:40]:40s}")

mp_handler.close()
print("\n✅ 完成")
