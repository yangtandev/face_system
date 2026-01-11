import cv2
import mediapipe as mp
import mediapipe.python.solutions as mp_solutions # Explicit import
import numpy as np
from collections import deque

class MediaPipeHandler:
    """
    MediaPipe Face Mesh 封裝器，模仿 MTCNN 的 API 介面。
    提供人臉偵測、關鍵點定位、以及視線 (Gaze) 檢查功能。
    
    [2026-01-11 Update] 
    V19 Production: 寬進嚴出模式 (偵測門檻 0.4 + 統一視線訊息)
    """
    def __init__(self, static_image_mode=True, max_num_faces=1, min_detection_confidence=0.4):
        self.mp_face_mesh = mp_solutions.face_mesh # Use explicit solutions
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=static_image_mode, # 改為 True 以確保每幀獨立偵測，消除追蹤誤差
            max_num_faces=max_num_faces,
            refine_landmarks=True, 
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=0.5
        )
        
        # 關鍵點 Index 定義 (MediaPipe Face Mesh)
        self.IDX_LEFT_EYE_IRIS = 468
        self.IDX_RIGHT_EYE_IRIS = 473
        self.IDX_NOSE_TIP = 1
        self.IDX_MOUTH_LEFT = 61
        self.IDX_MOUTH_RIGHT = 291
        
        # Gaze 偵測需要的眼角點 (用於向量投影)
        self.IDX_LEFT_EYE_INNER = 133
        self.IDX_LEFT_EYE_OUTER = 33
        self.IDX_RIGHT_EYE_INNER = 362
        self.IDX_RIGHT_EYE_OUTER = 263
        
        # Vertical Gaze 關鍵點
        self.IDX_RIGHT_EYE_UPPER = 159
        self.IDX_RIGHT_EYE_LOWER = 145
        self.IDX_LEFT_EYE_UPPER = 386
        self.IDX_LEFT_EYE_LOWER = 374
        
        # Head Pose 關鍵點
        self.IDX_LEFT_CHEEK = 234
        self.IDX_RIGHT_CHEEK = 454

        # Gaze Smoothing History (存最近 5 幀的 ratio tuple)
        self.gaze_history = deque(maxlen=5)

    def detect(self, image, landmarks=True):
        """
        模仿 MTCNN.detect 的回傳格式。
        """
        if not isinstance(image, np.ndarray):
            image = np.array(image)
            
        h, w, _ = image.shape
        self.last_h = h
        self.last_w = w
        
        results = self.face_mesh.process(image)
        self.last_results = results 
        
        if not results.multi_face_landmarks:
            return None, None, None
            
        all_boxes = []
        all_probs = []
        all_points = []

        for face_landmarks in results.multi_face_landmarks:
            coords = np.array([(lm.x * w, lm.y * h) for lm in face_landmarks.landmark])
            x_min, y_min = np.min(coords, axis=0)
            x_max, y_max = np.max(coords, axis=0)
            all_boxes.append([int(x_min), int(y_min), int(x_max), int(y_max)])
            all_probs.append(1.0)
            
            p = face_landmarks.landmark
            points_5 = np.array([
                [p[self.IDX_LEFT_EYE_IRIS].x * w, p[self.IDX_LEFT_EYE_IRIS].y * h],
                [p[self.IDX_RIGHT_EYE_IRIS].x * w, p[self.IDX_RIGHT_EYE_IRIS].y * h],
                [p[self.IDX_NOSE_TIP].x * w, p[self.IDX_NOSE_TIP].y * h],
                [p[self.IDX_MOUTH_LEFT].x * w, p[self.IDX_MOUTH_LEFT].y * h],
                [p[self.IDX_MOUTH_RIGHT].x * w, p[self.IDX_MOUTH_RIGHT].y * h]
            ])
            all_points.append(points_5)

        return np.array(all_boxes), np.array(all_probs), np.array(all_points)

    def _calculate_gaze_ratio(self, landmarks, w, h):
        p = landmarks.landmark
        def get_projection_ratio(idx_start, idx_end, idx_point):
            start = np.array([p[idx_start].x * w, p[idx_start].y * h])
            end   = np.array([p[idx_end].x * w,   p[idx_end].y * h])
            point = np.array([p[idx_point].x * w, p[idx_point].y * h])
            vec_line = end - start
            vec_point = point - start
            denom = np.dot(vec_line, vec_line)
            if denom == 0: return 0.5
            return np.dot(vec_point, vec_line) / denom

        l_h = get_projection_ratio(self.IDX_LEFT_EYE_INNER, self.IDX_LEFT_EYE_OUTER, self.IDX_LEFT_EYE_IRIS)
        r_h = get_projection_ratio(self.IDX_RIGHT_EYE_INNER, self.IDX_RIGHT_EYE_OUTER, self.IDX_RIGHT_EYE_IRIS)
        l_v = get_projection_ratio(self.IDX_LEFT_EYE_UPPER, self.IDX_LEFT_EYE_LOWER, self.IDX_LEFT_EYE_IRIS)
        r_v = get_projection_ratio(self.IDX_RIGHT_EYE_UPPER, self.IDX_RIGHT_EYE_LOWER, self.IDX_RIGHT_EYE_IRIS)
        return (l_h + r_h)/2, l_h, r_h, (l_v + r_v)/2, l_v, r_v

    def _get_head_pose_angles(self, landmarks, w, h):
        p = landmarks.landmark
        model_points = np.array([
            (0.0, 0.0, 0.0), (0.0, 330.0, -65.0), (-225.0, -170.0, -135.0),
            (225.0, -170.0, -135.0), (-150.0, 150.0, -125.0), (150.0, 150.0, -125.0)
        ], dtype=np.float64)
        image_points = np.array([
            (p[1].x * w, p[1].y * h), (p[152].x * w, p[152].y * h), (p[33].x * w, p[33].y * h),
            (p[263].x * w, p[263].y * h), (p[61].x * w, p[61].y * h), (p[291].x * w, p[291].y * h)
        ], dtype=np.float64)
        focal_length = w
        center = (w / 2, h / 2)
        camera_matrix = np.array([[focal_length, 0, center[0]], [0, focal_length, center[1]], [0, 0, 1]], dtype=np.float64)
        dist_coeffs = np.zeros((4, 1))
        success, rotation_vector, translation_vector = cv2.solvePnP(model_points, image_points, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE)
        if not success: return 0, 0, 0
        rmat, _ = cv2.Rodrigues(rotation_vector)
        sy = np.sqrt(rmat[0, 0] * rmat[0, 0] + rmat[1, 0] * rmat[1, 0])
        singular = sy < 1e-6
        if not singular:
            x = np.arctan2(rmat[2, 1], rmat[2, 2])
            y = np.arctan2(-rmat[2, 0], sy)
            z = np.arctan2(rmat[1, 0], rmat[0, 0])
        else:
            x = np.arctan2(-rmat[1, 2], rmat[1, 1]); y = np.arctan2(-rmat[2, 0], sy); z = 0
        
        def normalize_to_zero(angle):
            angle = angle % 360
            if angle > 180: angle -= 360
            if angle > 90: angle -= 180
            elif angle < -90: angle += 180
            return angle
        return normalize_to_zero(np.degrees(x)), normalize_to_zero(np.degrees(y)), normalize_to_zero(np.degrees(z))

    def check_gaze(self, index=0):
        """
        V21 Final Production Logic (The Golden Master): 
        集結「視差補償」、「強制抬頭」、「雙眼同步」三大防線。
        [2026-01-11] 調整順序：優先檢查眼神 (Gaze)，再檢查姿勢 (Pose)。
        """
        if not hasattr(self, 'last_results') or not self.last_results.multi_face_landmarks:
            return True, "No Data"
            
        landmarks = self.last_results.multi_face_landmarks[index]
        w, h = self.last_w, self.last_h
        
        # 1. Pose Angles (計算幾何姿勢，供後續 Conditional Yaw 使用)
        pitch, yaw, roll = self._get_head_pose_angles(landmarks, w, h)
        abs_yaw = abs(yaw)

        # 2. Gaze Calculation (視線計算)
        avg_h, l_h, r_h, avg_v, l_v, r_v = self._calculate_gaze_ratio(landmarks, w, h)
        self.gaze_history.append((avg_h, l_h, r_h, avg_v, l_v, r_v))
        s_avg_h, s_l_h, s_r_h, s_avg_v, s_l_v, s_r_v = np.mean(self.gaze_history, axis=0)
        
        # 3. Horizontal Gaze (水平防線 - 統一訊息: 斜視)
        if not (0.35 < s_avg_h < 0.55):
            return False, f"斜視 (H:{s_avg_h:.2f})"
        
        h_diff = abs(s_l_h - s_r_h)
        if not (0.30 < s_l_h < 0.70) or not (0.30 < s_r_h < 0.70) or h_diff > 0.08:
            return False, f"斜視 (L{s_l_h:.2f}/R{s_r_h:.2f}/D{h_diff:.2f})"

        # 4. Vertical Gaze (垂直防線)
        if s_avg_v < 0.45:
            return False, f"斜視 (Up:{s_avg_v:.2f})"
        elif s_avg_v > 2.50:
            return False, f"請抬頭 (Down:{s_avg_v:.2f})"

        # 5. Pose Check (統一訊息: 未正視鏡頭)
        if pitch > 25 or pitch < -25 or abs_yaw > 35 or abs(roll) > 20:
            return False, f"未正視鏡頭 (P{int(pitch)}/Y{int(yaw)}/R{int(roll)})"

        # 6. Conditional Yaw (眼神差時臉要更正)
        if s_avg_v > 0.85 and abs_yaw > 25:
            return False, f"未正視鏡頭 (低頭側臉 Y:{int(yaw)})"
            
        return True, f"Pass (P{int(pitch)}/Y{int(yaw)}/V{s_avg_v:.2f})"