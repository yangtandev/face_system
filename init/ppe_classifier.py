
import os
import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image

# --- 模型定義 (必須與訓練時完全一致) ---
class MiniCNN(nn.Module):
    def __init__(self):
        super(MiniCNN, self).__init__()
        # Input: 3 x 64 x 64
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2), # -> 16 x 32 x 32
            
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2), # -> 32 x 16 x 16
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2), # -> 64 x 8 x 8
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 8 * 8, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 2) # 0: Fail, 1: Pass
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

class PPEClassifier:
    def __init__(self, model_dir="./models/ppe_classifiers", device=None):
        """
        初始化 PPE 分類器
        :param model_dir: 存放 .pth 權重檔的目錄
        :param device: 指定運算裝置 (cpu/cuda)，若為 None 則自動偵測
        """
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device
            
        self.buckle_model = self._load_model(os.path.join(model_dir, "buckle_net.pth"))
        self.vest_model = self._load_model(os.path.join(model_dir, "vest_net.pth"))
        
        # 預處理變換 (必須與訓練時一致)
        self.preprocess = transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
        print(f"[PPEClassifier] Models loaded on {self.device}")

    def _load_model(self, path):
        if not os.path.exists(path):
            print(f"[PPEClassifier] Warning: Model not found at {path}")
            return None
        
        model = MiniCNN().to(self.device)
        try:
            model.load_state_dict(torch.load(path, map_location=self.device))
            model.eval()
            return model
        except Exception as e:
            print(f"[PPEClassifier] Error loading model {path}: {e}")
            return None

    def predict_buckle(self, img_bgr):
        """
        輸入: OpenCV 格式圖片 (BGR)
        輸出: (label, score) -> "PASS"/"FAIL", 0.0~1.0
        """
        return self._predict(self.buckle_model, img_bgr)

    def predict_vest(self, img_bgr):
        """
        輸入: OpenCV 格式圖片 (BGR)
        輸出: (label, score) -> "PASS"/"FAIL", 0.0~1.0
        """
        return self._predict(self.vest_model, img_bgr)

    def _predict(self, model, img_bgr):
        """
        推論單張圖片
        """
        if model is None:
            return "ERR", 0.0
        
        if img_bgr is None or img_bgr.size == 0:
            return "ERR", 0.0

        try:
            # OpenCV BGR -> PIL RGB
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            img_pil = Image.fromarray(img_rgb)
            
            input_tensor = self.preprocess(img_pil).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                outputs = model(input_tensor)
                probs = torch.nn.functional.softmax(outputs, dim=1)
                conf, predicted = torch.max(probs, 1)
                
                # 0: Fail, 1: Pass
                label = "PASS" if predicted.item() == 1 else "FAIL"
                score = conf.item()
                
            return label, score
            
        except Exception as e:
            print(f"[PPEClassifier] Inference error: {e}")
            return "ERR", 0.0

# --- 自我測試區塊 ---
if __name__ == "__main__":
    # 簡單測試
    classifier = PPEClassifier()
    
    # 建立一個假的測試圖 (全黑 100x100 BGR)
    dummy_img = np.zeros((100, 100, 3), dtype=np.uint8)
    
    label, score = classifier.predict_buckle(dummy_img)
    print(f"Buckle Dummy Test: {label} ({score:.4f})")
    
    label, score = classifier.predict_vest(dummy_img)
    print(f"Vest Dummy Test: {label} ({score:.4f})")
