import torch
import clip
from PIL import Image
import os

class WardrobeClassifier:
    def __init__(self):
        self.device = "cpu"
        print("正在載入 CLIP 模型 (ViT-B/32)...")
        # 載入模型一次就好，放在 __init__ 裡
        self.model, self.preprocess = clip.load("ViT-B/32", device=self.device)
        print("模型載入完成！")

    def predict(self, img_path, candidate_labels):
        """
        回傳: (最高分標籤, 信心分數)
        """
        if not os.path.exists(img_path):
            return None, 0.0

        # 1. 圖片前處理
        image = self.preprocess(Image.open(img_path)).unsqueeze(0).to(self.device)
        
        # 2. 文字前處理 (自動加上 prompts)
        # 技巧：加上 "a photo of a..." 可以增加準確度
        text_inputs = clip.tokenize([f"a photo of a {l}" for l in candidate_labels]).to(self.device)

        # 3. 推論
        with torch.no_grad():
            image_features = self.model.encode_image(image)
            text_features = self.model.encode_text(text_inputs)
            
            # 計算相似度並轉為機率
            logits_per_image, _ = self.model(image, text_inputs)
            probs = logits_per_image.softmax(dim=-1).cpu().numpy()[0]

        # 4. 找出最高分的 index
        best_idx = probs.argmax()
        best_label = candidate_labels[best_idx]
        best_score = probs[best_idx]

        return best_label, best_score

# === 模擬 Line Bot 呼叫 ===
if __name__ == "__main__":
    # 模擬 Server 啟動時初始化
    classifier = WardrobeClassifier()
    
    # 假設這是使用者上傳的圖片 (請確認 data 資料夾有圖片)
    test_img = "data/test.jpg" 
    
    # 定義分類 (您可以隨時擴充這裡)
    tags = ["long-sleeved shirt", "short-sleeved t-shirt", "jeans", "dress", "jacket"]
    
    label, score = classifier.predict(test_img, tags)
    
    if label:
        print(f"辨識結果: {label}")
        print(f"信心水準: {score:.2%}")
        
        # 簡單的邏輯判斷
        if score > 0.6:
            print(">> 系統動作: 信心足夠，自動寫入資料庫...")
        else:
            print(">> 系統動作: 信心不足，請使用者手動確認...")