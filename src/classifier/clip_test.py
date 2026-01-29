import torch
import clip
from PIL import Image
import os

def test_clip():
    # 1. 設定裝置為 CPU
    device = "cpu"
    
    # 2. 載入 CLIP 模型 (ViT-B/32 是平衡速度與精準度的首選)
    print("正在載入 CLIP 模型...")
    model, preprocess = clip.load("ViT-B/32", device=device)

    # 3. 準備測試標籤 (這是你想讓機器人「辨認」的選項)
    labels = ["a long-sleeved shirt", "a short-sleeved t-shirt", "jeans", "a dress", "a black jacket"]
    text = clip.tokenize(labels).to(device)

    # 4. 準備圖片 (請確保你 data/ 資料夾下有一張名為 test.jpg 的衣服照片)
    # 如果還沒有照片，可以先用一個簡單的 try-except 抓錯
    img_path = "data/test.jpg"
    if not os.path.exists(img_path):
        print(f"請在 {img_path} 放置一張照片來進行測試！")
        return

    image = preprocess(Image.open(img_path)).unsqueeze(0).to(device)

    # 5. 進行推論
    with torch.no_grad():
        logits_per_image, logits_per_text = model(image, text)
        probs = logits_per_image.softmax(dim=-1).cpu().numpy()

    # 6. 輸出結果
    print("\n--- 辨識結果 ---")
    for label, prob in zip(labels, probs[0]):
        print(f"{label}: {prob:.2%}")

if __name__ == "__main__":
    test_clip()