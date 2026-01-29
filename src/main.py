import os
from database.database import WardrobeDB
# 假設您之前的 CLIP 類別存成了 classifier.py，若沒有，請將那段 Class 貼過來
# 這裡我們直接引用上一段對話的 WardrobeClassifier 邏輯
from classifier.clip_test import WardrobeClassifier 

def main():
    # 1. 初始化系統
    print("=== 系統啟動中 ===")
    db = WardrobeDB() # 初始化資料庫
    ai = WardrobeClassifier() # 載入 AI 模型 (會花一點時間)
    
    # 2. 模擬使用者上傳流程
    user_id = "henry" # 模擬 Line User ID
    
    # 請確保這裡有一張圖片，或是修改路徑
    current_image_path = "data/test.jpg" 
    
    if not os.path.exists(current_image_path):
        print(f"錯誤：找不到測試圖片 {current_image_path}，請放入圖片後再試。")
        return

    print(f"\n--- 收到使用者 {user_id} 上傳圖片 ---")

    # 3. AI 進行分類 (第一層：這是什麼衣服？)
    type_labels = ["t-shirt", "pants", "skirt", "shoes", "jacket"]
    category, conf = ai.predict(current_image_path, type_labels)
    print(f"AI 初步判定類別: {category} (信心度: {conf:.1%})")

    # 4. AI 進行風格判定 (第二層：這是什麼場合穿的？)
    # 這就是您提到的「運動系列」篩選關鍵
    style_labels = ["sports wear", "formal business", "casual daily", "party dress"]
    style_raw, style_conf = ai.predict(current_image_path, style_labels)
    
    # 簡化標籤名稱 (sports wear -> sports) 以便存入 DB
    style_tag = style_raw.split()[0] 
    print(f"AI 風格判定: {style_tag} (信心度: {style_conf:.1%})")

    # 5. 存入資料庫 (含去重檢查)
    if conf > 0.6:
        try:
            # 注意這裡接收兩個回傳值
            clothing_id, is_new = db.add_clothing(
                user_id=user_id,
                image_path=current_image_path,
                category=category,
                style_tag=style_tag,
                confidence=float(conf)
            )
            
            if is_new:
                print(f">> ✅ 新衣物已入庫！ ID: {clothing_id}")
            else:
                print(f">> ⚠️ 這件衣服之前存過了 (ID: {clothing_id})，跳過儲存。")
                
        except Exception as e:
            print(f"資料庫錯誤: {e}")
    else:
        print(">> 信心不足，轉入人工確認流程...")

    # 6. 模擬使用者查詢：「我需要運動系列的建議」
    print("\n--- 使用者查詢：『請給我運動系列的建議』 ---")
    recommendations = db.get_recommendation(user_id, style_tag="sports")
    
    if recommendations:
        print(f"系統找到 {len(recommendations)} 組建議：")
        for item in recommendations:
            print(f"- ID:{item['id']} | {item['category']} ({item['style_tag']}) | 權重:{item['weight']}")
            
            # 7. 模擬使用者點選了這一件 -> 權重增加
            print("   (使用者點選了這件，系統調整權重...)")
            db.update_weight(item['id'], delta=0.5)
    else:
        print("目前沒有運動類型的衣服庫存。")

if __name__ == "__main__":
    main()