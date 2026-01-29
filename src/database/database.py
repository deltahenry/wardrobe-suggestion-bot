import sqlite3
import hashlib
import os

class WardrobeDB:
    def __init__(self, db_name="wardrobe.db"):
        self.db_name = db_name
        self.init_db()

    def get_connection(self):
        return sqlite3.connect(self.db_name)

    def calculate_hash(self, image_path):
        """
        計算圖片的 MD5 Hash 值 (數位指紋)
        """
        hash_md5 = hashlib.md5()
        try:
            with open(image_path, "rb") as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    hash_md5.update(chunk)
            return hash_md5.hexdigest()
        except FileNotFoundError:
            return None

    def init_db(self):
        """ 初始化資料庫 """
        conn = self.get_connection()
        cursor = conn.cursor()
        
        # 建立 clothes 資料表 (含 image_hash 欄位)
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS clothes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT NOT NULL,
                image_path TEXT NOT NULL,
                image_hash TEXT UNIQUE, 
                category TEXT NOT NULL,
                style_tag TEXT, 
                confidence REAL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                weight REAL DEFAULT 1.0
            )
        ''')
        conn.commit()
        conn.close()

    def add_clothing(self, user_id, image_path, category, style_tag, confidence):
        """ 
        新增衣服 (含去重邏輯)
        回傳: (id, is_new)
        """
        # 1. 計算 Hash
        img_hash = self.calculate_hash(image_path)
        if not img_hash:
            raise ValueError("找不到圖片檔案，無法計算 Hash")

        conn = self.get_connection()
        cursor = conn.cursor()
        
        # 2. 檢查是否已存在
        cursor.execute("SELECT id, category FROM clothes WHERE image_hash = ?", (img_hash,))
        existing_data = cursor.fetchone()
        
        if existing_data:
            print(f"發現重複圖片！已存在於 ID: {existing_data[0]} (類別: {existing_data[1]})")
            conn.close()
            return existing_data[0], False
            
        # 3. 如果沒找到，執行插入
        try:
            cursor.execute('''
                INSERT INTO clothes (user_id, image_path, image_hash, category, style_tag, confidence)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (user_id, image_path, img_hash, category, style_tag, confidence))
            
            conn.commit()
            last_id = cursor.lastrowid
            conn.close()
            return last_id, True
            
        except sqlite3.IntegrityError:
            conn.close()
            return None, False

    def get_recommendation(self, user_id, style_tag=None):
        """ 取得穿搭建議 """
        conn = self.get_connection()
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        query = "SELECT * FROM clothes WHERE user_id = ?"
        params = [user_id]
        
        if style_tag:
            query += " AND style_tag LIKE ?"
            params.append(f"%{style_tag}%")
            
        # 依照權重降序排列
        query += " ORDER BY weight DESC LIMIT 3"
        
        cursor.execute(query, params)
        results = [dict(row) for row in cursor.fetchall()]
        conn.close()
        return results

    def update_weight(self, clothing_id, delta=0.1):
        """ 
        更新權重功能
        當使用者點選某件衣服時，增加它的權重
        """
        conn = self.get_connection()
        cursor = conn.cursor()
        try:
            cursor.execute('''
                UPDATE clothes SET weight = weight + ? WHERE id = ?
            ''', (delta, clothing_id))
            conn.commit()
            print(f"ID {clothing_id} 權重已更新 (delta: {delta})")
        except Exception as e:
            print(f"更新權重失敗: {e}")
        finally:
            conn.close()