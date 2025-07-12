# scripts/test_local_predict.py

import cv2
import os
import joblib
import numpy as np
from detectors.feature_extractor import extract_pose_features
from config import MODEL_PATH
import logging

# 設置日誌
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

# 自動從 scripts 回推到專案根目錄
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
TEST_DIR = os.path.join(BASE_DIR, 'data', 'test_images')

# 如果模型是 0/1 數字 label
label_map = {0: "normal", 1: "hunchback"}
EXPECTED_FEATURE_LENGTH = 66

try:
    model = joblib.load(MODEL_PATH)
    logging.info("模型加載成功")
except Exception as e:
    logging.error(f"模型加載失敗: {e}")
    exit(1)

logging.info("🔍 開始批次測試...\n")

for fname in os.listdir(TEST_DIR):
    if not fname.lower().endswith(('.jpg', '.jpeg', '.png')):
        continue

    img_path = os.path.join(TEST_DIR, fname)
    image = cv2.imread(img_path)

    logging.info(f"📷 圖片檔案：{fname}")

    if image is None:
        logging.warning("❌ 圖片讀取失敗，跳過\n")
        continue

    features = extract_pose_features(image)

    if not features or len(features) != EXPECTED_FEATURE_LENGTH:
        logging.warning(f"⚠️ 特徵擷取失敗或點數不足（目前長度：{len(features) if features else 0}）\n")
        continue

    logging.info(f"🧠 特徵長度：{len(features)}")
    logging.info(f"🔬 前5個特徵值：{np.round(features[:5], 3)}")

    result = model.predict([features])[0]

    # 顯示機率（若支援）
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba([features])[0]
        logging.info(f"📈 預測機率：normal={proba[0]:.2f}, hunchback={proba[1]:.2f}")
    else:
        logging.warning("⚠️ 模型不支援 predict_proba")

    label = label_map.get(result, result)  # 若是字串就直接印
    logging.info(f"✅ 預測結果：{label}\n")