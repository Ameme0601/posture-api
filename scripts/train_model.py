# scripts/train_model.py

import os
import cv2
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import StandardScaler
from detectors.feature_extractor import extract_pose_features
from config import MODEL_PATH

TRAIN_DIR = r"E:\pose_app\data\raw_images"

LABELS = {
    "normal": 0,
    "hunchback": 1
}

features = []
labels = []

print("🔍 開始讀取訓練資料...")

for label_name, label_value in LABELS.items():
    folder_path = os.path.join(TRAIN_DIR, label_name)
    print(f"🧭 嘗試讀取：{folder_path}")

    if not os.path.exists(folder_path):
        print(f"⚠️ 資料夾不存在：{folder_path}")
        continue

    for fname in os.listdir(folder_path):
        if not fname.lower().endswith(('.jpg', '.jpeg', '.png')):
            continue

        img_path = os.path.join(folder_path, fname)
        image = cv2.imread(img_path)

        if image is None:
            print(f"❌ 圖片讀取失敗：{img_path}")
            continue

        feature = extract_pose_features(image)

        if feature and len(feature) == 66:
            features.append(feature)
            labels.append(label_value)
        else:
            print(f"⚠️ 特徵擷取失敗或點數不足：{img_path}")

print(f"\n✅ 成功擷取特徵數量：{len(features)} 張圖片")

if len(features) < 2:
    print("❌ 訓練資料不足，請提供更多圖片")
    exit()

# 標準化特徵
scaler = StandardScaler()
X_scaled = scaler.fit_transform(features)

# 分割資料集
X_train, X_test, y_train, y_test = train_test_split(X_scaled, labels, test_size=0.2, random_state=42)

# 訓練模型
clf = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42)
clf.fit(X_train, y_train)

# 交叉驗證
scores = cross_val_score(clf, X_scaled, labels, cv=5)
print(f"📊 交叉驗證準確率：{scores.mean():.2f}")

# 評估
y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"\n📊 測試集準確率：{accuracy:.2f}")
print("📄 詳細報告：")
print(classification_report(y_test, y_pred))

# 儲存模型
os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
joblib.dump(clf, MODEL_PATH)
print(f"\n✅ 模型已儲存至：{MODEL_PATH}")