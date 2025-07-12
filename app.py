from flask import Flask, request, jsonify
import cv2
import numpy as np
import joblib
import os
from config import MODEL_PATH
from detectors.feature_extractor import extract_pose_features

app = Flask(__name__)
model = joblib.load(MODEL_PATH)

# ✅ 加入 App Inventor 測試專用的連線確認訊息
@app.route('/')
def home():
    return jsonify({
        'status': 'ok',
        'message': '✅ 體態偵測 API 已啟動（App Inventor 測試成功）'
    })

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': '未收到圖片檔案'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': '檔案名稱為空'}), 400

    try:
        img_bytes = file.read()
        nparr = np.frombuffer(img_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if image is None:
            return jsonify({'error': '無法解析圖片'}), 400

        features = extract_pose_features(image)
        if not features or len(features) != 66:
            return jsonify({'error': '特徵擷取失敗'}), 400

        result = model.predict([features])[0]
        label_map = {0: 'normal', 1: 'hunchback'}
        return jsonify({'result': label_map.get(result, 'unknown')})

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True, port=5000)
