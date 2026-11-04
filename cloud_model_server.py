"""
Cloud AI Model Server: Dịch vụ suy luận YOLO Mask Detection chạy trên Cloud (AWS EC2).
- Tự động tải file mask_model.pt từ AWS S3 khi khởi động.
- Mở API endpoint POST /predict để tiếp nhận frame từ Local gửi lên và trả kết quả.
"""
import os
import cv2
import time
import boto3
import numpy as np
from flask import Flask, request, jsonify
from ultralytics import YOLO
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)

# --- CẤU HÌNH TẢI MODEL TỪ AWS S3 ---
AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
AWS_REGION = os.getenv("AWS_REGION", "us-east-1")
BUCKET_NAME = os.getenv("AWS_S3_BUCKET_NAME", "iot-lab-2-image")
S3_MODEL_KEY = os.getenv("S3_MODEL_KEY", "models/mask_model.pt")
MODEL_PATH = os.getenv("MODEL_PATH", "mask_model.pt")

if not os.path.exists(MODEL_PATH):
    if BUCKET_NAME and AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY:
        print(f"--> [Cloud AI] Chưa có file {MODEL_PATH}. Đang tải từ S3: s3://{BUCKET_NAME}/{S3_MODEL_KEY} ...")
        s3 = boto3.client(
            "s3",
            aws_access_key_id=AWS_ACCESS_KEY_ID,
            aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
            region_name=AWS_REGION,
        )
        s3.download_file(BUCKET_NAME, S3_MODEL_KEY, MODEL_PATH)
        print(f"--> [Cloud AI] Đã tải thành công {MODEL_PATH} từ S3!")
    else:
        raise FileNotFoundError(f"Không tìm thấy {MODEL_PATH} và chưa cấu hình AWS S3 trong .env")

print(f"--> [Cloud AI] Đang nạp mô hình YOLO từ {MODEL_PATH}...")
model = YOLO(MODEL_PATH)

MODEL_CLASS_ALIASES = {
    "incorrectly worn mask": "wrong mask",
    "wrong mask": "wrong mask",
    "with mask": "mask",
    "mask": "mask",
    "without mask": "no mask",
    "no mask": "no mask",
}
_model_names = model.names.items() if isinstance(model.names, dict) else enumerate(model.names)
classNames = {
    int(class_id): MODEL_CLASS_ALIASES.get(str(name).lower().replace("_", " ").strip(), "Unknown")
    for class_id, name in _model_names
}
print(f"--> [Cloud AI] Sẵn sàng phục vụ! Các nhãn nhận dạng: {classNames}")


@app.route("/", methods=["GET"])
def health():
    return jsonify({
        "status": "online",
        "service": "YOLO Mask Detection Cloud Service",
        "model": MODEL_PATH,
        "classes": classNames,
    })


@app.route("/predict", methods=["POST"])
def predict():
    """Nhận frame ảnh từ Local, chạy YOLO trên Cloud, đo inference time và trả về JSON."""
    image_bytes = None
    if "image" in request.files:
        image_bytes = request.files["image"].read()
    elif request.data:
        image_bytes = request.data

    if not image_bytes:
        return jsonify({"error": "Không tìm thấy dữ liệu ảnh trong request"}), 400

    frame = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)
    if frame is None:
        return jsonify({"error": "Không thể giải mã frame ảnh"}), 400

    # Chạy suy luận trên Cloud (tối ưu imgsz=480 để tiết kiệm RAM & tăng tốc)
    infer_start = time.perf_counter()
    results = model.predict(frame, imgsz=480, conf=0.12, iou=0.50, max_det=100, verbose=False)
    infer_ms = (time.perf_counter() - infer_start) * 1000

    detections = []
    for result in results:
        for box in result.boxes:
            class_id = int(box.cls[0])
            label = classNames.get(class_id, "Unknown")
            if label == "Unknown":
                continue
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            detections.append({
                "box": [x1, y1, x2, y2],
                "label": label,
                "class_id": class_id,
                "confidence": float(box.conf[0]),
            })

    return jsonify({
        "detections": detections,
        "inference_time_ms": round(infer_ms, 2),
    })


if __name__ == "__main__":
    port = int(os.getenv("PORT", 5000))
    print(f"--> [Cloud AI] Server đang lắng nghe tại port {port}...")
    app.run(host="0.0.0.0", port=port, debug=False)
