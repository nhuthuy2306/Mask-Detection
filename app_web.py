import os
import ast
import math
import cv2
import time
import uuid
import threading
import logging
import pytz
from datetime import datetime
from dotenv import load_dotenv
import onnxruntime as ort
from flask import Flask, render_template, jsonify, request, make_response
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy import inspect, text
from sqlalchemy.pool import NullPool

# Tải cấu hình từ file .env
load_dotenv()
HTML_TEMPLATE = os.getenv("HTML_TEMPLATE", os.getenv("TEMPLATE_FILE", "index.html"))

# Cấu hình Múi giờ Việt Nam
VN_TZ = pytz.timezone('Asia/Ho_Chi_Minh')

app = Flask(__name__)
app.logger.setLevel(logging.INFO)
app.config['MAX_CONTENT_LENGTH'] = 8 * 1024 * 1024

# --- CẤU HÌNH SUPABASE (POSTGRESQL) ---
database_url = os.getenv("DATABASE_URL")
if database_url and database_url.startswith("postgresql+psycopg://"):
    database_url = database_url.replace(
        "postgresql+psycopg://", "postgresql+psycopg2://", 1
    )
app.config['SQLALCHEMY_DATABASE_URI'] = database_url
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['SQLALCHEMY_ENGINE_OPTIONS'] = {"poolclass": NullPool}

db = SQLAlchemy(app)

# --- KHAI BÁO BẢNG DATABASE ---
class DetectionLog(db.Model):
    __tablename__ = 'detection_logs'
    id = db.Column(db.Integer, primary_key=True)
    status = db.Column(db.String(50), nullable=False)
    confidence = db.Column(db.String(10), nullable=False)
    track_id = db.Column(db.Integer, nullable=True)
    image_url = db.Column(db.String(255), nullable=True)
    created_at = db.Column(db.DateTime, default=lambda: datetime.now(VN_TZ).replace(tzinfo=None))

    # ---- Các cột đo độ trễ (ms) ----
    # a) Thời gian suy luận trên Cloud (inference time của model)
    inference_time_ms = db.Column(db.Float, nullable=True)
    # b) Thời gian phản hồi local <-> Cloud (upload ảnh + nhận kết quả)
    roundtrip_time_ms = db.Column(db.Float, nullable=True)
    # c) Độ trễ toàn trình: thời gian người dùng thực sự phải chờ
    #    = thời gian từ lúc frame capture đến khi kết quả hiển thị trên UI
    total_latency_ms = db.Column(db.Float, nullable=True)

    def to_dict(self):
        return {
            "id": self.id,
            "time": self.created_at.strftime("%H:%M:%S"),
            "status": self.status,
            "confidence": self.confidence,
            "track_id": self.track_id,
            "image_url": self.image_url,
            "date": self.created_at.strftime("%d/%m/%Y"),
            # Trả về độ trễ cho frontend
            "inference_time_ms": round(self.inference_time_ms, 1) if self.inference_time_ms is not None else None,
            "roundtrip_time_ms": round(self.roundtrip_time_ms, 1) if self.roundtrip_time_ms is not None else None,
            "total_latency_ms": round(self.total_latency_ms, 1) if self.total_latency_ms is not None else None,
        }


class LatencySample(db.Model):
    """Latency measurements collected by the laptop webcam interface."""
    __tablename__ = 'latency_samples'
    id = db.Column(db.Integer, primary_key=True)
    detection_log_id = db.Column(db.Integer, db.ForeignKey('detection_logs.id'), unique=True, nullable=False)
    inference_time_ms = db.Column(db.Float, nullable=False)
    roundtrip_time_ms = db.Column(db.Float, nullable=False)
    total_latency_ms = db.Column(db.Float, nullable=False)
    created_at = db.Column(db.DateTime, default=lambda: datetime.now(VN_TZ).replace(tzinfo=None), nullable=False)

MODEL_PATH = os.getenv("MODEL_PATH", "mask_model.onnx")
session_options = ort.SessionOptions()
session_options.intra_op_num_threads = 1
session_options.inter_op_num_threads = 1
session_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
model = ort.InferenceSession(
    MODEL_PATH,
    sess_options=session_options,
    providers=["CPUExecutionProvider"],
)
model_input = model.get_inputs()[0]
MODEL_SIZE = int(model_input.shape[2]) if isinstance(model_input.shape[2], int) else 960
metadata_names = model.get_modelmeta().custom_metadata_map.get("names")
if metadata_names:
    parsed_names = ast.literal_eval(metadata_names)
    CLASS_NAMES = [name for _, name in sorted((int(key), value) for key, value in parsed_names.items())] if isinstance(parsed_names, dict) else list(parsed_names)
else:
    CLASS_NAMES = ["wrong mask", "mask", "no mask"]
MODEL_CLASS_ALIASES = {
    "incorrectly worn mask": "wrong mask",
    "wrong mask": "wrong mask",
    "with mask": "mask",
    "mask": "mask",
    "without mask": "no mask",
    "no mask": "no mask",
}
_model_names = enumerate(CLASS_NAMES)
classNames = {
    int(class_id): MODEL_CLASS_ALIASES.get(str(name).lower().replace("_", " ").strip(), "Unknown")
    for class_id, name in _model_names
}
if set(classNames.values()) != {"wrong mask", "mask", "no mask"}:
    raise ValueError(f"Unexpected ONNX class metadata: {CLASS_NAMES}")
app.logger.info("Loaded mask model class mapping: %s", classNames)
app.logger.warning(
    "No standalone face detector is installed; repeated mask_model.pt boxes are used as face proposals."
)

colors = [
    (0, 255, 255),   # Vàng cho "Đeo sai" (wrong mask)
    (0, 255, 0),     # Xanh lá cho "Đeo đúng" (mask)
    (0, 0, 255)      # Đỏ cho "Không đeo" (no mask)
]


def predict_onnx(frame, conf_threshold=0.12, iou_threshold=0.50, max_det=100):
    """Run YOLOv8 ONNX detection and return xyxy, class, confidence tuples."""
    import numpy as np

    height, width = frame.shape[:2]
    scale = min(MODEL_SIZE / width, MODEL_SIZE / height)
    resized_w, resized_h = int(round(width * scale)), int(round(height * scale))
    resized = cv2.resize(frame, (resized_w, resized_h), interpolation=cv2.INTER_LINEAR)
    pad_x, pad_y = (MODEL_SIZE - resized_w) // 2, (MODEL_SIZE - resized_h) // 2
    canvas = np.full((MODEL_SIZE, MODEL_SIZE, 3), 114, dtype=np.uint8)
    canvas[pad_y:pad_y + resized_h, pad_x:pad_x + resized_w] = resized
    tensor = canvas[:, :, ::-1].transpose(2, 0, 1)[None].astype(np.float32) / 255.0
    output = model.run(None, {model_input.name: tensor})[0]
    predictions = np.squeeze(output)
    if predictions.ndim != 2:
        return []
    # YOLOv8 detection exports are (channels, candidates); tolerate transposed exports.
    if predictions.shape[0] < predictions.shape[1]:
        predictions = predictions.T
    if predictions.shape[1] != 4 + len(CLASS_NAMES):
        raise ValueError(f"Unexpected ONNX output shape: {output.shape}")
    boxes_xywh = predictions[:, :4]
    class_scores = predictions[:, 4:]
    class_ids = np.argmax(class_scores, axis=1)
    scores = class_scores[np.arange(len(class_ids)), class_ids]
    selected = scores >= conf_threshold
    boxes_xywh, class_ids, scores = boxes_xywh[selected], class_ids[selected], scores[selected]
    if not len(scores):
        return []
    xyxy = np.empty_like(boxes_xywh)
    xyxy[:, 0] = boxes_xywh[:, 0] - boxes_xywh[:, 2] / 2
    xyxy[:, 1] = boxes_xywh[:, 1] - boxes_xywh[:, 3] / 2
    xyxy[:, 2] = boxes_xywh[:, 0] + boxes_xywh[:, 2] / 2
    xyxy[:, 3] = boxes_xywh[:, 1] + boxes_xywh[:, 3] / 2
    xyxy[:, [0, 2]] = (xyxy[:, [0, 2]] - pad_x) / scale
    xyxy[:, [1, 3]] = (xyxy[:, [1, 3]] - pad_y) / scale
    xyxy[:, [0, 2]] = xyxy[:, [0, 2]].clip(0, width)
    xyxy[:, [1, 3]] = xyxy[:, [1, 3]].clip(0, height)
    nms_boxes = [[int(x), int(y), int(w), int(h)] for x, y, w, h in
                 zip(xyxy[:, 0], xyxy[:, 1], xyxy[:, 2] - xyxy[:, 0], xyxy[:, 3] - xyxy[:, 1])]
    indices = cv2.dnn.NMSBoxes(nms_boxes, scores.tolist(), conf_threshold, iou_threshold)
    return [(xyxy[i].astype(int).tolist(), int(class_ids[i]), float(scores[i]))
            for i in list(indices.flatten())[:max_det]]

# --- CẤU HÌNH AWS S3 ---
BUCKET_NAME = os.getenv('AWS_S3_BUCKET_NAME')
AWS_REGION = os.getenv('AWS_REGION')
s3_client = None
if BUCKET_NAME:
    import boto3

    s3_client = boto3.client(
        's3',
        aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
        aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY'),
        region_name=AWS_REGION,
    )


browser_tracks = {}
next_browser_track_id = 1
next_confirmed_track_id = 1
last_logged_status = None


def box_iou(box_a, box_b):
    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[2], box_b[2])
    y2 = min(box_a[3], box_b[3])
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    area_a = max(0, box_a[2] - box_a[0]) * max(0, box_a[3] - box_a[1])
    area_b = max(0, box_b[2] - box_b[0]) * max(0, box_b[3] - box_b[1])
    union = area_a + area_b - intersection
    return intersection / union if union > 0 else 0.0


def suppress_duplicate_boxes(detections):
    """Suppress near-identical boxes, including duplicate boxes with different classes."""
    kept = []
    for detection in sorted(detections, key=lambda item: item["confidence"], reverse=True):
        box = detection["box"]
        area = max(1, (box[2] - box[0]) * (box[3] - box[1]))
        center_x = (box[0] + box[2]) / 2
        center_y = (box[1] + box[3]) / 2
        diagonal = math.hypot(box[2] - box[0], box[3] - box[1])
        duplicate = False
        for existing in kept:
            other = existing["box"]
            other_area = max(1, (other[2] - other[0]) * (other[3] - other[1]))
            intersection = max(0, min(box[2], other[2]) - max(box[0], other[0])) * max(
                0, min(box[3], other[3]) - max(box[1], other[1])
            )
            other_x = (other[0] + other[2]) / 2
            other_y = (other[1] + other[3]) / 2
            center_distance = math.hypot(center_x - other_x, center_y - other_y)
            min_diagonal = min(diagonal, math.hypot(other[2] - other[0], other[3] - other[1]))
            intersection_over_min_area = intersection / max(1, min(area, other_area))
            iou = box_iou(box, other)
            same_object_overlap = iou >= 0.50 or intersection_over_min_area >= 0.80
            if same_object_overlap and center_distance <= max(12, min_diagonal * 0.22):
                duplicate = True
                break
        if not duplicate:
            kept.append(detection)
    return kept


def associate_and_stabilize(detections, now):
    """Track face proposals, stabilize mask labels, and emit event logs once."""
    global next_browser_track_id, next_confirmed_track_id
    unmatched_ids = set(browser_tracks)
    outputs = []
    log_candidates = []
    frame_boxes = []

    for detection in sorted(detections, key=lambda item: item["confidence"], reverse=True):
        box = detection["box"]
        # A final same-frame guard prevents a second proposal for the same face
        # from creating another track after the first track was already matched.
        area = max(1, (box[2] - box[0]) * (box[3] - box[1]))
        center_x = (box[0] + box[2]) / 2
        center_y = (box[1] + box[3]) / 2
        diagonal = math.hypot(box[2] - box[0], box[3] - box[1])
        duplicate_in_frame = False
        for previous_box in frame_boxes:
            previous_area = max(1, (previous_box[2] - previous_box[0]) * (previous_box[3] - previous_box[1]))
            intersection = max(0, min(box[2], previous_box[2]) - max(box[0], previous_box[0])) * max(
                0, min(box[3], previous_box[3]) - max(box[1], previous_box[1])
            )
            previous_center_x = (previous_box[0] + previous_box[2]) / 2
            previous_center_y = (previous_box[1] + previous_box[3]) / 2
            previous_diagonal = math.hypot(
                previous_box[2] - previous_box[0], previous_box[3] - previous_box[1]
            )
            centers_close = math.hypot(center_x - previous_center_x, center_y - previous_center_y) <= max(
                12, min(diagonal, previous_diagonal) * 0.22
            )
            overlap_ratio = intersection / min(area, previous_area)
            if centers_close and (box_iou(box, previous_box) >= 0.50 or overlap_ratio >= 0.80):
                duplicate_in_frame = True
                break
        if duplicate_in_frame:
            continue
        frame_boxes.append(box)

        best_track = None
        best_score = -1.0

        for track_id in unmatched_ids:
            track = browser_tracks[track_id]
            # Keep a briefly missed face attached to its existing track ID.
            if now - track["last_seen"] > 3.0:
                continue
            previous = track["box"]
            previous_x = (previous[0] + previous[2]) / 2
            previous_y = (previous[1] + previous[3]) / 2
            distance = math.hypot(center_x - previous_x, center_y - previous_y)
            missed_seconds = max(0.0, now - track["last_seen"])
            reference_diagonal = min(diagonal, track["diagonal"])
            max_distance = max(
                42,
                reference_diagonal * (0.45 + 0.15 * min(missed_seconds, 2.0)),
            )
            overlap = box_iou(box, previous)
            if overlap < 0.12 and distance > max_distance:
                continue
            score = overlap + max(0, 1 - distance / max_distance) * 0.25
            if score > best_score:
                best_score = score
                best_track = track

        if best_track is None:
            track_id = next_browser_track_id
            next_browser_track_id += 1
            best_track = {
                "id": track_id,
                "display_id": None,
                "box": box,
                "diagonal": diagonal,
                "first_seen": now,
                "last_seen": now,
                "seen_count": 0,
                "observations": [],
                "candidate_label": None,
                "candidate_since": now,
                "stable_label": None,
                "stable_confidence": 0.0,
                "last_class_seen": 0.0,
                "logged_label": None,
            }
            browser_tracks[track_id] = best_track
        else:
            track_id = best_track["id"]
            unmatched_ids.discard(track_id)
            best_track["box"] = box
            best_track["diagonal"] = diagonal
            best_track["last_seen"] = now

        best_track["seen_count"] += 1
        if detection["confidence"] >= 0.50:
            best_track["last_class_seen"] = now
            best_track["observations"].append((now, detection["label"], detection["confidence"]))
        best_track["observations"] = [
            item for item in best_track["observations"] if now - item[0] <= 5.0
        ]

        observations = best_track["observations"]
        if observations:
            scores = {}
            counts = {}
            for observed_at, label, confidence in observations:
                recency_weight = 0.75 + 0.25 * max(0, 1 - (now - observed_at) / 5.0)
                scores[label] = scores.get(label, 0.0) + confidence * recency_weight
                counts[label] = counts.get(label, 0) + 1
            ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
            winner, winner_score = ranked[0]
            total_score = sum(scores.values())
            second_score = ranked[1][1] if len(ranked) > 1 else 0.0
            average_confidence = winner_score / max(1, counts[winner])
            confident_majority = (
                len(observations) >= 3
                and winner_score / max(total_score, 1e-6) >= 0.62
                and (winner_score - second_score) / max(total_score, 1e-6) >= 0.18
                and average_confidence >= 0.50
            )
            if confident_majority and winner != best_track["candidate_label"]:
                best_track["candidate_label"] = winner
                best_track["candidate_since"] = now
            if (confident_majority and winner == best_track["candidate_label"]
                    and now - best_track["candidate_since"] >= 1.0):
                best_track["stable_label"] = winner
                best_track["stable_confidence"] = average_confidence
            elif confident_majority and best_track["stable_label"] == winner:
                best_track["stable_confidence"] = average_confidence

        label_is_stable = (
            best_track["candidate_label"] is not None
            and now - best_track["candidate_since"] >= 1.0
            and best_track["stable_label"] == best_track["candidate_label"]
        )
        class_is_fresh = now - best_track["last_class_seen"] <= 3.0
        classification_is_stable = label_is_stable and class_is_fresh
        if classification_is_stable and best_track["display_id"] is None:
            best_track["display_id"] = next_confirmed_track_id
            next_confirmed_track_id += 1
        final_label = (
            best_track["stable_label"] if classification_is_stable
            else best_track["candidate_label"] if class_is_fresh
            else "\u0110ang ph\u00e2n lo\u1ea1i"
        )
        final_confidence = (
            best_track["stable_confidence"] if classification_is_stable
            else detection["confidence"]
        )
        output = {
            "box": box,
            "label": final_label,
            "confidence": final_confidence,
            "track_id": best_track["display_id"],
            "tracking_id": track_id,
            "label_stable": classification_is_stable,
        }
        # Show the box as soon as YOLO detects the object; label confirmation
        # only gates logging, not whether the face is outlined on camera.
        outputs.append(output)

        # One event per track state: first stable label, then only a different
        # stable label. The saved label is advanced only after the DB commit.
        if (classification_is_stable
                and best_track["logged_label"] != best_track["stable_label"]
                and best_track["stable_label"] != last_logged_status):
            log_candidates.append((output, best_track))

    for track_id in list(browser_tracks):
        if now - browser_tracks[track_id]["last_seen"] > 4.0:
            del browser_tracks[track_id]
    return outputs, log_candidates

with app.app_context():
    db.create_all()
    log_columns = {column["name"] for column in inspect(db.engine).get_columns("detection_logs")}
    if "track_id" not in log_columns:
        # Add the nullable column for existing databases without dropping old logs.
        with db.engine.begin() as connection:
            connection.execute(text("ALTER TABLE detection_logs ADD COLUMN track_id INTEGER"))
    highest_track_id = db.session.query(db.func.max(DetectionLog.track_id)).scalar() or 0
    next_confirmed_track_id = highest_track_id + 1
    latest_log = DetectionLog.query.order_by(DetectionLog.id.desc()).first()
    if latest_log:
        last_logged_status = {
            "mask": "mask",
            "no_mask": "no mask",
            "wrong_mask": "wrong mask",
        }.get(latest_log.status)


def upload_detection_image_async(log_id, object_name, image_bytes):
    def upload_task():
        try:
            s3_client.put_object(Bucket=BUCKET_NAME, Key=object_name,
                                 Body=image_bytes, ContentType="image/jpeg")
            with app.app_context():
                log_entry = db.session.get(DetectionLog, log_id)
                if log_entry:
                    log_entry.image_url = f"s3://{BUCKET_NAME}/{object_name}"
                    db.session.commit()
        except Exception as exc:
            with app.app_context():
                db.session.rollback()
            app.logger.warning("S3 upload failed for detection %s: %s", log_id, exc)

    threading.Thread(target=upload_task, daemon=True).start()


@app.route('/')
def index():
    response = make_response(render_template(HTML_TEMPLATE))
    response.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate'
    response.headers['Pragma'] = 'no-cache'
    response.headers['Expires'] = '0'
    return response


@app.route('/api/detect', methods=['POST'])
def detect_frame():
    """Run mask detection on the latest laptop webcam frame."""
    global last_logged_status
    payload = request.get_json(silent=True) or {}
    if not isinstance(payload, dict):
        return jsonify({"error": "Body must be a JSON object"}), 400
    image_data = payload.get("image", "")
    if not isinstance(image_data, str) or not image_data.startswith("data:image/") or "," not in image_data:
        return jsonify({"error": "A valid JPEG frame is required"}), 400

    import base64
    import numpy as np

    try:
        frame_bytes = base64.b64decode(image_data.split(",", 1)[1], validate=True)
        frame = cv2.imdecode(np.frombuffer(frame_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
        if frame is None:
            return jsonify({"error": "Could not decode the webcam frame"}), 400
    except (ValueError, TypeError):
        return jsonify({"error": "Invalid image data"}), 400

    request_started = time.perf_counter()
    frame_height, frame_width = frame.shape[:2]
    infer_started = time.perf_counter()
    results = predict_onnx(frame, conf_threshold=0.12, iou_threshold=0.50, max_det=100)
    inference_ms = (time.perf_counter() - infer_started) * 1000

    raw_detections = []
    for box, class_id, confidence in results:
        label = classNames.get(class_id, "Unknown")
        if label == "Unknown":
            continue
        raw_detections.append({
            "box": box,
            "label": label,
            "class_id": class_id,
            "confidence": confidence,
        })

    # The checkpoint has only mask-state classes, so its boxes serve as face proposals.
    deduplicated = suppress_duplicate_boxes(raw_detections)
    candidates = [item for item in deduplicated if item["confidence"] >= 0.12]
    detections, log_candidates = associate_and_stabilize(candidates, time.monotonic())

    log_ids = []
    cloud_error = None
    upload_bytes = None
    if log_candidates and BUCKET_NAME:
        encoded_ok, encoded_frame = cv2.imencode(".jpg", frame)
        if encoded_ok:
            upload_bytes = encoded_frame.tobytes()

    if log_candidates:
        try:
            with app.app_context():
                entries = []
                for detection, track in log_candidates:
                    canonical_status = {
                        "mask": "mask",
                        "no mask": "no_mask",
                        "wrong mask": "wrong_mask",
                    }[track["stable_label"]]
                    entry = DetectionLog(
                        status=canonical_status,
                        confidence=f"{detection['confidence'] * 100:.0f}%",
                        track_id=detection["track_id"],
                        image_url=None,
                        inference_time_ms=inference_ms,
                        roundtrip_time_ms=None,
                        total_latency_ms=None,
                    )
                    db.session.add(entry)
                    entries.append((entry, detection, track, canonical_status))
                db.session.commit()

                for entry, detection, track, canonical_status in entries:
                    track["logged_label"] = track["stable_label"]
                    last_logged_status = track["stable_label"]
                    detection["log_id"] = entry.id
                    log_ids.append(entry.id)
                    app.logger.info(
                        "Stored result=%s track_id=%s confidence=%.3f",
                        canonical_status, detection["track_id"], detection["confidence"],
                    )
                    if upload_bytes:
                        object_name = (
                            f"mask_logs/{datetime.now(VN_TZ):%Y%m%d_%H%M%S}_"
                            f"{canonical_status}_{uuid.uuid4().hex[:6]}.jpg"
                        )
                        upload_detection_image_async(entry.id, object_name, upload_bytes)
        except Exception as exc:
            db.session.rollback()
            app.logger.exception("Could not save stable mask detections: %s", exc)
            cloud_error = "Could not save stable detections to the database."

    processing_ms = (time.perf_counter() - request_started) * 1000
    return jsonify({
        "detections": detections,
        "log_ids": log_ids,
        "log_id": log_ids[0] if log_ids else None,
        "inference_time_ms": round(inference_ms, 1),
        "server_processing_ms": round(processing_ms, 1),
        "frame_width": frame_width,
        "frame_height": frame_height,
        "cloud_error": cloud_error,
    })


@app.route('/api/latency/record', methods=['POST'])
def record_client_latency():
    """Lưu RTT và độ trễ đầu-cuối do trình duyệt đo bằng đồng hồ đơn điệu."""
    payload = request.get_json(silent=True) or {}
    if not isinstance(payload, dict):
        return jsonify({"error": "Body phải là JSON object"}), 400
    try:
        log_id = int(payload.get('log_id'))
        roundtrip_ms = float(payload.get('roundtrip_time_ms'))
        total_ms = float(payload.get('total_latency_ms'))
        if (not math.isfinite(roundtrip_ms) or not math.isfinite(total_ms)
                or min(roundtrip_ms, total_ms) < 0):
            raise ValueError("latency must be non-negative")
    except (TypeError, ValueError):
        return jsonify({"error": "Số đo độ trễ không hợp lệ"}), 400

    try:
        log_entry = db.session.get(DetectionLog, log_id)
    except Exception as exc:
        db.session.rollback()
        app.logger.warning("Could not read detection log for latency update: %s", exc)
        return jsonify({"error": "Không kết nối được cơ sở dữ liệu để lưu độ trễ"}), 503
    if log_entry is None:
        return jsonify({"error": "Không tìm thấy log nhận diện"}), 404
    try:
        log_entry.roundtrip_time_ms = roundtrip_ms
        log_entry.total_latency_ms = total_ms
        db.session.add(LatencySample(
            detection_log_id=log_entry.id,
            inference_time_ms=log_entry.inference_time_ms,
            roundtrip_time_ms=roundtrip_ms,
            total_latency_ms=total_ms,
        ))
        db.session.commit()
    except Exception as exc:
        db.session.rollback()
        app.logger.warning("Could not update client latency: %s", exc)
        return jsonify({"error": "Không thể lưu số đo độ trễ vào cơ sở dữ liệu"}), 500
    return jsonify({"status": "saved"})


@app.route('/api/logs')
def get_logs():
    accepted_statuses = (
        "mask", "no_mask", "wrong_mask",
        "no mask", "wrong mask", "with mask", "with_mask",
        "without mask", "without_mask", "incorrectly worn mask", "incorrectly_worn_mask",
    )
    status_aliases = {
        "with mask": "mask",
        "with_mask": "mask",
        "no mask": "no_mask",
        "without mask": "no_mask",
        "without_mask": "no_mask",
        "wrong mask": "wrong_mask",
        "incorrectly worn mask": "wrong_mask",
        "incorrectly_worn_mask": "wrong_mask",
    }
    logs = (DetectionLog.query
            .filter(DetectionLog.status.in_(accepted_statuses))
            .order_by(DetectionLog.created_at.desc()).limit(50).all())
    output = []
    for log in logs:
        item = log.to_dict()
        item["status"] = status_aliases.get(item["status"], item["status"])
        output.append(item)
    return jsonify(output)


@app.route('/api/latency')
def get_latency_stats():
    """
    Trả về thống kê độ trễ trung bình của 50 lần đo gần nhất.

    Công thức:
      a) inference_time_ms   = T_infer_end - T_infer_start (model trên máy chạy Flask)
      b) roundtrip_time_ms   = T_response_received - T_request_sent (trình duyệt ↔ Flask)
      c) total_latency_ms    = T_result_ready - T_frame_capture
    """
    samples = LatencySample.query.order_by(LatencySample.created_at.desc()).limit(50).all()
    inference_vals = [sample.inference_time_ms for sample in samples]
    roundtrip_vals = [sample.roundtrip_time_ms for sample in samples]
    total_vals = [sample.total_latency_ms for sample in samples]

    def stats(vals):
        if not vals:
            return {"avg": None, "min": None, "max": None, "count": 0}
        return {
            "avg": round(sum(vals) / len(vals), 1),
            "min": round(min(vals), 1),
            "max": round(max(vals), 1),
            "count": len(vals)
        }

    return jsonify({
        "inference_time_ms": stats(inference_vals),
        "roundtrip_time_ms": stats(roundtrip_vals),
        "total_latency_ms": stats(total_vals),
        "formulas": {
            "inference_time_ms": "T_infer_end - T_infer_start  [model YOLO trên server Flask]",
            "roundtrip_time_ms": "T_response_received - T_request_sent  [trình duyệt ↔ Flask/cloud]",
            "total_latency_ms": "T_result_ready - T_frame_capture  [toàn trình người dùng chờ]"
        }
    })


@app.route('/api/logs/clear', methods=['GET', 'POST'])
def clear_logs():
    global browser_tracks, next_browser_track_id, last_logged_status
    browser_tracks = {}
    next_browser_track_id = 1
    try:
        db.session.query(LatencySample).delete()
        num_rows_deleted = db.session.query(DetectionLog).delete()
        db.session.commit()
        last_logged_status = None
        return jsonify({"status": "success", "message": f"Đã xóa {num_rows_deleted} record trong DB"})
    except Exception as e:
        db.session.rollback()
        return jsonify({"status": "error", "message": str(e)}), 500


if __name__ == '__main__':
    host = os.getenv('HOST', '0.0.0.0')
    port = int(os.getenv('PORT', 5000))
    app.run(host=host, port=port, debug=True)
