import os
import ast
import math
import cv2
import time
import uuid
import threading
from concurrent.futures import ThreadPoolExecutor
import logging
import json
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen
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


def env_int(name, default, minimum, maximum):
    try:
        value = int(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        value = default
    return min(maximum, max(minimum, value))


def env_float(name, default, minimum, maximum):
    try:
        value = float(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        value = default
    if not math.isfinite(value):
        value = default
    return min(maximum, max(minimum, value))

# Cấu hình Múi giờ Việt Nam
VN_TZ = pytz.timezone('Asia/Ho_Chi_Minh')

app = Flask(__name__)
app.logger.setLevel(logging.INFO)
MAX_FRAME_REQUEST_MB = env_int("MAX_FRAME_REQUEST_MB", 8, 1, 8)
app.config['MAX_CONTENT_LENGTH'] = MAX_FRAME_REQUEST_MB * 1024 * 1024

ONNX_INTRA_OP_NUM_THREADS = env_int("ONNX_INTRA_OP_NUM_THREADS", 1, 1, 2)
BACKGROUND_WORKERS = env_int("BACKGROUND_WORKERS", 1, 1, 2)
BACKGROUND_MAX_PENDING_TASKS = env_int("BACKGROUND_MAX_PENDING_TASKS", 8, 1, 32)
MAX_TRACK_SESSIONS = env_int("MAX_TRACK_SESSIONS", 4, 1, 64)
MAX_DETECTIONS_PER_FRAME = env_int("MAX_DETECTIONS_PER_FRAME", 100, 1, 100)
MAX_TRACKS_PER_SESSION = env_int(
    "MAX_TRACKS_PER_SESSION", 64, 1, 256
)
MAX_TRACK_OBSERVATIONS = env_int("MAX_TRACK_OBSERVATIONS", 6, 2, 16)
TRACK_SESSION_TTL_SECONDS = env_float("TRACK_SESSION_TTL_SECONDS", 60, 10, 600)
TRACK_EXPIRY_SECONDS = env_float("TRACK_EXPIRY_SECONDS", 3, 1, 10)
CAMERA_EVENT_DEDUPE_TTL_SECONDS = env_float(
    "CAMERA_EVENT_DEDUPE_TTL_SECONDS", 300, 30, 3600
)
MAX_CAMERA_EVENT_STATES = env_int("MAX_CAMERA_EVENT_STATES", 256, 64, 4096)

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "").strip()
if bool(TELEGRAM_BOT_TOKEN) != bool(TELEGRAM_CHAT_ID):
    app.logger.warning("Telegram alerts are disabled: configure both Telegram settings.")


def send_telegram_detection_async(
        log_id, status, confidence, track_id, created_at, image_bytes=None, session_id=None):
    """Send a saved detection alert and its webcam frame without delaying the API response."""
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        return

    status_names = {
        "mask": "Đeo khẩu trang đúng",
        "no_mask": "Không đeo khẩu trang",
        "wrong_mask": "Đeo khẩu trang sai cách",
    }
    message = (
        "🔔 Phát hiện mới\n"
        f"Trạng thái: {status_names.get(status, status)}\n"
        f"Độ tin cậy: {confidence}\n"
        f"Phiên/track: {(session_id[:8] + '/') if session_id else ''}"
        f"{track_id if track_id is not None else 'N/A'}\n"
        f"Thời gian: {created_at.strftime('%d/%m/%Y %H:%M:%S')}\n"
        f"Mã log: {log_id}"
    )

    def send_task():
        if not image_bytes:
            endpoint = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
            payload = json.dumps({"chat_id": TELEGRAM_CHAT_ID, "text": message}).encode("utf-8")
            telegram_request = Request(
                endpoint, data=payload, headers={"Content-Type": "application/json"}, method="POST"
            )
        else:
            boundary = f"----MaskAlert{uuid.uuid4().hex}"
            endpoint = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendPhoto"
            fields = [("chat_id", TELEGRAM_CHAT_ID), ("caption", message)]
            body = bytearray()
            for name, value in fields:
                body.extend(f"--{boundary}\r\n".encode())
                body.extend(f'Content-Disposition: form-data; name="{name}"\r\n\r\n'.encode())
                body.extend(value.encode("utf-8"))
                body.extend(b"\r\n")
            body.extend(f"--{boundary}\r\n".encode())
            body.extend(b'Content-Disposition: form-data; name="photo"; filename="detection.jpg"\r\n')
            body.extend(b"Content-Type: image/jpeg\r\n\r\n")
            body.extend(image_bytes)
            body.extend(b"\r\n")
            body.extend(f"--{boundary}--\r\n".encode())
            telegram_request = Request(
                endpoint, data=bytes(body),
                headers={"Content-Type": f"multipart/form-data; boundary={boundary}"}, method="POST"
            )
        try:
            with urlopen(telegram_request, timeout=8) as response:
                result = json.loads(response.read().decode("utf-8"))
            if not result.get("ok"):
                app.logger.warning(
                    "Telegram rejected detection alert for log %s (API error code %s).",
                    log_id, result.get("error_code", "unknown"),
                )
        except HTTPError as exc:
            app.logger.warning("Telegram returned HTTP %s for log %s.", exc.code, log_id)
        except (URLError, TimeoutError, OSError, ValueError) as exc:
            # Avoid logging exception text because it may include the tokenized URL.
            app.logger.warning(
                "Could not send Telegram alert for log %s (%s).", log_id, type(exc).__name__
            )

    submit_bounded_background_task(send_task, "Telegram alert")

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
    session_id = db.Column(db.String(36), nullable=True)
    object_track_id = db.Column(db.Integer, nullable=True)

    def to_dict(self):
        object_track_id = self.object_track_id
        return {
            "id": self.id,
            "log_id": self.id,
            "time": self.created_at.strftime("%H:%M:%S"),
            "status": self.status,
            "confidence": self.confidence,
            "track_id": object_track_id if object_track_id is not None else self.track_id,
            "object_track_id": object_track_id,
            "session_id": self.session_id,
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
session_options.intra_op_num_threads = ONNX_INTRA_OP_NUM_THREADS
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


def predict_onnx(frame, conf_threshold=0.12, iou_threshold=0.50, max_det=100, return_timings=False):
    """Run YOLOv8 ONNX detection and return xyxy, class, confidence tuples."""
    import numpy as np

    total_started = time.perf_counter()
    height, width = frame.shape[:2]
    scale = min(MODEL_SIZE / width, MODEL_SIZE / height)
    resized_w, resized_h = int(round(width * scale)), int(round(height * scale))
    resized = cv2.resize(frame, (resized_w, resized_h), interpolation=cv2.INTER_LINEAR)
    pad_x, pad_y = (MODEL_SIZE - resized_w) // 2, (MODEL_SIZE - resized_h) // 2
    canvas = np.full((MODEL_SIZE, MODEL_SIZE, 3), 114, dtype=np.uint8)
    canvas[pad_y:pad_y + resized_h, pad_x:pad_x + resized_w] = resized
    tensor = canvas[:, :, ::-1].transpose(2, 0, 1)[None].astype(np.float32) / 255.0
    preprocess_ms = (time.perf_counter() - total_started) * 1000
    model_started = time.perf_counter()
    output = model.run(None, {model_input.name: tensor})[0]
    model_ms = (time.perf_counter() - model_started) * 1000
    postprocess_started = time.perf_counter()
    predictions = np.squeeze(output)
    if predictions.ndim != 2:
        result = []
        timings = {"preprocess_time_ms": preprocess_ms, "onnx_time_ms": model_ms,
                   "postprocess_time_ms": (time.perf_counter() - postprocess_started) * 1000}
        return (result, timings) if return_timings else result
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
        result = []
        timings = {"preprocess_time_ms": preprocess_ms, "onnx_time_ms": model_ms,
                   "postprocess_time_ms": (time.perf_counter() - postprocess_started) * 1000}
        return (result, timings) if return_timings else result
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
    indices = class_aware_nms(
        nms_boxes, class_ids.tolist(), scores.tolist(), conf_threshold, iou_threshold, max_det
    )
    result = [(xyxy[i].astype(int).tolist(), int(class_ids[i]), float(scores[i]))
              for i in indices]
    timings = {"preprocess_time_ms": preprocess_ms, "onnx_time_ms": model_ms,
               "postprocess_time_ms": (time.perf_counter() - postprocess_started) * 1000}
    return (result, timings) if return_timings else result

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


browser_track_sessions = {}
browser_track_lock = threading.Lock()
event_commit_lock = threading.RLock()
camera_event_states = []
background_executor = ThreadPoolExecutor(
    max_workers=BACKGROUND_WORKERS, thread_name_prefix="mask-background"
)
background_task_slots = threading.BoundedSemaphore(BACKGROUND_MAX_PENDING_TASKS)


def background_capacity_available():
    if not background_task_slots.acquire(blocking=False):
        return False
    background_task_slots.release()
    return True


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


def normalize_box(box, frame_width, frame_height):
    return [
        min(1.0, max(0.0, float(box[0]) / max(frame_width, 1))),
        min(1.0, max(0.0, float(box[1]) / max(frame_height, 1))),
        min(1.0, max(0.0, float(box[2]) / max(frame_width, 1))),
        min(1.0, max(0.0, float(box[3]) / max(frame_height, 1))),
    ]


def camera_event_boxes_match(box_a, box_b):
    """Match a recent event zone by geometry only; this is not person identity."""
    overlap = box_iou(box_a, box_b)
    center_a = ((box_a[0] + box_a[2]) / 2, (box_a[1] + box_a[3]) / 2)
    center_b = ((box_b[0] + box_b[2]) / 2, (box_b[1] + box_b[3]) / 2)
    distance = math.hypot(center_a[0] - center_b[0], center_a[1] - center_b[1])
    diagonal_a = math.hypot(box_a[2] - box_a[0], box_a[3] - box_a[1])
    diagonal_b = math.hypot(box_b[2] - box_b[0], box_b[3] - box_b[1])
    area_a = max(1e-6, (box_a[2] - box_a[0]) * (box_a[3] - box_a[1]))
    area_b = max(1e-6, (box_b[2] - box_b[0]) * (box_b[3] - box_b[1]))
    area_similarity = min(area_a, area_b) / max(area_a, area_b)
    return area_similarity >= 0.35 and (
        overlap >= 0.18
        or distance <= max(0.025, min(diagonal_a, diagonal_b) * 0.35)
    )


def prepare_camera_event_candidates(
        log_candidates, camera_scope_id, frame_width, frame_height, now):
    """Filter same-state events repeated by track resets or a webcam reconnect.

    A camera scope is a per-tab random UUID persisted in sessionStorage. Matching
    uses only recent box geometry and status, never face appearance or identity.
    """
    status_by_label = {
        "mask": "mask",
        "no mask": "no_mask",
        "wrong mask": "wrong_mask",
    }
    global camera_event_states
    camera_event_states = [
        state for state in camera_event_states
        if now - state["last_seen"] <= CAMERA_EVENT_DEDUPE_TTL_SECONDS
    ]
    selected = []
    suppressed = []
    state_updates = []
    used_states = set()

    for detection, track in log_candidates:
        status = status_by_label[detection["event_label"]]
        box = normalize_box(detection["box"], frame_width, frame_height)
        matched_state = None
        if camera_scope_id:
            matches = [
                state for state in camera_event_states
                if state["camera_scope_id"] == camera_scope_id
                and id(state) not in used_states
                and camera_event_boxes_match(state["box"], box)
            ]
            if matches:
                matched_state = min(
                    matches,
                    key=lambda state: math.hypot(
                        (state["box"][0] + state["box"][2] - box[0] - box[2]) / 2,
                        (state["box"][1] + state["box"][3] - box[1] - box[3]) / 2,
                    ),
                )
                used_states.add(id(matched_state))

        if matched_state is not None and matched_state["status"] == status:
            # An earlier committed event already represents this active state.
            matched_state["last_seen"] = now
            matched_state["box"] = box
            detection["duplicate_event"] = {
                "log_id": matched_state.get("log_id"),
                "session_id": matched_state.get("session_id"),
                "object_track_id": matched_state.get("object_track_id"),
            }
            suppressed.append((detection, track))
            continue

        selected.append((detection, track, status))
        if camera_scope_id:
            state_updates.append((matched_state, camera_scope_id, status, box))

    return selected, suppressed, state_updates


def refresh_camera_event_states(detections, camera_scope_id, frame_width, frame_height, now):
    """Refresh bounded event zones from confirmed visible detections each frame."""
    if not camera_scope_id:
        return
    global camera_event_states
    camera_event_states = [
        state for state in camera_event_states
        if now - state["last_seen"] <= CAMERA_EVENT_DEDUPE_TTL_SECONDS
    ]
    used_states = set()
    for detection in detections:
        if not detection.get("label_stable"):
            continue
        box = normalize_box(detection["box"], frame_width, frame_height)
        matches = [
            state for state in camera_event_states
            if state["camera_scope_id"] == camera_scope_id
            and id(state) not in used_states
            and camera_event_boxes_match(state["box"], box)
        ]
        if not matches:
            continue
        matched_state = min(
            matches,
            key=lambda state: math.hypot(
                (state["box"][0] + state["box"][2] - box[0] - box[2]) / 2,
                (state["box"][1] + state["box"][3] - box[1] - box[3]) / 2,
            ),
        )
        used_states.add(id(matched_state))
        matched_state["last_seen"] = now
        matched_state["box"] = box


def commit_camera_event_states(state_updates, now, saved_events):
    """Advance cross-session event state only after all corresponding logs commit."""
    for state_update, saved_event in zip(state_updates, saved_events):
        matched_state, camera_scope_id, status, box = state_update
        log_id, _, _, _, track, _ = saved_event
        event_reference = {
            "log_id": log_id,
            "session_id": track.get("session_id"),
            "object_track_id": track["id"],
        }
        if matched_state is None:
            camera_event_states.append({
                "camera_scope_id": camera_scope_id,
                "status": status,
                "box": box,
                "last_seen": now,
                **event_reference,
            })
        else:
            matched_state.update(
                status=status, box=box, last_seen=now, **event_reference
            )
    if len(camera_event_states) > MAX_CAMERA_EVENT_STATES:
        camera_event_states.sort(key=lambda state: state["last_seen"])
        del camera_event_states[:len(camera_event_states) - MAX_CAMERA_EVENT_STATES]


def class_aware_nms(boxes, class_ids, scores, score_threshold, iou_threshold, max_det):
    """Apply NMS within each class so overlapping objects of different classes survive."""
    kept_indices = []
    for class_id in set(class_ids):
        class_indices = [index for index, value in enumerate(class_ids) if value == class_id]
        class_boxes = [boxes[index] for index in class_indices]
        class_scores = [float(scores[index]) for index in class_indices]
        selected = cv2.dnn.NMSBoxes(
            class_boxes, class_scores, score_threshold, iou_threshold
        )
        if selected is not None:
            import numpy as np
            kept_indices.extend(
                class_indices[index] for index in np.asarray(selected).reshape(-1)
            )
    kept_indices.sort(key=lambda index: float(scores[index]), reverse=True)
    return kept_indices[:max_det]


def submit_bounded_background_task(task, task_name):
    """Bound background memory; skip optional delivery work rather than blocking detection."""
    def run_task():
        try:
            task()
        except Exception as exc:
            app.logger.warning("Background %s failed (%s).", task_name, type(exc).__name__)
        finally:
            background_task_slots.release()

    if not background_task_slots.acquire(blocking=False):
        app.logger.warning(
            "Background queue is full; skipped %s. Database detection log is already saved.",
            task_name,
        )
        return False
    try:
        background_executor.submit(run_task)
    except RuntimeError:
        background_task_slots.release()
        app.logger.warning(
            "Background executor unavailable; skipped %s. Database detection log is already saved.",
            task_name,
        )
        return False
    return True


def encode_detection_crop(frame, box, margin_ratio=0.18):
    x1, y1, x2, y2 = [int(value) for value in box]
    frame_height, frame_width = frame.shape[:2]
    margin_x = max(8, int((x2 - x1) * margin_ratio))
    margin_y = max(8, int((y2 - y1) * margin_ratio))
    crop = frame[max(0, y1 - margin_y):min(frame_height, y2 + margin_y),
                 max(0, x1 - margin_x):min(frame_width, x2 + margin_x)]
    if not crop.size:
        return None
    encoded_ok, encoded = cv2.imencode(
        ".jpg", crop, [int(cv2.IMWRITE_JPEG_QUALITY), 88]
    )
    return encoded.tobytes() if encoded_ok else None


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
            same_object_overlap = iou >= 0.92 or intersection_over_min_area >= 0.96
            if same_object_overlap and center_distance <= max(6, min_diagonal * 0.10):
                duplicate = True
                break
        if not duplicate:
            kept.append(detection)
    return kept


def associate_and_stabilize(detections, now, track_state):
    """Track face proposals, stabilize mask labels, and emit event logs once."""
    tracks = track_state["tracks"]
    next_track_id = track_state["next_id"]
    outputs = []
    log_candidates = []
    for track_id in list(tracks):
        if now - tracks[track_id]["last_seen"] > TRACK_EXPIRY_SECONDS:
            del tracks[track_id]

    # Remove only almost identical proposals. Nearby, overlapping people should
    # remain separate detections and receive separate tracks.
    frame_detections = []
    frame_boxes = []
    for detection in sorted(detections, key=lambda item: item["confidence"], reverse=True):
        box = detection["box"]
        area = max(1, (box[2] - box[0]) * (box[3] - box[1]))
        center_x = (box[0] + box[2]) / 2
        center_y = (box[1] + box[3]) / 2
        diagonal = math.hypot(box[2] - box[0], box[3] - box[1])
        duplicate = False
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
            center_distance = math.hypot(center_x - previous_center_x, center_y - previous_center_y)
            near_identical = (
                (box_iou(box, previous_box) >= 0.92
                 or intersection / min(area, previous_area) >= 0.96)
                and center_distance <= max(6, min(diagonal, previous_diagonal) * 0.10)
            )
            if near_identical:
                duplicate = True
                break
        if not duplicate:
            frame_detections.append(detection)
            frame_boxes.append(box)

    # Build all plausible pairs first, then assign the best pairs one-to-one.
    # This avoids a high-confidence detection taking a track that fits another
    # nearby object much better.
    pair_scores = []
    for detection_index, detection in enumerate(frame_detections):
        box = detection["box"]
        center_x = (box[0] + box[2]) / 2
        center_y = (box[1] + box[3]) / 2
        diagonal = math.hypot(box[2] - box[0], box[3] - box[1])
        for track_id, track in tracks.items():
            elapsed = now - track["last_seen"]
            if elapsed > TRACK_EXPIRY_SECONDS:
                continue
            previous = track["box"]
            previous_x = (previous[0] + previous[2]) / 2
            previous_y = (previous[1] + previous[3]) / 2
            prediction_time = min(elapsed, 0.75)
            predicted_x = previous_x + track.get("velocity_x", 0.0) * prediction_time
            predicted_y = previous_y + track.get("velocity_y", 0.0) * prediction_time
            distance = math.hypot(center_x - predicted_x, center_y - predicted_y)
            reference_diagonal = min(diagonal, track["diagonal"])
            max_distance = max(24, reference_diagonal * 0.50 + min(elapsed, 1.5) * 24)
            overlap = box_iou(box, previous)
            if overlap < 0.05 and distance > max_distance:
                continue
            current_area = max(1, (box[2] - box[0]) * (box[3] - box[1]))
            previous_area = max(1, (previous[2] - previous[0]) * (previous[3] - previous[1]))
            area_similarity = min(current_area, previous_area) / max(current_area, previous_area)
            score = (
                overlap * 0.60
                + max(0.0, 1.0 - distance / max_distance) * 0.30
                + area_similarity * 0.10
            )
            pair_scores.append((score, distance, detection_index, track_id))

    assignments = {}
    matched_track_ids = set()
    for _, _, detection_index, track_id in sorted(
            pair_scores, key=lambda pair: (-pair[0], pair[1], pair[2], pair[3])):
        if detection_index in assignments or track_id in matched_track_ids:
            continue
        assignments[detection_index] = track_id
        matched_track_ids.add(track_id)

    for detection_index, detection in enumerate(frame_detections):
        box = detection["box"]
        diagonal = math.hypot(box[2] - box[0], box[3] - box[1])
        track_id = assignments.get(detection_index)
        best_track = tracks.get(track_id) if track_id is not None else None
        if best_track is None:
            if len(tracks) >= MAX_TRACKS_PER_SESSION:
                # Keep unmatched tracks through the occlusion grace period.
                # New detections stay visible but do not evict a briefly hidden
                # object or create an unstable temporary identity.
                outputs.append({
                    "box": box,
                    "label": detection["label"],
                    "confidence": detection["confidence"],
                    "track_id": None,
                    "tracking_id": None,
                    "label_stable": False,
                    "tracking_status": "capacity_full",
                })
                continue
            track_id = next_track_id
            next_track_id += 1
            best_track = {
                "id": track_id,
                "box": box,
                "diagonal": diagonal,
                "last_seen": now,
                "velocity_x": 0.0,
                "velocity_y": 0.0,
                "observations": [],
                "candidate_label": None,
                "candidate_since": now,
                "stable_label": None,
                "stable_confidence": 0.0,
                "last_class_seen": 0.0,
                "logged_label": None,
                "pending_label": None,
                "latest_log_id": None,
                "event_log_status": None,
                "duplicate_event": None,
            }
            tracks[track_id] = best_track
        else:
            elapsed = max(1e-3, now - best_track["last_seen"])
            old_x = (best_track["box"][0] + best_track["box"][2]) / 2
            old_y = (best_track["box"][1] + best_track["box"][3]) / 2
            current_x = (box[0] + box[2]) / 2
            current_y = (box[1] + box[3]) / 2
            speed_limit = max(30.0, best_track["diagonal"] * 0.8)
            observed_vx = max(-speed_limit, min(speed_limit, (current_x - old_x) / elapsed))
            observed_vy = max(-speed_limit, min(speed_limit, (current_y - old_y) / elapsed))
            best_track["velocity_x"] = (
                best_track.get("velocity_x", 0.0) * 0.65 + observed_vx * 0.35
            )
            best_track["velocity_y"] = (
                best_track.get("velocity_y", 0.0) * 0.65 + observed_vy * 0.35
            )
            best_track["box"] = box
            best_track["diagonal"] = diagonal
            best_track["last_seen"] = now

        if detection["confidence"] >= 0.60:
            best_track["last_class_seen"] = now
            best_track["observations"].append((now, detection["label"], detection["confidence"]))
        best_track["observations"] = [
            item for item in best_track["observations"] if now - item[0] <= 3.0
        ]
        if len(best_track["observations"]) > MAX_TRACK_OBSERVATIONS:
            best_track["observations"] = best_track["observations"][-MAX_TRACK_OBSERVATIONS:]

        observations = best_track["observations"]
        if observations:
            scores = {}
            counts = {}
            for observed_at, label, confidence in observations:
                recency_weight = 0.75 + 0.25 * max(0, 1 - (now - observed_at) / 3.0)
                scores[label] = scores.get(label, 0.0) + confidence * recency_weight
                counts[label] = counts.get(label, 0) + 1
            ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
            winner, winner_score = ranked[0]
            total_score = sum(scores.values())
            second_score = ranked[1][1] if len(ranked) > 1 else 0.0
            average_confidence = winner_score / max(1, counts[winner])
            confident_majority = (
                len(observations) >= 2
                and winner_score / max(total_score, 1e-6) >= 0.65
                and (winner_score - second_score) / max(total_score, 1e-6) >= 0.15
                and average_confidence >= 0.55
            )
            if confident_majority and winner != best_track["candidate_label"]:
                best_track["candidate_label"] = winner
                consecutive_winner_times = []
                for observed_at, label, _ in reversed(observations):
                    if label != winner:
                        break
                    consecutive_winner_times.append(observed_at)
                best_track["candidate_since"] = min(consecutive_winner_times, default=now)
            transition_confirmed = (
                now - best_track["candidate_since"] >= 0.16
            )
            if (confident_majority and winner == best_track["candidate_label"]
                    and transition_confirmed):
                best_track["stable_label"] = winner
                best_track["stable_confidence"] = average_confidence
            elif confident_majority and best_track["stable_label"] == winner:
                best_track["stable_confidence"] = average_confidence

        class_is_fresh = now - best_track["last_class_seen"] <= TRACK_EXPIRY_SECONDS
        classification_is_stable = best_track["stable_label"] is not None and class_is_fresh
        # Keep the last confirmed state visible while a possible transition is being verified.
        final_label = (
            best_track["stable_label"] if best_track["stable_label"] is not None and class_is_fresh
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
            "track_id": track_id,
            "tracking_id": track_id,
            "label_stable": classification_is_stable,
            "log_id": best_track.get("latest_log_id"),
            "latest_log_id": best_track.get("latest_log_id"),
            "event_log_status": best_track.get("event_log_status"),
            "duplicate_event": best_track.get("duplicate_event"),
        }
        # Show the box as soon as YOLO detects the object; label confirmation
        # only gates logging, not whether the face is outlined on camera.
        outputs.append(output)

        # One event per track state: first stable label, then only a different
        # stable label. The saved label is advanced only after the DB commit.
        if (classification_is_stable
                and best_track["logged_label"] != best_track["stable_label"]
                and best_track["pending_label"] != best_track["stable_label"]):
            best_track["pending_label"] = best_track["stable_label"]
            output["event_label"] = best_track["stable_label"]
            log_candidates.append((output, best_track))

    track_state["next_id"] = next_track_id
    return outputs, log_candidates


def process_frame_tracks_and_logs(
        candidates, track_state, session_id, camera_scope_id, frame_width,
        frame_height, inference_ms):
    """Serialize event decision, dedupe, commit, and in-memory acknowledgement."""
    with event_commit_lock:
        now = time.monotonic()
        with browser_track_lock:
            track_state["last_seen"] = now
            detections, log_candidates = associate_and_stabilize(
                candidates, now, track_state
            )
            for detection in detections:
                detection["session_id"] = session_id
                track = track_state["tracks"].get(detection["tracking_id"])
                if track is not None:
                    track["session_id"] = session_id
                    detection["log_id"] = track.get("latest_log_id")
                    detection["latest_log_id"] = track.get("latest_log_id")
                    detection["event_log_status"] = track.get("event_log_status")
                    detection["duplicate_event"] = track.get("duplicate_event")

        refresh_camera_event_states(
            detections, camera_scope_id, frame_width, frame_height, now
        )
        selected, suppressed, state_updates = prepare_camera_event_candidates(
            log_candidates, camera_scope_id, frame_width, frame_height, now
        )
        with browser_track_lock:
            for detection, track in suppressed:
                # A matching event in the bounded camera ledger already has a
                # committed database row, so this track can inherit that state.
                track["logged_label"] = detection["event_label"]
                track["pending_label"] = None
                track["event_log_status"] = "duplicate_suppressed"
                track["duplicate_event"] = detection.get("duplicate_event")
                detection["event_log_status"] = "duplicate_suppressed"

        log_ids = []
        saved_events = []
        cloud_error = None
        database_commit_ms = 0.0
        if selected:
            database_started = time.perf_counter()
            try:
                with app.app_context():
                    entries = []
                    for detection, track, canonical_status in selected:
                        entry = DetectionLog(
                            status=canonical_status,
                            confidence=f"{detection['confidence'] * 100:.0f}%",
                            track_id=None,
                            session_id=session_id,
                            object_track_id=detection["tracking_id"],
                            image_url=None,
                            inference_time_ms=inference_ms,
                            roundtrip_time_ms=None,
                            total_latency_ms=None,
                        )
                        db.session.add(entry)
                        entries.append((entry, detection, track, canonical_status))
                    db.session.flush()
                    for entry, _, _, _ in entries:
                        # Keep the legacy column as the database log ID.
                        entry.track_id = entry.id
                    saved_events = [
                        (entry.id, entry.confidence, entry.created_at, detection,
                         track, canonical_status)
                        for entry, detection, track, canonical_status in entries
                    ]
                    db.session.commit()
            except Exception as exc:
                database_commit_ms = (time.perf_counter() - database_started) * 1000
                db.session.rollback()
                saved_events = []
                log_ids = []
                with browser_track_lock:
                    for detection, track, _ in selected:
                        if track["pending_label"] == detection["event_label"]:
                            track["pending_label"] = None
                        track["event_log_status"] = "save_failed"
                        detection["event_log_status"] = "save_failed"
                app.logger.warning(
                    "Could not save stable mask detections (%s).", type(exc).__name__
                )
                cloud_error = "Could not save stable detections to the database."
            else:
                database_commit_ms = (time.perf_counter() - database_started) * 1000
                with browser_track_lock:
                    for log_id, _, _, detection, track, _ in saved_events:
                        track["logged_label"] = detection["event_label"]
                        track["pending_label"] = None
                        track["session_id"] = session_id
                        track["latest_log_id"] = log_id
                        track["event_log_status"] = "saved"
                        track["duplicate_event"] = None
                        detection["log_id"] = log_id
                        detection["latest_log_id"] = log_id
                        detection["event_log_status"] = "saved"
                        detection["duplicate_event"] = None
                        detection["event_log_id"] = log_id
                        log_ids.append(log_id)
                commit_camera_event_states(state_updates, now, saved_events)

        return detections, saved_events, log_ids, cloud_error, database_commit_ms

with app.app_context():
    db.create_all()
    log_columns = {column["name"] for column in inspect(db.engine).get_columns("detection_logs")}
    detection_log_migrations = {
        "track_id": "INTEGER",
        "session_id": "VARCHAR(36)",
        "object_track_id": "INTEGER",
    }
    missing_columns = {
        name: column_type for name, column_type in detection_log_migrations.items()
        if name not in log_columns
    }
    if missing_columns:
        with db.engine.begin() as connection:
            for name, column_type in missing_columns.items():
                connection.execute(text(
                    f"ALTER TABLE detection_logs ADD COLUMN {name} {column_type}"
                ))


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
            app.logger.warning(
                "S3 upload failed for detection %s (%s).", log_id, type(exc).__name__
            )

    submit_bounded_background_task(upload_task, "S3 image upload")


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
    request_started = time.perf_counter()
    request_parse_started = time.perf_counter()
    legacy_json = request.mimetype != "image/jpeg"
    if legacy_json:
        payload = request.get_json(silent=True) or {}
        if not isinstance(payload, dict):
            return jsonify({"error": "Body must be a JSON object"}), 400
        image_data = payload.get("image", "")
        client_session_id = payload.get("session_id")
        camera_scope_id = payload.get("camera_id")
    else:
        payload = None
        image_data = None
        client_session_id = request.args.get("session_id")
        camera_scope_id = request.args.get("camera_id")
    if not isinstance(client_session_id, str) or not client_session_id or len(client_session_id) > 80:
        return jsonify({"error": "A valid webcam session_id is required"}), 400
    try:
        client_session_id = str(uuid.UUID(client_session_id))
    except (ValueError, AttributeError):
        return jsonify({"error": "A valid webcam session_id is required"}), 400
    if camera_scope_id is not None:
        if not isinstance(camera_scope_id, str) or not camera_scope_id or len(camera_scope_id) > 80:
            return jsonify({"error": "A valid webcam camera_id is required"}), 400
        try:
            camera_scope_id = str(uuid.UUID(camera_scope_id))
        except (ValueError, AttributeError):
            return jsonify({"error": "A valid webcam camera_id is required"}), 400
    import base64
    import numpy as np

    payload_bytes = None
    if legacy_json:
        if not isinstance(image_data, str) or not image_data.startswith("data:image/") or "," not in image_data:
            return jsonify({"error": "A valid JPEG frame is required"}), 400
        try:
            payload_bytes = base64.b64decode(image_data.split(",", 1)[1], validate=True)
        except (ValueError, TypeError):
            return jsonify({"error": "Invalid image data"}), 400
    else:
        payload_bytes = request.get_data(cache=False, as_text=False)
    request_parse_ms = (time.perf_counter() - request_parse_started) * 1000
    if not payload_bytes:
        return jsonify({"error": "A valid JPEG frame is required"}), 400

    decode_started = time.perf_counter()
    try:
        frame = cv2.imdecode(np.frombuffer(payload_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
        if frame is None:
            return jsonify({"error": "Could not decode the webcam frame"}), 400
    except (ValueError, TypeError):
        return jsonify({"error": "Invalid image data"}), 400
    decode_ms = (time.perf_counter() - decode_started) * 1000

    with browser_track_lock:
        session_now = time.monotonic()
        for old_session_id in [
            key for key, value in browser_track_sessions.items()
            if session_now - value["last_seen"] > TRACK_SESSION_TTL_SECONDS
        ]:
            del browser_track_sessions[old_session_id]
        track_state = browser_track_sessions.get(client_session_id)
        if track_state is None:
            if len(browser_track_sessions) >= MAX_TRACK_SESSIONS:
                return jsonify({"error": "Máy chủ đang xử lý đủ số phiên webcam; thử lại sau."}), 503
            track_state = {"tracks": {}, "next_id": 1, "last_seen": session_now}
            browser_track_sessions[client_session_id] = track_state
        track_state["last_seen"] = session_now

    frame_height, frame_width = frame.shape[:2]
    results, inference_stages = predict_onnx(
        frame, conf_threshold=0.12, iou_threshold=0.50,
        max_det=MAX_DETECTIONS_PER_FRAME, return_timings=True
    )
    inference_ms = sum(inference_stages.values())

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
    event_processing_started = time.perf_counter()
    detections, saved_events, log_ids, cloud_error, database_commit_ms = process_frame_tracks_and_logs(
        candidates, track_state, client_session_id, camera_scope_id,
        frame_width, frame_height, inference_ms,
    )
    event_processing_ms = (time.perf_counter() - event_processing_started) * 1000

    event_image_started = time.perf_counter()
    full_frame_bytes = None
    if s3_client and saved_events:
        if not background_capacity_available():
            app.logger.warning(
                "S3 queue is full; skipped image uploads. Database detection logs are saved."
            )
        else:
            try:
                encoded_ok, encoded_frame = cv2.imencode(".jpg", frame)
                if encoded_ok:
                    full_frame_bytes = encoded_frame.tobytes()
                else:
                    app.logger.warning("Could not JPEG-encode the event frame for S3.")
            except Exception as exc:
                app.logger.warning("Could not prepare event frame for S3 (%s).", type(exc).__name__)

    for log_id, confidence, created_at, detection, _, canonical_status in saved_events:
        app.logger.info(
            "Stored result=%s log_id=%s session_id=%s track_id=%s confidence=%.3f",
            canonical_status, log_id, client_session_id,
            detection["tracking_id"], detection["confidence"],
        )
        telegram_image = None
        if TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID:
            try:
                telegram_image = encode_detection_crop(frame, detection["box"])
            except Exception as exc:
                app.logger.warning(
                    "Could not crop Telegram image for log %s (%s).",
                    log_id, type(exc).__name__,
                )
            if telegram_image is None:
                app.logger.warning("No image crop available for Telegram log %s.", log_id)
            try:
                send_telegram_detection_async(
                    log_id, canonical_status, confidence,
                    detection["tracking_id"], created_at, telegram_image,
                    session_id=client_session_id,
                )
            except Exception as exc:
                app.logger.warning(
                    "Could not enqueue Telegram alert for log %s (%s).",
                    log_id, type(exc).__name__,
                )
    # Reserve available queue slots for per-object Telegram alerts first;
    # optional S3 copies use only the remaining capacity.
    if full_frame_bytes:
        for log_id, _, _, _, _, canonical_status in saved_events:
            object_name = (
                f"mask_logs/{datetime.now(VN_TZ):%Y%m%d_%H%M%S}_"
                f"{canonical_status}_{uuid.uuid4().hex[:6]}.jpg"
            )
            try:
                upload_detection_image_async(log_id, object_name, full_frame_bytes)
            except Exception as exc:
                app.logger.warning(
                    "Could not enqueue S3 image for log %s (%s).",
                    log_id, type(exc).__name__,
                )
    event_image_prepare_ms = (time.perf_counter() - event_image_started) * 1000

    processing_ms = (time.perf_counter() - request_started) * 1000
    return jsonify({
        "detections": detections,
        "untracked_detection_count": sum(
            1 for detection in detections
            if detection.get("tracking_status") == "capacity_full"
        ),
        "log_ids": log_ids,
        "log_id": log_ids[0] if log_ids else None,
        "inference_time_ms": round(inference_ms, 1),
        "server_processing_ms": round(processing_ms, 1),
        "decode_time_ms": round(decode_ms, 1),
        "request_parse_time_ms": round(request_parse_ms, 1),
        "frame_payload_bytes": len(payload_bytes),
        "event_processing_time_ms": round(event_processing_ms, 1),
        "database_commit_time_ms": round(database_commit_ms, 1),
        "event_image_prepare_time_ms": round(event_image_prepare_ms, 1),
        **{key: round(value, 1) for key, value in inference_stages.items()},
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
        raw_log_ids = payload.get("log_ids")
        if raw_log_ids is None:
            raw_log_ids = [payload.get("log_id")]
        if not isinstance(raw_log_ids, list) or not raw_log_ids or len(raw_log_ids) > MAX_DETECTIONS_PER_FRAME:
            raise ValueError("invalid log IDs")
        log_ids = list(dict.fromkeys(int(value) for value in raw_log_ids))
        if any(log_id <= 0 for log_id in log_ids):
            raise ValueError("invalid log IDs")
        roundtrip_ms = float(payload.get('roundtrip_time_ms'))
        total_ms = float(payload.get('total_latency_ms'))
        if (not math.isfinite(roundtrip_ms) or not math.isfinite(total_ms)
                or min(roundtrip_ms, total_ms) < 0):
            raise ValueError("latency must be non-negative")
    except (TypeError, ValueError):
        return jsonify({"error": "Số đo độ trễ không hợp lệ"}), 400

    try:
        log_entries = (DetectionLog.query
                       .filter(DetectionLog.id.in_(log_ids)).all())
    except Exception as exc:
        db.session.rollback()
        app.logger.warning(
            "Could not read detection log for latency update (%s).", type(exc).__name__
        )
        return jsonify({"error": "Không kết nối được cơ sở dữ liệu để lưu độ trễ"}), 503
    if len(log_entries) != len(log_ids):
        return jsonify({"error": "Không tìm thấy log nhận diện"}), 404
    try:
        for log_entry in log_entries:
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
        app.logger.warning(
            "Could not update client latency (%s).", type(exc).__name__
        )
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
    with event_commit_lock:
        try:
            db.session.query(LatencySample).delete()
            num_rows_deleted = db.session.query(DetectionLog).delete()
            db.session.commit()
            with browser_track_lock:
                browser_track_sessions.clear()
            camera_event_states.clear()
            return jsonify({"status": "success", "message": f"Đã xóa {num_rows_deleted} record trong DB"})
        except Exception as exc:
            db.session.rollback()
            app.logger.warning("Could not clear detection logs (%s).", type(exc).__name__)
            return jsonify({"status": "error", "message": "Could not clear detection logs."}), 500


if __name__ == '__main__':
    host = os.getenv('HOST', '0.0.0.0')
    port = int(os.getenv('PORT', 5000))
    app.run(host=host, port=port, debug=True)
