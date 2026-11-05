# Hệ thống nhận diện khẩu trang qua webcam

Ứng dụng dùng webcam trong trình duyệt để nhận diện trạng thái khẩu trang. Trình duyệt gửi từng khung hình đến Flask; server chạy model bằng **ONNX Runtime**, trả về nhãn, độ tin cậy và tọa độ khung. Các sự kiện nhận diện ổn định và số liệu độ trễ được lưu vào PostgreSQL/Supabase. Lưu ảnh lên AWS S3 là tùy chọn.

## Luồng hoạt động

1. Người dùng mở trang bằng HTTPS và cấp quyền webcam.
2. Trình duyệt chụp và gửi ảnh JPEG đến `POST /api/detect`.
3. Flask tiền xử lý ảnh, chạy `mask_model.onnx` và trả về kết quả nhận diện.
4. Giao diện vẽ khung và nhãn lên video.
5. Khi trạng thái được xác nhận ổn định và thay đổi, ứng dụng ghi log vào PostgreSQL/Supabase.

Các lớp của model là `wrong mask`, `mask`, `no mask`. Model hiện dùng các khung nhận diện làm đề xuất khuôn mặt; project chưa cài một face detector riêng.

## Model và dữ liệu huấn luyện

- `mask_model.pt`: checkpoint dùng để export hoặc huấn luyện tiếp.
- `mask_model.onnx`: model dùng khi chạy Flask; inference không cần PyTorch.
- `export_onnx.py` và `colab_export_onnx.ipynb`: công cụ export model trên Google Colab.
- `requirements-export.txt`: thư viện chỉ cần cho bước export.

> Repo không chứa dataset, file cấu hình huấn luyện hoặc lịch sử train. Vì vậy cần kiểm tra nguồn dữ liệu, cách gán nhãn, số lượng ảnh, cách chia train/validation/test và chỉ số đánh giá từ nhóm trước khi báo cáo các thông tin đó.

Model ONNX hiện có trong repo nhận đầu vào **640×640** để giảm thời gian suy luận và bộ nhớ so với bản 960×960 trước đây. Notebook và script export cũng dùng kích thước này mặc định.

## Export ONNX trên Google Colab

1. Mở `colab_export_onnx.ipynb` bằng Google Colab.
2. Chạy các cell và tải `mask_model.pt` lên khi notebook yêu cầu.
3. Notebook export `mask_model.onnx` với kích thước 640×640 và tải file về.
4. Thay `mask_model.onnx` trong project bằng file vừa tải.

PyTorch/Ultralytics chỉ dùng cho bước export. Server Flask chỉ dùng ONNX Runtime.

Có thể export bằng dòng lệnh trong môi trường đã cài `requirements-export.txt`:

```bash
python export_onnx.py mask_model.pt --imgsz 640
```

## Cài đặt và chạy ứng dụng

```bash
pip install -r requirements.txt
python app_web.py
```

Tạo `.env` cục bộ với `DATABASE_URL` trỏ đến PostgreSQL/Supabase. Có thể đặt thêm `HTML_TEMPLATE` (ví dụ `index_chat.html`) và `MODEL_PATH` nếu model không ở đường dẫn mặc định `mask_model.onnx`.

Với giao diện webcam laptop, đặt `HTML_TEMPLATE=index_chat.html`. Giao diện chỉ gửi một frame tại một thời điểm và chụp theo đúng kích thước video webcam đang hoạt động (mặc định yêu cầu 1280×720, thiết bị có thể cấp kích thước khác). JPEG được gửi ở chất lượng 0.65; server giải mã frame gốc rồi letterbox vào tensor 640×640 mà model yêu cầu. Frame lớn làm tăng dung lượng truyền và bộ nhớ tạm; `MAX_FRAME_REQUEST_MB` mặc định là 8 MB và request vượt giới hạn nhận HTTP 413. Giao diện báo lỗi này và thử lại sau 5 giây, không tự thu nhỏ frame. Chưa đo tác động RAM/độ trễ trên Render; cần theo dõi request size, memory peak và các stage latency sau deploy.

Track được ghép một-một theo vị trí, kích thước, chuyển động ước lượng và độ chồng lấn của bounding box; nhãn không được dùng để ghép track. Track chưa khớp được giữ tới hết thời gian mất dấu 3 giây, không bị đẩy ra chỉ để dành chỗ cho detection mới. Cấu hình mặc định giới hạn 4 phiên, 64 track mỗi phiên và 6 quan sát nhãn mỗi track. Mỗi frame vẫn trả về mọi detection (tối đa 100 sau NMS). Nếu đủ 64 track, detection mới vẫn được vẽ nhưng trả `track_id=null`, có `tracking_status=capacity_full`, và chưa được xác nhận/log cho tới khi slot trống. API trả `untracked_detection_count`; giao diện hiển thị số đối tượng đang thiếu track.

Mỗi lần bật webcam có `session_id` mới. Để tránh ghi lại cùng trạng thái khi track bị cấp ID mới hoặc khi kết nối lại/tải lại trang, giao diện giữ một `camera_id` ngẫu nhiên trong `sessionStorage` của tab. Server dùng một vùng nhớ giới hạn (tối đa `MAX_CAMERA_EVENT_STATES`, mặc định 256 trạng thái, hết hạn sau `CAMERA_EVENT_DEDUPE_TTL_SECONDS`, mặc định 300 giây không được nhìn thấy) để so trạng thái và vị trí bounding box đã chuẩn hóa. Cùng trạng thái tại cùng vùng được xem là sự kiện đang tiếp diễn; trạng thái đổi thì ghi log mới. Các detection khác nhau trong cùng frame được đối sánh một-một nên vẫn có thể ghi riêng kể cả cùng nhãn. Đây chỉ là chống lặp theo vị trí/thời gian, không nhận diện danh tính; người khác xuất hiện gần cùng vị trí trong cửa sổ 5 phút có thể bị xem là sự kiện tiếp diễn. Bộ nhớ này mất khi process Render khởi động lại, nên sau restart một trạng thái đang diễn ra có thể tạo log mới.

Ảnh Telegram được crop riêng quanh từng đối tượng trong sự kiện. Tác vụ Telegram/S3 dùng mặc định 1 worker và tối đa 8 tác vụ đang chờ/chạy. Khi hàng đợi đầy, tác vụ phụ có thể bị bỏ và ghi cảnh báo; log nhận diện đã commit trong database vẫn được giữ. S3 chỉ được khởi tạo khi `AWS_S3_BUCKET_NAME` được cấu hình và lưu frame gốc cho mỗi sự kiện.

Số luồng ONNX mặc định là 1 (`ONNX_INTRA_OP_NUM_THREADS`, có thể chọn 1 hoặc 2). Mặc định này ưu tiên giới hạn tài nguyên cho instance RAM thấp. Chưa đo RSS hoặc tốc độ trên Render; benchmark trên máy phát triển không được dùng để kết luận cấu hình Render. Sau deploy, chạy webcam với cùng một cảnh, chờ model ấm lên, ghi Memory peak và inference latency trong Dashboard; redeploy lần lượt với 1 rồi 2 luồng, sau đó chọn cấu hình nhanh nhất có peak RAM dưới khoảng 430 MB. Render environment variables cần cập nhật trước mỗi lần deploy/khởi động lại.

Để nhận cảnh báo kèm ảnh trên Telegram, cấu hình cả `TELEGRAM_BOT_TOKEN` và `TELEGRAM_CHAT_ID` trong `.env` hoặc Secrets của hosting. Ứng dụng gửi ảnh crop quanh đối tượng cùng caption; nếu không có cấu hình Telegram thì cảnh báo Telegram bị tắt. Lỗi Telegram được ghi log mà không làm gián đoạn lưu log nhận diện. Không ghi token vào log hoặc commit token vào repository.

Để lưu ảnh nhận diện lên S3, cấu hình `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `AWS_REGION` và `AWS_S3_BUCKET_NAME`. Nếu không cấu hình bucket, ứng dụng không khởi tạo AWS SDK.

**Không commit `.env` hoặc giá trị bí mật lên GitHub.** Dùng biến môi trường/Secrets trên dịch vụ hosting.

## Deploy trên Render

Repository được deploy dưới dạng Flask web service. Cấu hình các biến môi trường/Secrets trên Render, tối thiểu cần `DATABASE_URL`; đặt `HTML_TEMPLATE` và `MODEL_PATH` nếu cần.

Lệnh chạy Gunicorn dùng trên Render:

```bash
gunicorn --workers 1 --threads 1 --timeout 120 --bind 0.0.0.0:$PORT app_web:app
```

Trên Render 528 MB, dùng một worker và một thread như lệnh trên hoặc Dockerfile. Giữ `ONNX_INTRA_OP_NUM_THREADS=1` ban đầu; sau đó đo với giá trị 2 nếu muốn so tốc độ. Trình duyệt chỉ gửi một frame tại một thời điểm cho mỗi phiên và không thu nhỏ ảnh webcam trước khi gửi. Kích thước tensor đầu vào model vẫn là 640×640; preprocess phía server letterbox ảnh nguồn vào kích thước đó. So sánh memory peak và RTT sau deploy vì ảnh webcam gốc lớn hơn làm tăng dữ liệu truyền và bộ nhớ tạm.

## Lưu trữ dữ liệu

- **GitHub:** mã nguồn và model artifact; không lưu `.env`.
- **Render:** chạy Flask và ONNX Runtime.
- **PostgreSQL/Supabase:** lưu bảng `detection_logs` và `latency_samples`.
- **AWS S3 (tùy chọn):** lưu ảnh sự kiện nhận diện khi đã cấu hình bucket.

Ứng dụng chỉ ghi log sau khi trạng thái được xác nhận ổn định. `log_id` là ID bản ghi database tăng liên tục; `object_track_id` là ID đối tượng trong `session_id` và có thể bắt đầu lại từ 1 trong phiên mới. API detection trả riêng `session_id`, `track_id`, `log_id`/`latest_log_id`; log ID chỉ được gắn vào track sau khi database commit thành công. Overlay hiển thị `Track <session>:<id> · Log <id>`; Detection Logs dùng cùng cặp thông tin. Khi trùng sự kiện qua phiên, giao diện ghi rõ log/track tham chiếu thuộc phiên trước, không nhận nó là log của track mới. Khi lưu DB lỗi, không cấp log ID mới và overlay báo lỗi. Các ID này không phải định danh sinh trắc học.

## Công thức độ trễ

Tất cả số đo tính bằng mili giây (ms):

- **Inference latency:** thời gian Flask chạy `predict_onnx()`, gồm tiền xử lý, ONNX Runtime và NMS.

  `T_infer = t_infer_end - t_infer_start`

- **RTT:** thời gian từ lúc trình duyệt gửi request đến lúc nhận response; gồm truyền dữ liệu lên/xuống và xử lý phía server.

  `T_RTT = t_response_received - t_request_sent`

- **Total latency hiện tại:** thời gian từ lúc chụp frame đến lúc nhận response; gồm mã hóa ảnh, RTT và xử lý request. Phép đo hiện dừng ở lúc nhận response, chưa tính thời điểm khung được vẽ xong.

  `T_total = t_response_received - t_frame_captured`

Dashboard giữ tối đa 50 mẫu độ trễ gần nhất. `GET /api/latency` đọc mẫu; `POST /api/latency/record` lưu số đo liên quan đến log nhận diện. Với database cũ, chạy `python migrate_latency.py` một lần để thêm các cột độ trễ cần thiết.

API `POST /api/detect` trả thêm `decode_time_ms`, `preprocess_time_ms`, `onnx_time_ms`, `postprocess_time_ms` và `server_processing_ms` để xác định đoạn gây chậm. `inference_time_ms` là tổng tiền xử lý, chạy model và hậu xử lý. Mỗi frame gửi kèm `session_id` riêng cho phiên webcam; nhãn ổn định được ghi một lần cho mỗi trạng thái của track; các đối tượng khác nhau trong cùng frame được ghi riêng. Các giới hạn có thể chỉnh trong `.env.example`: `MAX_FRAME_REQUEST_MB`, `MAX_DETECTIONS_PER_FRAME`, `MAX_TRACK_SESSIONS`, `MAX_TRACKS_PER_SESSION`, `MAX_TRACK_OBSERVATIONS`, `TRACK_SESSION_TTL_SECONDS`, `TRACK_EXPIRY_SECONDS`, `CAMERA_EVENT_DEDUPE_TTL_SECONDS`, `MAX_CAMERA_EVENT_STATES`, `BACKGROUND_WORKERS` và `BACKGROUND_MAX_PENDING_TASKS`.
