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

Model ONNX hiện có trong repo nhận đầu vào **960×960**. Notebook và script export mặc định dùng **640×640** để giảm tính toán và bộ nhớ; sau khi export, cần thay `mask_model.onnx` trong repo rồi deploy lại để áp dụng.

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

Để lưu ảnh nhận diện lên S3, cấu hình `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `AWS_REGION` và `AWS_S3_BUCKET_NAME`. Nếu không cấu hình bucket, ứng dụng không khởi tạo AWS SDK.

**Không commit `.env` hoặc giá trị bí mật lên GitHub.** Dùng biến môi trường/Secrets trên dịch vụ hosting.

## Deploy trên Render

Repository được deploy dưới dạng Flask web service. Cấu hình các biến môi trường/Secrets trên Render, tối thiểu cần `DATABASE_URL`; đặt `HTML_TEMPLATE` và `MODEL_PATH` nếu cần.

Lệnh chạy Gunicorn dùng trên Render:

```bash
gunicorn --workers 1 --threads 2 --timeout 120 --bind 0.0.0.0:$PORT app_web:app
```

Render Free có giới hạn CPU/RAM; model 960×960 có thể inference chậm hoặc dùng nhiều bộ nhớ. Export lại model 640×640 giúp giảm tải, nhưng tốc độ còn phụ thuộc cấu hình dịch vụ.

## Lưu trữ dữ liệu

- **GitHub:** mã nguồn và model artifact; không lưu `.env`.
- **Render:** chạy Flask và ONNX Runtime.
- **PostgreSQL/Supabase:** lưu bảng `detection_logs` và `latency_samples`.
- **AWS S3 (tùy chọn):** lưu ảnh sự kiện nhận diện khi đã cấu hình bucket.

Ứng dụng chỉ ghi log sau khi trạng thái được xác nhận ổn định. ID track là số thứ tự trong database, không phải định danh sinh trắc học.

## Công thức độ trễ

Tất cả số đo tính bằng mili giây (ms):

- **Inference latency:** thời gian Flask chạy `predict_onnx()`, gồm tiền xử lý, ONNX Runtime và NMS.

  `T_infer = t_infer_end - t_infer_start`

- **RTT:** thời gian từ lúc trình duyệt gửi request đến lúc nhận response; gồm truyền dữ liệu lên/xuống và xử lý phía server.

  `T_RTT = t_response_received - t_request_sent`

- **Total latency hiện tại:** thời gian từ lúc chụp frame đến lúc nhận response; gồm mã hóa ảnh, RTT và xử lý request. Phép đo hiện dừng ở lúc nhận response, chưa tính thời điểm khung được vẽ xong.

  `T_total = t_response_received - t_frame_captured`

Dashboard giữ tối đa 50 mẫu độ trễ gần nhất. `GET /api/latency` đọc mẫu; `POST /api/latency/record` lưu số đo liên quan đến log nhận diện. Với database cũ, chạy `python migrate_latency.py` một lần để thêm các cột độ trễ cần thiết.
