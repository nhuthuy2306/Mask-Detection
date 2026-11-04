# Lab 3: Webcam laptop, YOLO và Cloud

Ứng dụng dùng webcam laptop trong trình duyệt, gửi frame JPEG tới Flask để chạy `mask_model.pt`, sau đó vẽ các hộp nhận diện lên giao diện. Khi nhãn có confidence từ 50% thay đổi, sự kiện được ghi vào bảng `detection_logs` trong PostgreSQL/Supabase; lưu ảnh lên AWS S3 là tùy chọn.

## Cấu hình và chạy

1. Tạo môi trường Python và cài dependencies:

   ```powershell
   pip install -r requirements.txt
   ```

2. Điền `.env` với `HTML_TEMPLATE=index_chat.html` và `DATABASE_URL`. Cấu hình `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `AWS_REGION` và `AWS_S3_BUCKET_NAME` nếu cần lưu ảnh lên S3. Không đưa thông tin xác thực thật vào mã nguồn hoặc repository.
3. Đảm bảo `mask_model.pt` nằm cạnh `app_web.py`. Chạy migration một lần để thêm các cột latency vào bảng cũ; bảng `latency_samples` được Flask tạo tự động khi khởi động:

   ```powershell
   python migrate_latency.py
   ```

4. Khởi động ứng dụng:

   ```powershell
   python app_web.py
   ```

5. Mở `http://localhost:5000`, chọn **Bật webcam laptop** và cấp quyền camera.

## Đo độ trễ

- Suy luận: `T_infer_end - T_infer_start`, đo quanh `model.predict()` tại server Flask.
- RTT: `T_response_received - T_request_sent`, đo quanh yêu cầu frame từ trình duyệt và phản hồi Flask.
- Đầu-cuối: `T_result_rendered - T_frame_captured`, từ lúc chụp frame ở trình duyệt đến lúc vẽ kết quả.

The dashboard keeps up to 50 recent webcam samples for latency statistics. `GET /api/latency` returns these samples from `latency_samples`; `POST /api/latency/record` stores measurements and updates the related detection log.

## Dùng webcam khi triển khai cloud

Trình duyệt chỉ cho phép `getUserMedia()` trên HTTPS hoặc `localhost`. Khi truy cập ứng dụng từ laptop qua Internet, cần triển khai Flask sau HTTPS và cấu hình biến môi trường AWS cùng PostgreSQL/Supabase trên máy chủ. Model YOLO chạy tại nơi tiến trình Flask đang chạy; khi Flask được triển khai trên cloud thì inference cũng chạy trên cloud. Việc cấu hình mã nguồn không tự triển khai ứng dụng hoặc tạo bucket/cloud database.
