# Webcam mask detection: Flask + ONNX

The Flask server receives webcam frames from the browser and runs the mask detector with ONNX Runtime. It writes detection and latency records to PostgreSQL/Supabase; uploading images to AWS S3 is optional.

## Export the model in Google Colab

1. Open `colab_export_onnx.ipynb` in Google Colab.
2. Run the cells and upload `mask_model.pt` when prompted. PyTorch and Ultralytics are used in Colab only to convert the checkpoint.
3. Download the resulting `mask_model.onnx` and place it beside `app_web.py` (or set `MODEL_PATH` to its location).

The exported model uses a fixed 640×640 input to reduce memory use and retains the three classes in this order: `wrong mask`, `mask`, `no mask`.

## Run the Flask app

```bash
pip install -r requirements.txt
python app_web.py
```

Set `DATABASE_URL` and `HTML_TEMPLATE` in the environment. Set `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `AWS_REGION`, and `AWS_S3_BUCKET_NAME` only if image uploads to S3 are needed. Do not commit credentials. The app loads `mask_model.onnx` by default; `MODEL_PATH` can override it.

For deployment from Colab, upload the project files and `mask_model.onnx` to the Colab runtime, install `requirements.txt`, set the environment variables in the notebook, and run `python app_web.py`. Use a secure HTTPS tunnel so the laptop browser can access the webcam page; Colab sessions and their public tunnel URLs are temporary. Keep database and AWS credentials in Colab secrets or enter them at runtime, never in a notebook committed to the repository.

## Persistent HTTPS deployment with Google Cloud Run

Colab is temporary. For a stable HTTPS URL that remains available after the Colab session ends, deploy the Flask app to Cloud Run. The service can scale to zero when idle, so the first request after an idle period may take longer. Keeping an instance warm can incur charges. Cloud Run has a monthly free tier, with usage billed beyond that tier.

Before deploying, put `mask_model.onnx` in this project directory and make sure `templates/` and `requirements.txt` are present. Do not include `.env`; the Docker build excludes it. Create a Secret Manager secret named `DATABASE_URL` with the PostgreSQL connection string, and grant the Cloud Run service identity permission to access that secret.

With Google Cloud CLI installed and authenticated, deploy from this directory:

```bash
gcloud config set project YOUR_GCP_PROJECT_ID
gcloud run deploy mask-detection \
  --source . \
  --region asia-southeast1 \
  --allow-unauthenticated \
  --memory 2Gi \
  --cpu 2 \
  --timeout 120 \
  --set-env-vars HTML_TEMPLATE=index_chat.html,MODEL_PATH=/app/mask_model.onnx \
  --set-secrets DATABASE_URL=DATABASE_URL:latest
```

Cloud Run prints the stable HTTPS service URL after deployment. If S3 uploads are enabled, add the AWS values through Cloud Run's environment and secrets settings; do not put secret values in the command or source files.

To repeat the checkpoint conversion outside the notebook, install `requirements-export.txt` in a conversion environment and run:

```bash
python export_onnx.py mask_model.pt --imgsz 640
```

## Latency measurements

- Inference: `T_infer_end - T_infer_start`, measured around ONNX Runtime inference in Flask.
- RTT: `T_response_received - T_request_sent`, measured around the browser frame request.
- End to end: `T_result_rendered - T_frame_captured`.

The dashboard keeps the latest 50 webcam samples. `GET /api/latency` returns samples from `latency_samples`; `POST /api/latency/record` stores measurements and updates the related detection log. Run `python migrate_latency.py` once to add latency columns to an existing database; the app creates `latency_samples` on startup.
