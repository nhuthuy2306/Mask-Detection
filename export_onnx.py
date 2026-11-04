"""Export the trained YOLO checkpoint to ONNX (run in Google Colab)."""
import argparse
from ultralytics import YOLO


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", nargs="?", default="mask_model.pt")
    parser.add_argument("--imgsz", type=int, default=640)
    args = parser.parse_args()

    model = YOLO(args.checkpoint)
    path = model.export(format="onnx", imgsz=args.imgsz, opset=12, simplify=True,
                        dynamic=False, batch=1)
    print(f"ONNX model exported to: {path}")


if __name__ == "__main__":
    main()
