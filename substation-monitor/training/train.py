"""Fine-tune a YOLO model on a YOLO-format dataset. (Stub — filled in Phase 1.)

    python training/train.py --data datasets/wildlife/data.yaml --weights yolo11n.pt
"""
import argparse


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="YOLO dataset yaml")
    ap.add_argument("--weights", default="yolo11n.pt")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--imgsz", type=int, default=1280)
    args = ap.parse_args()

    from ultralytics import YOLO

    YOLO(args.weights).train(data=args.data, epochs=args.epochs, imgsz=args.imgsz)


if __name__ == "__main__":
    main()
