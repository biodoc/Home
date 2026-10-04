"""Phase 1 POC: run the pretrained detector over sample clips and report.

Usage:
    python scripts/poc_eval.py --config config/default.yaml --footage data/footage

Writes output/poc_detections.csv (one row per detection per sampled frame)
and output/frames/ (annotated frames with hits) for manual review.
"""
import argparse
import csv
from collections import Counter
from pathlib import Path

import cv2

from monitor.config import load_config
from monitor.detect.yolo import YoloDetector

VIDEO_EXT = {".mp4", ".avi", ".mkv", ".mov", ".ts"}


def iter_sampled_frames(path: Path, sample_fps: float):
    cap = cv2.VideoCapture(str(path))
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    step = max(1, round(src_fps / sample_fps))
    i = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if i % step == 0:
            yield i / src_fps, frame
        i += 1
    cap.release()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/default.yaml")
    ap.add_argument("--footage", default="data/footage")
    ap.add_argument("--out", default="output")
    args = ap.parse_args()

    cfg = load_config(args.config)
    detectors = [YoloDetector(n, m, cfg.label_map(n)) for n, m in cfg.models.items()]

    out = Path(args.out)
    (out / "frames").mkdir(parents=True, exist_ok=True)
    clips = sorted(p for p in Path(args.footage).iterdir() if p.suffix.lower() in VIDEO_EXT)
    if not clips:
        raise SystemExit(f"No video files in {args.footage}")

    counts: Counter[str] = Counter()
    frames_seen = frames_hit = 0
    with open(out / "poc_detections.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["clip", "t_sec", "model", "class", "label", "confidence",
                    "bbox_x", "bbox_y", "bbox_w", "bbox_h"])
        for clip in clips:
            print(f"{clip.name} ...")
            for t, frame in iter_sampled_frames(clip, cfg.sample_fps):
                frames_seen += 1
                hits = [(d.name, det) for d in detectors for det in d.detect(frame)]
                if not hits:
                    continue
                frames_hit += 1
                for model, det in hits:
                    counts[det.cls] += 1
                    w.writerow([clip.name, f"{t:.2f}", model, det.cls, det.label,
                                f"{det.confidence:.3f}", *(f"{v:.0f}" for v in det.bbox)])
                    x, y, bw, bh = map(int, det.bbox)
                    cv2.rectangle(frame, (x, y), (x + bw, y + bh), (0, 0, 255), 2)
                    cv2.putText(frame, f"{det.cls} {det.confidence:.2f}", (x, y - 6),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                cv2.imwrite(str(out / "frames" / f"{clip.stem}_{t:08.2f}.jpg"), frame)

    print(f"\nFrames sampled: {frames_seen}   frames with detections: {frames_hit}")
    for cls, n in counts.most_common():
        print(f"  {cls:10s} {n}")
    print(f"\nReview annotated frames in {out / 'frames'} to tally hits, misses, false positives.")


if __name__ == "__main__":
    main()
