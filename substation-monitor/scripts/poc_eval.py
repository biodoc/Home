"""Phase 1 POC: run the pretrained detector over sample clips and report.

Usage:
    python scripts/poc_eval.py --config config/default.yaml --footage data/footage

Writes:
  output/poc_detections.csv  one row per detection per sampled frame
  output/poc_summary.csv     one row per clip: frames sampled/hit, max confidence per class
  output/frames/             annotated frames with hits, for manual review

Name clips by what is actually in them (e.g. squirrel_day_01.mp4,
empty_night_02.mp4) so the per-clip summary shows misses and false positives
at a glance.

Compare model sizes with --weights (e.g. yolo11n.pt vs yolo11m.pt); small
animals usually need more than the nano model.
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
    ap.add_argument("--weights", help="override weights for every model, e.g. yolo11m.pt")
    ap.add_argument("--conf", type=float, help="override confidence threshold")
    args = ap.parse_args()

    cfg = load_config(args.config)
    for m in cfg.models.values():
        if args.weights:
            m.weights = args.weights
        if args.conf is not None:
            m.confidence = args.conf
    detectors = [YoloDetector(n, m, cfg.label_map(n)) for n, m in cfg.models.items()]

    out = Path(args.out)
    (out / "frames").mkdir(parents=True, exist_ok=True)
    clips = sorted(p for p in Path(args.footage).iterdir() if p.suffix.lower() in VIDEO_EXT)
    if not clips:
        raise SystemExit(f"No video files in {args.footage}")

    class_names = list(cfg.classes)
    counts: Counter[str] = Counter()
    frames_seen = frames_hit = 0
    with open(out / "poc_detections.csv", "w", newline="") as f, \
         open(out / "poc_summary.csv", "w", newline="") as fs:
        w = csv.writer(f)
        w.writerow(["clip", "t_sec", "model", "class", "label", "confidence",
                    "bbox_x", "bbox_y", "bbox_w", "bbox_h"])
        ws = csv.writer(fs)
        ws.writerow(["clip", "frames_sampled", "frames_hit",
                     *(f"max_conf_{c}" for c in class_names)])
        for clip in clips:
            print(f"{clip.name} ...")
            clip_seen = clip_hit = 0
            max_conf: dict[str, float] = {}
            for t, frame in iter_sampled_frames(clip, cfg.sample_fps):
                clip_seen += 1
                hits = [(d.name, det) for d in detectors for det in d.detect(frame)]
                if not hits:
                    continue
                clip_hit += 1
                for model, det in hits:
                    counts[det.cls] += 1
                    max_conf[det.cls] = max(max_conf.get(det.cls, 0.0), det.confidence)
                    w.writerow([clip.name, f"{t:.2f}", model, det.cls, det.label,
                                f"{det.confidence:.3f}", *(f"{v:.0f}" for v in det.bbox)])
                    x, y, bw, bh = map(int, det.bbox)
                    cv2.rectangle(frame, (x, y), (x + bw, y + bh), (0, 0, 255), 2)
                    cv2.putText(frame, f"{det.cls} {det.confidence:.2f}", (x, y - 6),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                cv2.imwrite(str(out / "frames" / f"{clip.stem}_{t:08.2f}.jpg"), frame)
            frames_seen += clip_seen
            frames_hit += clip_hit
            ws.writerow([clip.name, clip_seen, clip_hit,
                         *(f"{max_conf[c]:.2f}" if c in max_conf else "" for c in class_names)])

    print(f"\nFrames sampled: {frames_seen}   frames with detections: {frames_hit}")
    for cls, n in counts.most_common():
        print(f"  {cls:10s} {n}")
    print(f"\nReview annotated frames in {out / 'frames'} to tally hits, misses, false positives.")


if __name__ == "__main__":
    main()
