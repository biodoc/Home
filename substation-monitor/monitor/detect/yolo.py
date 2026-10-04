import numpy as np

from monitor.config import ModelConfig
from monitor.types import Detection


class YoloDetector:
    """Ultralytics YOLO wrapper. Drops labels not mapped to a monitor class."""

    def __init__(self, name: str, cfg: ModelConfig, label_map: dict[str, str]):
        from ultralytics import YOLO  # optional dependency: pip install .[detect]

        self.name = name
        self.cfg = cfg
        self.label_map = label_map
        self.model = YOLO(cfg.weights)

    def detect(self, frame: np.ndarray) -> list[Detection]:
        result = self.model.predict(
            frame, conf=self.cfg.confidence, imgsz=self.cfg.imgsz, verbose=False
        )[0]
        out = []
        for box in result.boxes:
            label = result.names[int(box.cls)]
            cls = self.label_map.get(label)
            if cls is None:
                continue
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            out.append(Detection(cls, label, float(box.conf), (x1, y1, x2 - x1, y2 - y1)))
        return out
