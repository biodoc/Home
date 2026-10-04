from dataclasses import dataclass


@dataclass
class Detection:
    """One detection in one frame, already mapped to a monitor class."""

    cls: str                    # monitor class, e.g. "squirrel"
    label: str                  # raw model label, e.g. "cat"
    confidence: float
    bbox: tuple[float, float, float, float]  # x, y, w, h in pixels
    track_id: int | None = None

    @property
    def centroid(self) -> tuple[float, float]:
        x, y, w, h = self.bbox
        return x + w / 2, y + h / 2
