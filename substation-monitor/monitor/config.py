from dataclasses import dataclass, field
from pathlib import Path

import yaml


@dataclass
class ModelConfig:
    weights: str
    confidence: float = 0.25
    imgsz: int = 640


@dataclass
class ClassConfig:
    model: str
    labels: list[str]


@dataclass
class CameraConfig:
    id: str
    source: str


@dataclass
class SecurityConfig:
    classes: list[str] = field(default_factory=lambda: ["person"])
    off_hours_start: str = "19:00"
    off_hours_end: str = "06:00"
    loiter_sec: float = 30.0
    perimeter: dict[str, list[list[float]]] = field(default_factory=dict)


@dataclass
class Config:
    site: str
    cameras: list[CameraConfig]
    models: dict[str, ModelConfig]
    classes: dict[str, ClassConfig]
    sample_fps: float = 2.0
    clip_retention_days: int = 60
    security: SecurityConfig = field(default_factory=SecurityConfig)
    root: Path = field(default_factory=Path.cwd)

    def label_map(self, model: str) -> dict[str, str]:
        """Raw model label -> monitor class, for one model."""
        return {
            label: name
            for name, c in self.classes.items()
            if c.model == model
            for label in c.labels
        }


def load_config(path: str | Path) -> Config:
    path = Path(path)
    raw = yaml.safe_load(path.read_text())

    models = {k: ModelConfig(**v) for k, v in raw["models"].items()}
    classes = {k: ClassConfig(**v) for k, v in raw["classes"].items()}
    for name, c in classes.items():
        if c.model not in models:
            raise ValueError(f"class '{name}' references unknown model '{c.model}'")

    sec = raw.get("security") or {}
    security = SecurityConfig(
        classes=sec.get("classes", ["person"]),
        off_hours_start=sec.get("off_hours", {}).get("start", "19:00"),
        off_hours_end=sec.get("off_hours", {}).get("end", "06:00"),
        loiter_sec=sec.get("loiter_sec", 30.0),
        perimeter=sec.get("perimeter") or {},
    )
    for name in security.classes:
        if name not in classes:
            raise ValueError(f"security class '{name}' is not a defined class")

    return Config(
        site=raw["site"],
        cameras=[CameraConfig(**c) for c in raw["cameras"]],
        models=models,
        classes=classes,
        sample_fps=raw.get("sampling", {}).get("fps", 2.0),
        clip_retention_days=raw.get("retention", {}).get("clip_days", 60),
        security=security,
        root=path.parent.parent,
    )
