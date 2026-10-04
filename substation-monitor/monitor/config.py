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
class Config:
    site: str
    cameras: list[CameraConfig]
    models: dict[str, ModelConfig]
    classes: dict[str, ClassConfig]
    sample_fps: float = 2.0
    clip_retention_days: int = 60
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

    return Config(
        site=raw["site"],
        cameras=[CameraConfig(**c) for c in raw["cameras"]],
        models=models,
        classes=classes,
        sample_fps=raw.get("sampling", {}).get("fps", 2.0),
        clip_retention_days=raw.get("retention", {}).get("clip_days", 60),
        root=path.parent.parent,
    )
