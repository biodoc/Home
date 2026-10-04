from pathlib import Path

import pytest

from monitor.config import load_config
from monitor.types import Detection

ROOT = Path(__file__).resolve().parent.parent


def test_default_config_loads():
    cfg = load_config(ROOT / "config" / "default.yaml")
    assert cfg.clip_retention_days == 60
    assert {"squirrel", "raccoon", "bird", "snake", "person"} <= cfg.classes.keys()


def test_label_map():
    cfg = load_config(ROOT / "config" / "default.yaml")
    m = cfg.label_map("wildlife")
    assert m["bird"] == "bird"
    assert m["cat"] == "squirrel"


def test_unknown_model_rejected(tmp_path):
    (tmp_path / "config").mkdir()
    p = tmp_path / "config" / "bad.yaml"
    p.write_text(
        "site: x\ncameras: []\nmodels: {}\n"
        "classes:\n  bird: {model: nope, labels: [bird]}\n"
    )
    with pytest.raises(ValueError):
        load_config(p)


def test_centroid():
    d = Detection("bird", "bird", 0.9, (10, 20, 30, 40))
    assert d.centroid == (25, 40)
