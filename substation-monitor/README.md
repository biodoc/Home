# Substation Wildlife Monitor

Edge computer-vision system that watches substation camera feeds, detects
wildlife (squirrel, raccoon, bird, snake) and people without hard hats, and
maps where animals concentrate and nest to target outage mitigation.

Full plan: Google Drive → Projects / Substation Wildlife Monitor / Project Plan.

## Status: light scaffold (Phase 1 POC)

```
config/default.yaml   cameras, models, config-driven classes, sampling, retention
monitor/config.py     config loader
monitor/types.py      Detection record
monitor/detect/       Detector interface + YOLO wrapper
scripts/poc_eval.py   run pretrained model over sample clips, report hits
training/             fine-tuning (runs on the GPU workstation)
tests/
```

Not yet scaffolded: motion trigger, tracking (ByteTrack), event CSV logger,
clip store, overlap calibration, dashboard, plugins.

## Run the POC

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[detect,dev]"
# drop 10–20 clips (day + IR night, with and without animals) into data/footage/
python scripts/poc_eval.py
pytest
```

Note: pretrained COCO has no squirrel/raccoon/snake classes. The POC measures
how well it finds *something* (bird, person, cat/dog stand-ins) at your camera
distances; fine-tuning adds the real classes.

## Licensing

Ultralytics is AGPL-3.0. Fine for a pilot; confirm with your employer before a
utility deployment (enterprise license, or switch to an Apache-2.0 model such
as RT-DETR).
