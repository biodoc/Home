# Substation Wildlife Monitor

Computer-vision system that watches substation camera feeds, detects wildlife
(squirrel, bird, raccoon) and intruders (person in perimeter during off-hours,
loitering), and maps where animals concentrate and nest to target outage
mitigation.

V1 runs as a home bench test: one UniFi G4 Bullet via Protect RTSP, with
inference and training on the RTX 3070 Linux workstation. Jetson edge
deployment is deferred.

Validation is human review. The dashboard doubles as a review queue: approve /
reject verdicts on surfaced detections become labels for retraining.

Full plan: Google Drive → Projects / Substation Wildlife Monitor / Project Plan.

## Status: light scaffold (Phase 1 POC)

```
config/default.yaml   cameras, models, config-driven classes, sampling, retention
monitor/config.py     config loader
monitor/types.py      Detection record
monitor/detect/       Detector interface + YOLO wrapper
scripts/poc_eval.py   run pretrained model over sample clips, report hits
training/             fine-tuning (RTX 3070, 8 GB — size model/batch to fit)
tests/
```

Not yet scaffolded: motion trigger, tracking (ByteTrack), event CSV logger,
clip store, intruder rule (perimeter/off-hours/loiter), dashboard + review
queue, overlap calibration, plugins.

## Run the POC

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[detect,dev]"
# drop 10–20 clips (day + IR night, with and without animals) into data/footage/
python scripts/poc_eval.py
pytest
```

Note: pretrained COCO has no squirrel/raccoon classes. The POC measures
how well it finds *something* (bird, person, cat/dog stand-ins) at your camera
distances; fine-tuning adds the real classes.

## Licensing

Ultralytics is AGPL-3.0 — accepted for this personal project; an Ultralytics
Enterprise license covers later commercial use.
