# Training (runs on the GPU workstation)

The Jetson only runs inference. Training and retraining happen here, and the
exported weights (TensorRT `.engine` for the Jetson) are deployed out.

Planned flow:
1. Build a YOLO-format dataset from public sources (Caltech Camera Traps,
   iNaturalist, Open Images, COCO) covering squirrel, bird, raccoon,
   day + IR night.
2. Fine-tune YOLO11 (`train.py`).
3. Pseudo-label site footage; human-review low-confidence cases; retrain.
4. Export to TensorRT on the Jetson.

Person detection uses pretrained COCO; a hard-hat model is optional and would
follow the same flow with its own dataset.
