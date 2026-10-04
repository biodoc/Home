from typing import Protocol

import numpy as np

from monitor.types import Detection


class Detector(Protocol):
    """Anything that turns a BGR frame into mapped detections.

    Wildlife and any optional add-on models (e.g. hard-hat) implement this, so the pipeline
    can run any number of them on the same frame.
    """

    name: str

    def detect(self, frame: np.ndarray) -> list[Detection]: ...
