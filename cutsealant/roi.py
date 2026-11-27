"""Shared inference-crop geometry for detection and placement overlays."""
from dataclasses import dataclass
import json
import math
from pathlib import Path


DEFAULT_INFERENCE_ROI = {
    'version': 1,
    'reference_size': [1344, 756],
    'roi_xyxy': [454, 312, 850, 670],
}


@dataclass(frozen=True)
class InferenceROI:
    reference_size: tuple[int, int]
    roi_xyxy: tuple[int, int, int, int]

    @classmethod
    def load(cls, path):
        """Reject missing/invalid ROI files rather than infer the whole frame."""
        with Path(path).open(encoding='utf-8-sig') as stream:
            data = json.load(stream)
        if not isinstance(data, dict) or type(data.get('version')) is not int or data['version'] != 1:
            raise ValueError('Inference ROI requires version 1')
        size, box = data.get('reference_size'), data.get('roi_xyxy')
        if (not isinstance(size, list) or len(size) != 2
                or any(type(value) is not int or value <= 0 for value in size)):
            raise ValueError('Inference ROI reference_size must contain two positive integers [width, height]')
        if (not isinstance(box, list) or len(box) != 4
                or any(type(value) is not int for value in box)):
            raise ValueError('Inference ROI roi_xyxy must contain four integers [x1, y1, x2, y2]')
        x1, y1, x2, y2 = box
        if not (0 <= x1 < x2 <= size[0] and 0 <= y1 < y2 <= size[1]):
            raise ValueError('Inference ROI must be nonempty and inside reference_size')
        return cls(tuple(size), tuple(box))

    def scaled_bounds(self, frame_shape):
        """Scale reference coordinates to the original frame; right/bottom exclusive."""
        height, width = frame_shape[:2]
        if height <= 0 or width <= 0:
            raise ValueError('Inference ROI requires a nonempty frame')
        ref_width, ref_height = self.reference_size
        x1, y1, x2, y2 = self.roi_xyxy
        return (max(0, math.floor(x1 * width / ref_width)),
                max(0, math.floor(y1 * height / ref_height)),
                min(width, math.ceil(x2 * width / ref_width)),
                min(height, math.ceil(y2 * height / ref_height)))
