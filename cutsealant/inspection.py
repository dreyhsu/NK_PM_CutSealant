"""Inspection functionality extracted from CutSealant V6.0.7."""
from dataclasses import dataclass

import cv2
import numpy as np

@dataclass(frozen=True)
class InspectionResult:
    """Snapshot: True in by_box means NG; missing boxes retain legacy skip behavior."""
    is_ng: bool
    by_box: dict[str, bool]


class SealantInspector:
    """Composed service; context supplies shared application interfaces."""

    def __init__(self, context):
        self.context = context

    def inspect(self, frame, boxes):
        """Analyze detected crops and return an independent result snapshot."""
        is_ng = self._cut_sealant_detection(frame, boxes)
        return InspectionResult(is_ng, dict(self.context.sealant_detection_results))

    def _cut_sealant_detection(self, frame, boxes):
        """Detect if sealant is properly cut and store individual results"""
        ctx = self.context
        if len(boxes) == 0:
            return False

        frame_hsl = cv2.cvtColor(frame, cv2.COLOR_BGR2HLS)
        result_list = []
        ctx.sealant_detection_results = {}

        for idx, box in enumerate(boxes):
            try:
                crop_img = frame_hsl[box[1]:box[3], box[0]:box[2], :]
                if crop_img.size == 0:
                    continue

                kmeans_img, center = ctx._img_kmeans(crop_img[:, :, 1])
                seg_sealant = cv2.inRange(kmeans_img, max(center), max(center))

                cnt_sealant, _ = cv2.findContours(seg_sealant, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

                if cnt_sealant:
                    cnt_area = [cv2.contourArea(cnt) for cnt in cnt_sealant]
                    sealant_idx = np.argmax(cnt_area)
                    closed_loop = ctx._loop_closed_identify(cnt_sealant[sealant_idx])
                    result_list.append(closed_loop)

                    box_key = f"{box[0]}_{box[1]}_{box[2]}_{box[3]}"
                    ctx.sealant_detection_results[box_key] = not closed_loop

            except Exception as e:
                error_msg = f"Sealant detection error: {e}"
                print(error_msg)
                if ctx.logger:
                    ctx.logger.log_error("sealant_detection_error", e, {
                        "box_index": idx,
                        "box_coordinates": box.tolist() if hasattr(box, 'tolist') else str(box)
                    })
                continue

        return any(item == False for item in result_list)

    def _loop_closed_identify(self, cnt):
        """Check if contour is a closed loop"""
        ctx = self.context
        if len(cnt) < 3:
            return False
        epsilon = 0.02 * cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, epsilon, True)
        return cv2.isContourConvex(approx)

    def _img_kmeans(self, img):
        """K-means clustering for image segmentation"""
        ctx = self.context
        Z = img.reshape(-1).astype(np.float32)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1)

        ret, label, center = cv2.kmeans(
            Z,
            ctx.detection_config['kmeans_clusters'],
            None,
            criteria,
            ctx.detection_config['kmeans_rounds'],
            cv2.KMEANS_RANDOM_CENTERS
        )
        center = center.astype(np.uint8)
        res = center[label.flatten()]

        return res.reshape(img.shape), center
