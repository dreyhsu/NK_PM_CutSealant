"""Detector functionality extracted from CutSealant V6.0.7."""
import queue

import numpy as np

def load_yolo(path):
    """Load Ultralytics only when a model is actually needed."""
    from ultralytics import YOLO
    return YOLO(path)


class YoloDetector:
    """Composed service; context supplies shared application interfaces."""

    def __init__(self, context):
        self.context = context
        self.model = None
        self.thread = None

    def _yolo_worker(self):
        """YOLO inference worker thread with enhanced fault tolerance for 24/7 operation"""
        ctx = self.context
        consecutive_failures = 0
        max_failures = 30  # Increased from 5 to 30 for better tolerance
        cuda_error_count = 0
        max_cuda_errors = 3

        try:
            while not ctx.stop_event.is_set():
                try:
                    frame = ctx.frame_queue.get(timeout=1.0)
                    if frame is None:
                        break

                    boxes, scores, categories = ctx._yolo_detect(frame)
                    consecutive_failures = 0

                    try:
                        ctx.result_queue.put((frame, boxes, scores, categories), timeout=0.1)
                    except queue.Full:
                        pass

                except queue.Empty:
                    continue
                except RuntimeError as e:
                    # Handle CUDA errors specifically
                    error_str = str(e).lower()
                    if 'cuda' in error_str or 'out of memory' in error_str:
                        cuda_error_count += 1
                        print(f"[WARNING] CUDA error detected ({cuda_error_count}/{max_cuda_errors}): {e}")

                        if ctx.logger:
                            ctx.logger.log_error("cuda_error", str(e), {
                                "cuda_error_count": cuda_error_count
                            })

                        # Try to recover from CUDA OOM
                        if 'out of memory' in error_str:
                            ctx._try_cuda_memory_cleanup()

                        if cuda_error_count >= max_cuda_errors:
                            print("[ERROR] Too many CUDA errors, attempting model reload...")
                            if ctx._try_reload_model():
                                cuda_error_count = 0
                                print("[INFO] Model reloaded successfully")
                            else:
                                print("[ERROR] Model reload failed")
                                break

                        ctx.clock.sleep(1.0)
                        continue
                    else:
                        # Other RuntimeError
                        consecutive_failures += 1
                        raise

                except Exception as e:
                    consecutive_failures += 1
                    error_msg = f"YOLO worker error: {e}"

                    # Log warning at intervals instead of every error
                    if consecutive_failures in [1, 5, 10, 20, 30]:
                        print(f"[WARNING] {error_msg} (failure {consecutive_failures}/{max_failures})")
                        if ctx.logger:
                            ctx.logger.log_error("yolo_worker_error", str(e), {
                                "consecutive_failures": consecutive_failures
                            })

                    if consecutive_failures >= max_failures:
                        error_msg = f"YOLO worker: {max_failures} consecutive failures, stopping"
                        print(f"[ERROR] {error_msg}")
                        if ctx.logger:
                            ctx.logger.log_error("yolo_worker_fatal_error", error_msg)
                        break

                    ctx.clock.sleep(0.5)

        except Exception as e:
            if ctx.logger:
                ctx.logger.log_error("yolo_thread_fatal_error", str(e))

        print("[INFO] YOLO thread stopped")

    def _try_cuda_memory_cleanup(self):
        """Attempt to clean up CUDA memory"""
        ctx = self.context
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                print("[INFO] CUDA memory cache cleared")
        except Exception as e:
            print(f"[WARNING] CUDA memory cleanup failed: {e}")

    def _try_reload_model(self):
        """Attempt to reload the YOLO model"""
        ctx = self.context
        try:
            print("[INFO] Attempting to reload YOLO model...")

            # Clear CUDA cache first
            ctx._try_cuda_memory_cleanup()

            # Reload model
            ctx.model = ctx.model_factory(ctx.paths['model_path'])
            ctx.model.to(ctx.device)

            if ctx.logger:
                ctx.logger.log_system_event("model_reloaded", {
                    "model_path": ctx.paths['model_path'],
                    "device": ctx.device
                })

            return True
        except Exception as e:
            print(f"[ERROR] Model reload failed: {e}")
            if ctx.logger:
                ctx.logger.log_error("model_reload_failed", str(e))
            return False

    def _yolo_detect(self, frame):
        """Infer only the ROI, returning boxes in original full-frame coordinates."""
        ctx = self.context
        try:
            if ctx.inference_roi is None:
                raise ValueError('Inference ROI must be loaded before detection')
            x1, y1, x2, y2 = ctx.inference_roi.scaled_bounds(frame.shape)
            crop = frame[y1:y2, x1:x2].copy()
            if crop.size == 0:
                raise ValueError('Inference ROI crop is empty')
            results = ctx.model.predict(
                crop,
                conf=ctx.yolo_config['confidence_threshold'],
                iou=ctx.yolo_config['iou_threshold'],
                agnostic_nms=False,
                max_det=ctx.yolo_config['max_detections'],
                augment=ctx.yolo_config['use_augmentation'],
                verbose=False,
                half=ctx.yolo_config['use_half_precision']
            )

            if len(results) > 0 and results[0].boxes is not None:
                result = results[0]
                boxes = result.boxes.xyxy.cpu().numpy().copy()
                scores = result.boxes.conf.cpu().numpy()
                categories = result.boxes.cls.cpu().numpy().astype(np.int32)
                boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, x2 - x1)
                boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, y2 - y1)
                boxes = boxes.astype(np.int32)
                valid = (boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1])
                boxes, scores, categories = boxes[valid], scores[valid], categories[valid]
                boxes += np.array([x1, y1, x1, y1], dtype=np.int32)
            else:
                boxes = np.array([]).reshape(0, 4)
                scores = np.array([])
                categories = np.array([])

        except Exception as e:
            error_msg = f"YOLO detection error: {e}"
            print(error_msg)
            if ctx.logger:
                ctx.logger.log_error("yolo_detection_error", e)
            boxes = np.array([]).reshape(0, 4)
            scores = np.array([])
            categories = np.array([])

        return boxes, scores, categories
