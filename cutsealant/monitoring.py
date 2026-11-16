"""Monitoring functionality extracted from CutSealant V6.0.7."""
import os
from datetime import datetime

import cv2

class Monitoring:
    """Composed service; context supplies shared application interfaces."""

    def __init__(self, context):
        self.context = context

    def _log_detection_results(self, sealant_result, frame_shape, centroid):
        """Log detection results"""
        ctx = self.context
        if not ctx.logger:
            return

        # Calculate detection statistics
        total_sealants = len(ctx.sealant_detection_results)
        ng_count = sum(1 for is_broken in ctx.sealant_detection_results.values() if is_broken)
        ok_count = total_sealants - ng_count

        # Prepare detection details for logging
        detection_details = []
        for i, (box_key, is_broken) in enumerate(ctx.sealant_detection_results.items(), 1):
            box_coords = box_key.split('_')
            detection_details.append({
                "sealant_id": i,
                "box_coordinates": {
                    "x1": int(box_coords[0]),
                    "y1": int(box_coords[1]),
                    "x2": int(box_coords[2]),
                    "y2": int(box_coords[3])
                },
                "status": "NG" if is_broken else "OK",
                "is_broken": is_broken
            })

        h, w = frame_shape[:2]
        frame_info = {
            "frame_width": w,
            "frame_height": h,
            "carrier_centroid": centroid,
            "source": ctx.source_name,
            "detection_state": ctx.detection_state
        }

        # Log detection results
        ctx.logger.log_detection_result(
            total_sealants, ok_count, ng_count,
            detection_details, frame_info
        )

    def _print_statistics(self):
        """Print processing statistics"""
        ctx = self.context
        elapsed_time = ctx.clock.time() - ctx.start_time
        fps = ctx.frame_count / elapsed_time if elapsed_time > 0 else 0
        detection_rate = (ctx.detection_count / ctx.frame_count * 100) if ctx.frame_count > 0 else 0

        stats = {
            "frames_processed": ctx.frame_count,
            "detections": ctx.detection_count,
            "detection_rate_percent": round(detection_rate, 1),
            "average_fps": round(fps, 1),
            "running_time_seconds": round(elapsed_time, 1),
            "detection_state": ctx.detection_state,
            "detection_result": ctx.detection_result
        }

        print(f"\n=== Statistics ===")
        print(f"Frames processed: {stats['frames_processed']}")
        print(f"Detections: {stats['detections']}")
        print(f"Detection rate: {stats['detection_rate_percent']}%")
        print(f"Average FPS: {stats['average_fps']}")
        print(f"Running time: {stats['running_time_seconds']}s")
        print(f"Detection state: {stats['detection_state']}")
        print(f"Detection result: {stats['detection_result']}")
        print("==================\n")

        # Log performance statistics
        if ctx.logger:
            ctx.logger.log_performance_stats(stats)

    def _reset_statistics(self):
        """Reset statistics"""
        ctx = self.context
        ctx.frame_count = 0
        ctx.detection_count = 0
        ctx.start_time = ctx.clock.time()
        print("Statistics reset")
        if ctx.logger:
            ctx.logger.log_system_event("statistics_reset")

    def _print_performance(self):
        """Print performance info"""
        ctx = self.context
        elapsed_time = ctx.clock.time() - ctx.start_time
        fps = ctx.frame_count / elapsed_time if elapsed_time > 0 else 0
        print(f"Performance: {ctx.frame_count} frames, {fps:.1f} FPS, State: {ctx.detection_state}")

    def _save_img_data(self, frame, boxes, result):
        """Save image data for validation"""
        ctx = self.context
        try:
            # Create validation directory if it doesn't exist
            validation_dir = ctx.paths['validation_path']
            if not os.path.exists(validation_dir):
                os.makedirs(validation_dir)

            # Generate filename with timestamp and result
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')[:-3]  # milliseconds
            result_str = 'NG' if result else 'OK'
            filename = f"{validation_dir}{timestamp}_{result_str}.jpg"

            # Save the frame
            cv2.imwrite(filename, frame)

            # Log the save operation
            if ctx.logger:
                ctx.logger.log_system_event("image_saved", {
                    "filename": filename,
                    "result": result_str,
                    "detection_state": ctx.detection_state
                })

            print(f"Validation image saved: {filename}")

        except Exception as e:
            error_msg = f"Failed to save validation image: {e}"
            print(error_msg)
            if ctx.logger:
                ctx.logger.log_error("image_save_error", e)

    def _log_heartbeat(self):
        """Log heartbeat for system health monitoring (24/7 mode)"""
        ctx = self.context
        current_time = ctx.clock.time()

        if current_time - ctx.last_heartbeat_time >= ctx.heartbeat_interval:
            ctx.last_heartbeat_time = current_time

            # Calculate runtime statistics
            elapsed_time = current_time - ctx.start_time
            fps = ctx.frame_count / elapsed_time if elapsed_time > 0 else 0

            # Get memory usage if available
            memory_info = ctx._get_memory_usage()

            heartbeat_data = {
                "uptime_seconds": round(elapsed_time, 1),
                "uptime_hours": round(elapsed_time / 3600, 2),
                "total_frames": ctx.frame_count,
                "total_detections": ctx.detection_count,
                "average_fps": round(fps, 1),
                "detection_state": ctx.detection_state,
                "camera_reconnect_attempts": ctx.camera_reconnect_attempts,
                "memory_usage": memory_info
            }

            if ctx.logger:
                ctx.logger.log_system_event("heartbeat", heartbeat_data)

            # Print brief status to console
            print(f"[HEARTBEAT] Uptime: {elapsed_time/3600:.1f}h | Frames: {ctx.frame_count} | FPS: {fps:.1f} | State: {ctx.detection_state}")

    def _get_memory_usage(self):
        """Get current memory usage statistics"""
        ctx = self.context
        try:
            import psutil
            process = psutil.Process()
            mem_info = process.memory_info()
            return {
                "rss_mb": round(mem_info.rss / (1024 * 1024), 1),
                "vms_mb": round(mem_info.vms / (1024 * 1024), 1)
            }
        except ImportError:
            return {"note": "psutil not available"}
        except Exception:
            return {"note": "error getting memory info"}
