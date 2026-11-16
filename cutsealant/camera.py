"""Camera functionality extracted from CutSealant V6.0.7."""
import os
import queue
from glob import glob

import cv2

try:
    from tis_camera_wrapper import TISCameraWrapper
    TIS_AVAILABLE = True
except ImportError:
    TIS_AVAILABLE = False

class CameraSource:
    """Composed service; context supplies shared application interfaces."""

    def __init__(self, context):
        self.context = context
        self.capture = None
        self.thread = None

    def _setup_video_source(self):
        """Setup video source (camera, dfk_camera, or file) with error handling"""
        ctx = self.context
        try:
            video_config = ctx.config_manager.get_video_source_config()

            if video_config['mode'] == 'camera':
                # 一般 USB/UVC 相機
                print(f"Initializing camera {video_config['camera_index']}")
                cap = ctx.capture_factory(video_config['camera_index'], cv2.CAP_DSHOW)

                if not cap.isOpened():
                    error_msg = f"Failed to open camera {video_config['camera_index']}"
                    print(error_msg)
                    if ctx.logger:
                        ctx.logger.log_error("camera_open_error", error_msg)
                    return None, None

                # Set camera properties
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, video_config['camera_width'])
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, video_config['camera_height'])
                cap.set(cv2.CAP_PROP_AUTOFOCUS, video_config['camera_focus'])
                cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)

                # 設定手動焦距值 (範圍通常是 0-255)
                cap.set(cv2.CAP_PROP_FOCUS, 0)
                source_name = f"Camera_{video_config['camera_index']}"
                print(f"Camera initialized: {video_config['camera_width']}x{video_config['camera_height']}")

            elif video_config['mode'] == 'dfk_camera':
                # TIS DFK 工業相機
                if not TIS_AVAILABLE:
                    error_msg = "TIS camera support not available. Please ensure tis_camera_wrapper.py and TIS_API.py are in the project directory."
                    print(error_msg)
                    if ctx.logger:
                        ctx.logger.log_error("tis_not_available", error_msg)
                    return None, None

                print(f"Initializing TIS DFK camera...")

                # 決定使用索引還是名稱搜尋
                camera_index = None
                tis_camera_idx = video_config.get('tis_camera_index', -1)
                if tis_camera_idx >= 0:
                    camera_index = tis_camera_idx

                # 建立 TIS 相機包裝器
                cap = TISCameraWrapper(
                    dll_path=video_config.get('tis_dll_path', './tisgrabber_x64.dll'),
                    device_name=video_config.get('tis_device_name', 'DFK 33UX178'),
                    camera_index=camera_index,
                    logger=ctx.logger
                )

                if not cap.isOpened():
                    error_msg = "Failed to open TIS DFK camera"
                    print(error_msg)
                    if ctx.logger:
                        ctx.logger.log_error("dfk_camera_open_error", error_msg)
                    return None, None

                # 取得設備資訊
                device_info = cap.get_device_info()
                source_name = f"TIS_{device_info['device_name']}"
                print(f"TIS DFK camera initialized: {device_info['resolution']}")

                if ctx.logger:
                    ctx.logger.log_system_event("dfk_camera_initialized", device_info)

            elif video_config['mode'] == 'video':
                video_files = glob(ctx.paths['video_path'])
                if not video_files:
                    error_msg = f"No video files found matching: {ctx.paths['video_path']}"
                    print(error_msg)
                    if ctx.logger:
                        ctx.logger.log_error("video_file_not_found", error_msg)
                    return None, None

                video_file = video_files[video_config['video_index']]
                print(f"Opening video file: {video_file}")
                cap = ctx.capture_factory(video_file)

                if not cap.isOpened():
                    error_msg = f"Failed to open video file: {video_file}"
                    print(error_msg)
                    if ctx.logger:
                        ctx.logger.log_error("video_open_error", error_msg)
                    return None, None

                source_name = os.path.basename(video_file)
                print(f"Video file opened: {video_file}")

            else:
                error_msg = f"Invalid video source mode: {video_config['mode']}. Supported modes: 'camera', 'dfk_camera', 'video'"
                print(error_msg)
                if ctx.logger:
                    ctx.logger.log_error("config_error", error_msg)
                raise ValueError(error_msg)

            return cap, source_name

        except Exception as e:
            if ctx.logger:
                ctx.logger.log_error("video_source_init_error", e)
            raise

    def _capture_frames(self):
        """Capture frames from video source with enhanced fault tolerance for 24/7 operation"""
        ctx = self.context
        consecutive_failures = 0
        max_failures = 50  # Increased from 10 to 50 for better tolerance (~5 seconds)

        try:
            while not ctx.stop_event.is_set():
                # Check if camera is still open
                if not ctx.cap or not ctx.cap.isOpened():
                    # Attempt reconnection for camera modes
                    video_config = ctx.config_manager.get_video_source_config()
                    if video_config['mode'] in ['camera', 'dfk_camera']:
                        if ctx._attempt_camera_reconnect():
                            consecutive_failures = 0
                            continue
                        else:
                            # Reconnection failed, wait and retry
                            ctx.clock.sleep(1.0)
                            continue
                    else:
                        # Video file mode - no reconnection
                        break

                try:
                    ret, frame = ctx.cap.read()
                    if not ret:
                        consecutive_failures += 1

                        # Log warning periodically instead of immediately stopping
                        if consecutive_failures == 10:
                            print(f"[WARNING] Camera read failures: {consecutive_failures}")
                            if ctx.logger:
                                ctx.logger.log_system_event("capture_warning", {
                                    "consecutive_failures": consecutive_failures,
                                    "message": "Multiple frame read failures detected"
                                })

                        if consecutive_failures >= max_failures:
                            error_msg = f"Camera read failures reached {max_failures}, attempting reconnection..."
                            print(f"[WARNING] {error_msg}")
                            if ctx.logger:
                                ctx.logger.log_error("capture_max_failures", error_msg)

                            # Try to reconnect instead of stopping
                            video_config = ctx.config_manager.get_video_source_config()
                            if video_config['mode'] in ['camera', 'dfk_camera']:
                                if ctx._attempt_camera_reconnect():
                                    consecutive_failures = 0
                                    print("[INFO] Camera reconnected successfully")
                                    continue

                            # If not camera mode or reconnection failed, break
                            break

                        ctx.clock.sleep(0.1)
                        continue

                    # Successful frame read
                    consecutive_failures = 0
                    ctx.last_frame_received_time = ctx.clock.time()
                    ctx.idle_warning_logged = False

                    try:
                        ctx.frame_queue.put(frame, timeout=0.1)
                    except queue.Full:
                        pass

                except Exception as e:
                    error_msg = f"Capture thread error: {e}"
                    print(f"[ERROR] {error_msg}")
                    if ctx.logger:
                        ctx.logger.log_error("capture_thread_error", str(e))

                    # Don't break immediately, try to recover
                    consecutive_failures += 1
                    if consecutive_failures >= max_failures:
                        break
                    ctx.clock.sleep(0.5)

        except Exception as e:
            if ctx.logger:
                ctx.logger.log_error("capture_thread_fatal_error", str(e))

        print("[INFO] Capture thread stopped")

    def _attempt_camera_reconnect(self):
        """Attempt to reconnect to camera"""
        ctx = self.context
        ctx.camera_reconnect_attempts += 1

        if ctx.camera_reconnect_attempts > ctx.max_camera_reconnect_attempts:
            error_msg = f"Max camera reconnection attempts ({ctx.max_camera_reconnect_attempts}) reached"
            print(f"[ERROR] {error_msg}")
            if ctx.logger:
                ctx.logger.log_error("camera_reconnect_max_attempts", error_msg)
            return False

        print(f"[INFO] Attempting camera reconnection (attempt {ctx.camera_reconnect_attempts}/{ctx.max_camera_reconnect_attempts})...")

        try:
            # Release old capture if exists
            if ctx.cap:
                try:
                    ctx.cap.release()
                except:
                    pass

            ctx.clock.sleep(ctx.camera_reconnect_delay)

            # Reinitialize camera
            ctx.cap, ctx.source_name = ctx._setup_video_source()

            if ctx.cap and ctx.cap.isOpened():
                print(f"[INFO] Camera reconnected successfully: {ctx.source_name}")
                if ctx.logger:
                    ctx.logger.log_system_event("camera_reconnected", {
                        "attempt": ctx.camera_reconnect_attempts,
                        "source": ctx.source_name
                    })
                ctx.camera_reconnect_attempts = 0  # Reset counter on success
                return True
            else:
                print(f"[WARNING] Camera reconnection attempt {ctx.camera_reconnect_attempts} failed")
                return False

        except Exception as e:
            print(f"[ERROR] Camera reconnection error: {e}")
            if ctx.logger:
                ctx.logger.log_error("camera_reconnect_error", str(e))
            return False
