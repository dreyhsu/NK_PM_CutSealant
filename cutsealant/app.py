"""Application composition, main loop and backward-compatible processor facade.

Services use this object as their context. Carrier state lives in workflow.state;
camera/model handles live in their respective services. Explicit forwarding
methods retain the original processor API without mixins or circular imports.
"""
import os
import queue
import subprocess
import threading
import time
from datetime import datetime

import cv2

from .camera import CameraSource
from .config import ConfigManager
from .detector import YoloDetector, load_yolo
from .display import Display
from .inspection import SealantInspector
from .logs import JSONLogger, RejudgeLogger, StatisticsLogger
from .monitoring import Monitoring
from .recording import VideoRecorder
from .review import ReviewController
from .tower import TowerController
from .workflow import InspectionState, InspectionWorkflow


class SafeVideoProcessor:
    """Compose services while preserving the original processor's behavior/API."""

    _OWNERS = {
        **{name: ('workflow', 'state', name) for name in InspectionState.__dataclass_fields__},
        'cap': ('camera', 'capture'),
        'capture_thread': ('camera', 'thread'),
        'model': ('detector', 'model'),
        'yolo_thread': ('detector', 'thread'),
        'last_light_status': ('tower', 'last_status'),
    }

    def __getattr__(self, name):
        path = self._OWNERS.get(name)
        if path:
            owner = self
            for part in path:
                owner = object.__getattribute__(owner, part)
            return owner
        raise AttributeError(f"{type(self).__name__!s} has no attribute {name!r}")

    def __setattr__(self, name, value):
        path = self._OWNERS.get(name)
        if path:
            owner = self
            for part in path[:-1]:
                owner = object.__getattribute__(owner, part)
            setattr(owner, path[-1], value)
        else:
            object.__setattr__(self, name, value)

    def __init__(self, *, model_factory=None, capture_factory=None, command_runner=None, clock=None):
        self.clock = clock if clock is not None else time
        self.model_factory = model_factory if model_factory is not None else load_yolo
        self.capture_factory = capture_factory if capture_factory is not None else lambda *a, **kw: cv2.VideoCapture(*a, **kw)
        self.command_runner = command_runner
        self.frame_queue = queue.Queue(maxsize=2)
        self.result_queue = queue.Queue(maxsize=2)
        self.camera = CameraSource(self)
        self.detector = YoloDetector(self)
        self.inspection = SealantInspector(self)
        self.workflow = InspectionWorkflow(self)
        self.tower = TowerController(self)
        self.display = Display(self)
        self.review = ReviewController(self)
        self.monitoring = Monitoring(self)
        self.running = False
        self._cleanup_done = False
        self.stop_event = threading.Event()
        self.cap = None
        self.model = None
        self.logger = None
        self.video_recorder = None
        self.statistics_logger = None  # NEW: CSV statistics logger

        # Performance tracking
        self.frame_count = 0
        self.detection_count = 0
        self.start_time = self.clock.time()

        # Configuration
        self.config_manager = None
        self.paths = {}
        self.yolo_config = {}
        self.detection_config = {}
        self.display_config = {}
        self.device = 'cpu'
        self.light_tower_config = {'mode': 'usb'}
        self.class_map = {}
        self.recording_config = {}

        # Video source info
        self.source_name = "Unknown"
        self.frame_shape = None

        # Detection results storage
        self.current_sealant_result = None
        self.sealant_detection_results = {}

        # Worker threads
        self.capture_thread = None
        self.yolo_thread = None

        # === NEW: Detection State Machine ===
        self.detection_state = 'WAITING'  # WAITING, DETECTED, RESULT_HOLD
        self.detection_result = None      # None, 'Pass', 'NG'
        self.last_light_status = None     # Track last light status to avoid duplicate calls
        self.detection_triggered_time = 0 # Track when detection was last triggered

        # === NEW: Tolerance mechanism for detection box flickering ===
        self.absent_frame_count = 0       # Count consecutive frames with missing objects
        self.result_hold_start_time = 0   # Timestamp when entering RESULT_HOLD state

        # === NEW: Detection cycle time tracking ===
        self.detection_start_time = 0     # Timestamp when detection cycle starts

        # === Display mode control ===
        self.fit_screen_mode = True       # True = fit screen, False = original size
        self.is_fullscreen = False        # Track fullscreen state
        self.cached_screen_size = None    # Cache screen size to avoid repeated tkinter calls

        # === Screenshot settings ===
        self.screenshot_path = './screenshots/'

        # === 24/7 Monitoring Mode - Enhanced tolerance settings ===
        self.last_heartbeat_time = self.clock.time()
        self.heartbeat_interval = 60  # Log heartbeat every 60 seconds
        self.last_frame_received_time = self.clock.time()
        self.idle_warning_threshold = 300  # Warn after 5 minutes of no frames
        self.idle_warning_logged = False

        # Camera reconnection settings
        self.camera_reconnect_attempts = 0
        self.max_camera_reconnect_attempts = 5
        self.camera_reconnect_delay = 5  # seconds between reconnect attempts

        # === V6.0: Manual Review / Rejudge settings ===
        self.rejudge_logger = None           # RejudgeLogger instance
        self.manual_review_window = None     # ManualReviewWindow instance
        self.pending_review = None           # Pending review data for current carrier
        self.carrier_count = 0               # Carrier sequence number
        self.user_reviewed_flag = False      # Whether user pressed M key in current RESULT_HOLD
        self.current_raw_frame = None        # Store raw frame for review window
        self.current_sealant_boxes = []      # Store sealant boxes for review window
        self.current_scores = {}             # Store confidence scores for sealants
        self.rejudge_config = {}             # Rejudge configuration
        self.review_log_written = False      # Whether rejudge log has been written for current carrier
        self.fullscreen_paused_for_review = False  # Whether fullscreen was exited to show review window
        self.camera_health_config = {}

    def initialize(self):
        """Initialize all components with enhanced error handling"""
        try:
            print("Initializing video processor...")

            # Load configuration
            self.config_manager = ConfigManager()
            self.paths = self.config_manager.get_paths()
            self.yolo_config = self.config_manager.get_yolo_config()
            self.detection_config = self.config_manager.get_detection_config()
            self.camera_health_config = self.config_manager.get_camera_health_config()
            self.display_config = self.config_manager.get_display_config()
            self.device = self.config_manager.get_device()
            self.light_tower_config = self.config_manager.get_light_tower_config()
            self.class_map = self.config_manager.get_classes()
            self.recording_config = self.config_manager.get_recording_config()

            # Initialize logger
            logging_config = self.config_manager.get_logging_config()
            if logging_config['save_detection_log']:
                self.logger = JSONLogger(
                    logging_config['log_file_path'],
                    logging_config['log_level']
                )

                # Log configuration
                config_data = {
                    "paths": self.paths,
                    "yolo_config": self.yolo_config,
                    "detection_config": self.detection_config,
                    "camera_health_config": self.camera_health_config,
                    "display_config": self.display_config,
                    "device": self.device,
                    "light_tower_config": self.light_tower_config,
                    "classes": self.class_map,
                    "recording_config": self.recording_config
                }
                self.logger.log_configuration(config_data)

                # Initialize statistics logger (CSV)
                log_dir = os.path.dirname(logging_config['log_file_path'])
                self.statistics_logger = StatisticsLogger(log_dir, self.logger.session_id)

                # V6.0: Initialize rejudge logger
                self.rejudge_logger = RejudgeLogger(log_dir, self.logger.session_id)

            # V6.0: Load rejudge configuration
            self.rejudge_config = self.config_manager.get_rejudge_config()

            # Load YOLO model
            print(f"Loading YOLO model: {self.paths['model_path']}")
            if not os.path.exists(self.paths['model_path']):
                error_msg = f"Model file not found: {self.paths['model_path']}"
                print(error_msg)
                if self.logger:
                    self.logger.log_error("model_file_not_found", error_msg)
                return False

            self.model = self.model_factory(self.paths['model_path'])
            self.model.to(self.device)
            print(f"Model loaded successfully on device: {self.device}")

            # Initialize video source
            self.cap, self.source_name = self._setup_video_source()
            if self.cap is None:
                return False

            # Get frame shape for recording
            ret, test_frame = self.cap.read()
            if ret:
                self.frame_shape = test_frame.shape
                # Put frame back for processing
                try:
                    self.frame_queue.put(test_frame, timeout=0.1)
                except queue.Full:
                    pass

            # Initialize video recorder if enabled
            if self.recording_config['enable_recording'] and self.logger:
                self.video_recorder = VideoRecorder(
                    self.recording_config,
                    self.frame_shape,
                    self.logger.session_id,
                    self.logger
                )

            print("Video processor initialized successfully")
            if self.logger:
                self.logger.log_system_event("initialization_complete")

            return True

        except Exception as e:
            error_msg = f"Initialization failed: {e}"
            print(error_msg)
            if self.logger:
                self.logger.log_error("initialization_error", e)
            import traceback
            traceback.print_exc()
            return False

    def start_threads(self):
        """Start worker threads"""
        try:
            self.capture_thread = threading.Thread(
                target=self._capture_frames,
                name="CaptureThread"
            )
            self.yolo_thread = threading.Thread(
                target=self._yolo_worker,
                name="YOLOThread"
            )

            self.capture_thread.daemon = True
            self.yolo_thread.daemon = True

            self.capture_thread.start()
            self.yolo_thread.start()

            print("Worker threads started")
            if self.logger:
                self.logger.log_system_event("threads_started")

        except Exception as e:
            error_msg = f"Failed to start threads: {e}"
            print(error_msg)
            if self.logger:
                self.logger.log_error("thread_start_error", e)
            raise

    def run(self):
        """Main processing loop with detection state machine"""
        try:
            if not self.initialize():
                return False

            self.start_threads()
            self.running = True
            self.tower.start()

            # Initialize processing variables
            stop_time = 0
            previous_centroid = [0, 0]

            # === NEW: Initialize detection state machine ===
            self.detection_state = 'WAITING'
            self.camera_error_start_time = 0
            self.detection_result = None
            self.workflow.panel.reset()
            current_status = self._call_light_once('Standby')

            # Reset sealant detection state
            self.current_sealant_result = None
            self.sealant_detection_results = {}

            # === NEW: Initialize tolerance mechanism ===
            self.absent_frame_count = 0
            self.result_hold_start_time = 0

            consecutive_empty_results = 0
            max_empty_results = 300000

            # === 24/7 Monitoring Mode: Removed hard timeout limit ===
            # Instead of stopping, we log warnings and continue running
            frame_timeout_count = 0

            print("Starting main processing loop with state machine...")
            print(f"Initial state: {self.detection_state}")
            print("*** 24/7 Monitoring Mode Enabled ***")
            print("\n=== Keyboard Shortcuts ===")
            print("  Q: Quit | S: Screenshot | F: Fit screen | G: Fullscreen")
            print("  P: Statistics | R: Reset stats | V: Toggle recording")
            print("  T: Retry tower after recovery exhaustion | Close window: Quit")
            print("  C: Reset camera alignment error | M: Review cases on carrier")
            print("===========================\n")

            # Create display window with proper flags for fullscreen support
            self._create_display_window(self.display_config['window_name'])

            if self.logger:
                self.logger.log_system_event("main_loop_started", {
                    "mode": "24/7_monitoring"
                })
                self.logger.log_state_change(None, self.detection_state, "initialization")

            while self.running and not self.stop_event.is_set():
                # Pump HighGUI even while frames are unavailable.
                key = cv2.waitKey(1) & 0xFF
                if cv2.getWindowProperty(
                        self.display_config['window_name'], cv2.WND_PROP_VISIBLE) < 1:
                    break
                # === Heartbeat logging for system health monitoring ===
                self._log_heartbeat()

                try:
                    frame, boxes, scores, categories = self.result_queue.get(timeout=0.1)
                    frame_timeout_count = 0
                    self.last_frame_received_time = self.clock.time()
                    self.idle_warning_logged = False

                except queue.Empty:
                    if key == ord('q'):
                        break
                    if key in (ord('t'), ord('T')):
                        self.tower.retry_latest()
                    if key in (ord('c'), ord('C')) and self.detection_state == 'CAMERA_ERROR':
                        current_status = self.workflow.reset_camera_error(current_status)
                        stop_time = 0
                        previous_centroid = [0, 0]
                    frame_timeout_count += 1

                    # === 24/7 Mode: Log warning instead of stopping ===
                    idle_duration = self.clock.time() - self.last_frame_received_time

                    # Log warning at intervals (every 30 seconds of idle)
                    if frame_timeout_count % 300 == 0:  # About every 30 seconds
                        print(f"[WARNING] No frames received for {idle_duration:.0f} seconds")
                        if self.logger:
                            self.logger.log_system_event("frame_idle_warning", {
                                "idle_duration_seconds": round(idle_duration, 1),
                                "timeout_count": frame_timeout_count
                            })

                    # Extended idle warning (configurable threshold)
                    if idle_duration >= self.idle_warning_threshold and not self.idle_warning_logged:
                        print(f"[WARNING] Extended idle period: {idle_duration:.0f} seconds without frames")
                        if self.logger:
                            self.logger.log_system_event("extended_idle_warning", {
                                "idle_duration_seconds": round(idle_duration, 1),
                                "threshold": self.idle_warning_threshold
                            })
                        self.idle_warning_logged = True

                    # Continue running - DO NOT BREAK for 24/7 operation
                    continue

                if frame is None:
                    # Only break if explicitly signaled to stop
                    print("[INFO] Received stop signal (None frame)")
                    break

                self.frame_count += 1

                # Check for empty results
                if len(categories) == 0:
                    consecutive_empty_results += 1
                else:
                    consecutive_empty_results = 0
                    self.detection_count += 1

                # === Fixed: Check video end only for 'video' mode, not by source_name ===
                video_config = self.config_manager.get_video_source_config()
                # if (video_config['mode'] == 'video' and
                #     consecutive_empty_results >= max_empty_results):
                #     print("[INFO] Video appears to have ended or no detections for too long.")
                #     if self.logger:
                #         self.logger.log_system_event("video_ended")
                #     break

                # Process frame
                raw_frame = frame.copy()
                sealant_boxes = []

                # Process detections
                if len(categories) > 0 and self.class_map['carrier'] in categories:
                    correct_carrier_box = self._check_correct_carrier(categories, boxes)
                    if correct_carrier_box is not None:
                        sealant_boxes = self._draw_detections(frame, boxes, categories, correct_carrier_box)

                # === NEW: Enhanced detection logic with state machine ===
                detected_classes = set(categories)
                carrier_present = self.class_map['carrier'] in detected_classes
                case_count = sum(
                    1 for category in categories
                    if category == self.class_map['sealant']
                )

                if carrier_present:
                    carrier_arriving = (
                        not self.workflow.panel.cycle_started
                        or self.workflow.panel.complete
                    )
                    if carrier_arriving and self.detection_state != 'CAMERA_ERROR':
                        current_status = self._call_light_once('Tracking')
                    # Keep evaluating while the carrier remains present, even if
                    # one or more case detections are temporarily blocked.
                    # V6.0: Pass scores for confidence tracking
                    current_status, stop_time, previous_centroid = self._handle_objects_present(
                        frame, raw_frame, boxes, categories, sealant_boxes, scores,
                        current_status, stop_time, previous_centroid
                    )
                else:
                    # Objects are absent - handle based on current state
                    current_status, stop_time, previous_centroid = self._handle_objects_absent(
                        current_status, stop_time, previous_centroid
                    )

                carrier_released = (
                    not carrier_present
                    and self.absent_frame_count
                    >= self.detection_config.get('absent_tolerance_frames', 5)
                )
                self.workflow.update_panel(
                    carrier_present,
                    case_count,
                    carrier_released,
                )
                self._add_ui_elements(frame, current_status)

                # Record after all inference annotations, including workflow.
                if self.video_recorder and self.video_recorder.is_recording:
                    if self.recording_config['record_type'] == 'raw':
                        self.video_recorder.record_frame(raw_frame)
                    else:  # 'inference'
                        self.video_recorder.record_frame(frame)

                # V6.0: Update manual review window if open
                if self.manual_review_window and self.manual_review_window.is_open:
                    review_result = self.manual_review_window.update()
                    if review_result in ['confirm', 'cancel']:
                        self._handle_review_complete(review_result)

                self._display_frame(frame, self.display_config['window_name'])

                # Check for quit command
                frame_key = cv2.waitKey(1) & 0xFF
                if frame_key != 255:
                    key = frame_key
                if key == ord('q'):
                    print("Quit command received")
                    if self.logger:
                        self.logger.log_system_event("quit_command_received")
                    break
                elif key == ord('s') or key == ord('S'):
                    # Screenshot function
                    self._take_screenshot(frame)
                elif key in (ord('t'), ord('T')):
                    self.tower.retry_latest()
                elif key in (ord('c'), ord('C')):
                    if self.detection_state == 'CAMERA_ERROR':
                        current_status = self.workflow.reset_camera_error(current_status)
                        stop_time = 0
                        previous_centroid = [0, 0]
                elif key == ord('p') or key == ord('P'):
                    # Print statistics (moved from 's')
                    self._print_statistics()
                elif key == ord('r'):
                    self._reset_statistics()
                elif key == ord('f') or key == ord('F'):
                    # Toggle fit screen mode
                    self._toggle_fit_screen()
                elif key == ord('g') or key == ord('G'):
                    # Toggle fullscreen mode
                    self._toggle_fullscreen(self.display_config['window_name'])
                elif key == ord('v') or key == ord('V'):
                    # Toggle video recording (can start even if not enabled in config)
                    self._toggle_recording()
                elif key == ord('m') or key == ord('M'):
                    # V6.0: Open manual review window (only in RESULT_HOLD state)
                    if self.detection_state == 'RESULT_HOLD':
                        self._open_manual_review()

                # Performance monitoring
                if self.frame_count % 100 == 0:
                    self._print_performance()

        except Exception as e:
            error_msg = f"Main loop error: {e}"
            print(error_msg)
            if self.logger:
                self.logger.log_error("main_loop_error", e)
            import traceback
            traceback.print_exc()
        finally:
            self.stop()

        return True

    def stop(self):
        """Stop video processor and cleanup resources"""
        if self._cleanup_done:
            return
        self._cleanup_done = True
        print("Stopping video processor...")
        self.running = False
        self.stop_event.set()
        self.tower.stop()

        # V6.0: Write any pending review before stopping
        if self.pending_review:
            print("[V6.0] Writing pending review before shutdown...")
            self._write_pending_review_log()

        # V6.0: Close manual review window if open
        if self.manual_review_window and self.manual_review_window.is_open:
            self.manual_review_window.close()

        # Wait for threads to stop
        if self.capture_thread and self.capture_thread.is_alive():
            self.capture_thread.join(timeout=2.0)
        if self.yolo_thread and self.yolo_thread.is_alive():
            self.yolo_thread.join(timeout=2.0)

        # Release video capture
        if self.cap:
            self.cap.release()

        # Close display windows
        cv2.destroyAllWindows()

        # Log final statistics and close logger
        if self.logger:
            elapsed_time = self.clock.time() - self.start_time
            fps = self.frame_count / elapsed_time if elapsed_time > 0 else 0
            final_stats = {
                "session_duration_seconds": round(elapsed_time, 1),
                "total_frames_processed": self.frame_count,
                "total_detections": self.detection_count,
                "final_average_fps": round(fps, 1),
                "final_detection_state": self.detection_state,
                "final_detection_result": self.detection_result,
                "total_carriers": self.carrier_count  # V6.0: Add carrier count
            }
            self.logger.log_performance_stats(final_stats)
            self.logger.close()

        # Close statistics logger
        if self.statistics_logger:
            self.statistics_logger.close()

        # Stop video recording
        if self.video_recorder:
            self.video_recorder.stop_recording()

        print("Video processor stopped")

    def _toggle_recording(self):
        """Toggle video recording on/off"""
        # Check if frame_shape is available
        if self.frame_shape is None:
            print("Cannot start recording: frame shape not available")
            return

        # Initialize recorder if not exists
        if not self.video_recorder:
            session_id = self.logger.session_id if self.logger else datetime.now().strftime('%Y%m%d_%H%M%S')
            self.video_recorder = VideoRecorder(
                self.recording_config,
                self.frame_shape,
                session_id,
                self.logger
            )

        if self.video_recorder:
            if self.video_recorder.is_recording:
                # Stop recording
                self.video_recorder.stop_recording()
                print("Recording stopped by user")
                if self.logger:
                    self.logger.log_system_event("recording_stopped_by_user")
            else:
                # Start new recording - use start_new_recording method
                self.video_recorder.start_new_recording()

                if self.video_recorder.is_recording:
                    print("Recording started by user")
                    if self.logger:
                        self.logger.log_system_event("recording_started_by_user", {
                            "filename": self.video_recorder.output_filename
                        })
                else:
                    print("Failed to start recording")

    def _setup_video_source(self, *args, **kwargs):
        """Compatibility forwarding to camera."""
        return self.camera._setup_video_source(*args, **kwargs)

    def _capture_frames(self, *args, **kwargs):
        """Compatibility forwarding to camera."""
        return self.camera._capture_frames(*args, **kwargs)

    def _attempt_camera_reconnect(self, *args, **kwargs):
        """Compatibility forwarding to camera."""
        return self.camera._attempt_camera_reconnect(*args, **kwargs)

    def _yolo_worker(self, *args, **kwargs):
        """Compatibility forwarding to detector."""
        return self.detector._yolo_worker(*args, **kwargs)

    def _try_cuda_memory_cleanup(self, *args, **kwargs):
        """Compatibility forwarding to detector."""
        return self.detector._try_cuda_memory_cleanup(*args, **kwargs)

    def _try_reload_model(self, *args, **kwargs):
        """Compatibility forwarding to detector."""
        return self.detector._try_reload_model(*args, **kwargs)

    def _yolo_detect(self, *args, **kwargs):
        """Compatibility forwarding to detector."""
        return self.detector._yolo_detect(*args, **kwargs)

    def _cut_sealant_detection(self, *args, **kwargs):
        """Compatibility forwarding to inspection."""
        return self.inspection._cut_sealant_detection(*args, **kwargs)

    def _loop_closed_identify(self, *args, **kwargs):
        """Compatibility forwarding to inspection."""
        return self.inspection._loop_closed_identify(*args, **kwargs)

    def _img_kmeans(self, *args, **kwargs):
        """Compatibility forwarding to inspection."""
        return self.inspection._img_kmeans(*args, **kwargs)

    def _handle_objects_present(self, *args, **kwargs):
        """Compatibility forwarding to workflow."""
        return self.workflow._handle_objects_present(*args, **kwargs)

    def _handle_objects_absent(self, *args, **kwargs):
        """Compatibility forwarding to workflow."""
        return self.workflow._handle_objects_absent(*args, **kwargs)

    def _check_detection_trigger(self, *args, **kwargs):
        """Compatibility forwarding to workflow."""
        return self.workflow._check_detection_trigger(*args, **kwargs)

    def _call_light_once(self, *args, **kwargs):
        """Compatibility forwarding to tower."""
        return self.tower._call_light_once(*args, **kwargs)

    def _call_light(self, *args, **kwargs):
        """Compatibility forwarding to tower."""
        return self.tower._call_light(*args, **kwargs)

    def _check_correct_carrier(self, *args, **kwargs):
        """Compatibility forwarding to display."""
        return self.display._check_correct_carrier(*args, **kwargs)

    def _calculate_iou(self, *args, **kwargs):
        """Compatibility forwarding to display."""
        return self.display._calculate_iou(*args, **kwargs)

    def _find_matching_box(self, *args, **kwargs):
        """Compatibility forwarding to display."""
        return self.display._find_matching_box(*args, **kwargs)

    def _draw_detections(self, *args, **kwargs):
        """Compatibility forwarding to display."""
        return self.display._draw_detections(*args, **kwargs)

    def _get_screen_size(self, *args, **kwargs):
        """Compatibility forwarding to display."""
        return self.display._get_screen_size(*args, **kwargs)

    def _create_display_window(self, *args, **kwargs):
        """Compatibility forwarding to display."""
        return self.display._create_display_window(*args, **kwargs)

    def _display_frame(self, *args, **kwargs):
        """Compatibility forwarding to display."""
        return self.display._display_frame(*args, **kwargs)

    def _toggle_fit_screen(self, *args, **kwargs):
        """Compatibility forwarding to display."""
        return self.display._toggle_fit_screen(*args, **kwargs)

    def _take_screenshot(self, *args, **kwargs):
        """Compatibility forwarding to display."""
        return self.display._take_screenshot(*args, **kwargs)

    def _toggle_fullscreen(self, *args, **kwargs):
        """Compatibility forwarding to display."""
        return self.display._toggle_fullscreen(*args, **kwargs)

    def _add_ui_elements(self, *args, **kwargs):
        """Compatibility forwarding to display."""
        return self.display._add_ui_elements(*args, **kwargs)

    def _open_manual_review(self, *args, **kwargs):
        """Compatibility forwarding to review."""
        return self.review._open_manual_review(*args, **kwargs)

    def _handle_review_complete(self, *args, **kwargs):
        """Compatibility forwarding to review."""
        return self.review._handle_review_complete(*args, **kwargs)

    def _write_pending_review_log(self, *args, **kwargs):
        """Compatibility forwarding to review."""
        return self.review._write_pending_review_log(*args, **kwargs)

    def _log_detection_results(self, *args, **kwargs):
        """Compatibility forwarding to monitoring."""
        return self.monitoring._log_detection_results(*args, **kwargs)

    def _print_statistics(self, *args, **kwargs):
        """Compatibility forwarding to monitoring."""
        return self.monitoring._print_statistics(*args, **kwargs)

    def _reset_statistics(self, *args, **kwargs):
        """Compatibility forwarding to monitoring."""
        return self.monitoring._reset_statistics(*args, **kwargs)

    def _print_performance(self, *args, **kwargs):
        """Compatibility forwarding to monitoring."""
        return self.monitoring._print_performance(*args, **kwargs)

    def _save_img_data(self, *args, **kwargs):
        """Compatibility forwarding to monitoring."""
        return self.monitoring._save_img_data(*args, **kwargs)

    def _log_heartbeat(self, *args, **kwargs):
        """Compatibility forwarding to monitoring."""
        return self.monitoring._log_heartbeat(*args, **kwargs)

    def _get_memory_usage(self, *args, **kwargs):
        """Compatibility forwarding to monitoring."""
        return self.monitoring._get_memory_usage(*args, **kwargs)


def main():
    """Main function with improved error handling"""
    processor = SafeVideoProcessor()

    try:
        success = processor.run()
        if success:
            print("Program completed successfully")
        else:
            print("Program failed to initialize")
            return 1

    except KeyboardInterrupt:
        print("\nProgram interrupted by user")
    except Exception as e:
        print(f"Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    finally:
        processor.stop()

    return 0
