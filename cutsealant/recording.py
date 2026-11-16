"""Optional raw or annotated video recording."""
import os
from datetime import datetime

import cv2


class VideoRecorder:
    """Video recorder for saving raw or inference frames"""

    def __init__(self, recording_config, frame_shape, session_id, logger=None):
        self.recording_config = recording_config
        self.frame_shape = frame_shape
        self.session_id = session_id
        self.logger = logger
        self.video_writer = None
        self.is_recording = False
        self.output_filename = None

        # Store calculated recording dimensions
        self.recording_width = None
        self.recording_height = None

        if recording_config['enable_recording']:
            self._setup_recording()

    def _calculate_recording_dimensions(self):
        """Calculate and store recording dimensions based on quality setting"""
        height, width = self.frame_shape[:2]

        if self.recording_config['quality'] == 'low':
            width = int(width * 0.5)
            height = int(height * 0.5)
        elif self.recording_config['quality'] == 'medium':
            width = int(width * 0.7)
            height = int(height * 0.7)

        # Ensure even dimensions for codec compatibility
        width = width - (width % 2)
        height = height - (height % 2)

        self.recording_width = width
        self.recording_height = height

        return width, height

    def _setup_recording(self):
        """Setup video recording"""
        try:
            # Create recordings directory
            output_dir = self.recording_config['output_path']
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)

            # Generate filename with timestamp (only if not already set)
            if self.output_filename is None:
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                record_type = self.recording_config['record_type']
                video_format = self.recording_config['video_format']
                self.output_filename = f"{output_dir}{self.session_id}_{record_type}_{timestamp}.{video_format}"

            video_format = self.recording_config['video_format']

            # Set codec based on format - optimized for macOS
            if video_format.lower() == 'mp4':
                # Try macOS-friendly codecs in order
                codecs_to_try = ['avc1', 'mp4v', 'h264']
                fourcc = None

                for codec in codecs_to_try:
                    test_fourcc = cv2.VideoWriter_fourcc(*codec)
                    test_writer = cv2.VideoWriter('test.tmp', test_fourcc, 1, (100, 100))
                    if test_writer.isOpened():
                        test_writer.release()
                        if os.path.exists('test.tmp'):
                            os.remove('test.tmp')
                        fourcc = test_fourcc
                        print(f"Using codec: {codec}")
                        break
                    test_writer.release()

                if fourcc is None:
                    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # fallback

            elif video_format.lower() == 'avi':
                fourcc = cv2.VideoWriter_fourcc(*'MJPG')  # Better compatibility than XVID
            else:
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')

            # Calculate recording dimensions
            width, height = self._calculate_recording_dimensions()

            # Initialize video writer
            self.video_writer = cv2.VideoWriter(
                self.output_filename,
                fourcc,
                self.recording_config['fps'],
                (width, height)
            )

            if self.video_writer.isOpened():
                self.is_recording = True
                record_type = self.recording_config['record_type']
                print(f"Recording started: {self.output_filename}")
                print(f"Record type: {record_type}")
                print(f"Quality: {self.recording_config['quality']} ({width}x{height})")

                if self.logger:
                    self.logger.log_system_event("recording_started", {
                        "filename": self.output_filename,
                        "record_type": record_type,
                        "quality": self.recording_config['quality'],
                        "resolution": f"{width}x{height}"
                    })
            else:
                error_msg = f"Failed to initialize video writer: {self.output_filename}"
                print(error_msg)
                if self.logger:
                    self.logger.log_error("video_writer_init_error", error_msg)

        except Exception as e:
            error_msg = f"Recording setup error: {e}"
            print(error_msg)
            if self.logger:
                self.logger.log_error("recording_setup_error", e)

    def record_frame(self, frame):
        """Record a frame to video file"""
        if not self.is_recording or self.video_writer is None:
            return

        try:
            # Use pre-calculated recording dimensions for consistent sizing
            if self.recording_width is None or self.recording_height is None:
                self._calculate_recording_dimensions()

            # Resize frame to match video writer dimensions
            frame_resized = cv2.resize(
                frame,
                (self.recording_width, self.recording_height),
                interpolation=cv2.INTER_AREA
            )

            self.video_writer.write(frame_resized)

        except Exception as e:
            error_msg = f"Frame recording error: {e}"
            print(error_msg)
            if self.logger:
                self.logger.log_error("frame_recording_error", e)

    def start_new_recording(self):
        """Start a new recording session (for toggle functionality)"""
        # Reset filename to generate new one
        self.output_filename = None
        self._setup_recording()

    def stop_recording(self):
        """Stop recording and release resources"""
        if self.is_recording and self.video_writer is not None:
            try:
                self.video_writer.release()
                self.is_recording = False

                # Get file size
                file_size = 0
                if os.path.exists(self.output_filename):
                    file_size = os.path.getsize(self.output_filename)

                print(f"Recording stopped: {self.output_filename}")
                print(f"File size: {file_size / (1024*1024):.2f} MB")

                if self.logger:
                    self.logger.log_system_event("recording_stopped", {
                        "filename": self.output_filename,
                        "file_size_mb": round(file_size / (1024*1024), 2)
                    })

            except Exception as e:
                error_msg = f"Recording stop error: {e}"
                print(error_msg)
                if self.logger:
                    self.logger.log_error("recording_stop_error", e)
