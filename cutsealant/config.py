"""INI settings and adjacent inference ROI JSON; existing path semantics remain."""
import configparser
import json
import math
import os
from pathlib import Path

from .roi import DEFAULT_INFERENCE_ROI, InferenceROI


class ConfigManager:
    """Read config/config.ini with logging support."""

    def __init__(self, config_file='config/config.ini'):
        self.config = configparser.ConfigParser()
        self.config_file = config_file
        self.load_config()

    def load_config(self):
        """Load configuration from ini file"""
        if not os.path.exists(self.config_file):
            self.create_default_config()

        #self.config.read(self.config_file)
        self.config.read(self.config_file, encoding="utf-8-sig")
    def create_default_config(self):
        """Create default configuration file if it doesn't exist"""
        print(f"Creating default config file: {self.config_file}")

        self.config['PATHS'] = {
            'model_path': './models/best.pt',
            'video_path': './videoData/*.avi',
            'validation_path': './validation/',
            'stack_light_path': 'LR6main.py'
        }

        self.config['VIDEO_SOURCE'] = {
            'mode': 'camera',
            'camera_index': '0',
            'camera_width': '1080',
            'camera_height': '1920',
            'camera_focus': '0',
            'video_index': '0',
            # TIS DFK camera settings
            'tis_dll_path': './tisgrabber_x64.dll',
            'tis_device_name': 'DFK 33UX178',
            'tis_camera_index': '-1'
        }

        self.config['YOLO_SETTINGS'] = {
            'confidence_threshold': '0.55',
            'iou_threshold': '0.45',
            'max_detections': '100',
            'use_half_precision': 'true',
            'use_augmentation': 'false'
        }

        self.config['DETECTION_SETTINGS'] = {
            'fps': '15',
            'stop_threshold_seconds': '0.95',
            'distance_threshold_pixels': '10',
            'expected_case_count': '4',  # Legacy API only; cycles discover occupied quadrants.
            'case_pass_consecutive_frames': '10',
            'case_fail_consecutive_frames': '5',
            'kmeans_clusters': '3',
            'kmeans_rounds': '10',
            # Tolerance mechanism for detection box flickering
            'absent_tolerance_frames': '5',
            'result_hold_min_seconds': '2.0'
        }

        self.config['DISPLAY_SETTINGS'] = {
            'show_crosshair': 'true',
            'show_timestamp': 'true',
            'show_status': 'true',
            'window_name': 'frame',
            'auto_fit_screen': 'true',
            'max_scale': '1.0',
            'min_scale': '0.1'
        }

        self.config['DEVICE_SETTINGS'] = {
            'device': 'cpu'
        }

        self.config['LIGHT_TOWER'] = {
            'mode': 'usb'
        }

        self.config['CLASSES'] = {
            'case': '0',
            'cover': '1',
            'base': '2',
            'carrier': '3'
        }

        # Add logging settings
        self.config['LOGGING_SETTINGS'] = {
            'log_level': 'INFO',
            'save_detection_log': 'true',
            'log_file_path': './logs/detection.log'
        }

        # Add recording settings
        self.config['RECORDING_SETTINGS'] = {
            'enable_recording': 'false',
            'record_type': 'inference',  # 'raw' or 'inference'
            'output_path': './recordings/',
            'video_format': 'mp4',
            'fps': '15',
            'quality': 'medium'  # 'low', 'medium', 'high'
        }

        self.config['CASE_CROP_SAVING'] = {
            'enabled': 'false',
            'output_path': './case_crops/',
        }

        self.config['CAMERA_HEALTH'] = {
            'roi_x_ratio': '0.25',
            'roi_y_ratio': '0.25',
            'roi_w_ratio': '0.50',
            'roi_h_ratio': '0.50',
            'alert_blink_interval': '0.5',
            'alert_font_scale_ratio': '0.04',
            'alert_border_thickness_ratio': '0.012',
        }

        # Create the destination directory before saving a new config.
        config_dir = os.path.dirname(self.config_file)
        if config_dir:
            os.makedirs(config_dir, exist_ok=True)

        # Save default config
        with open(self.config_file, 'w') as f:
            self.config.write(f)
        roi_path = Path(self.config_file).with_name('inference_roi.json')
        if not roi_path.exists():
            roi_path.write_text(json.dumps(DEFAULT_INFERENCE_ROI, indent=2) + '\n', encoding='utf-8')

    def get_inference_roi(self):
        """Load the crop next to the active INI; no whole-frame fallback."""
        return InferenceROI.load(Path(self.config_file).with_name('inference_roi.json'))

    def get_paths(self):
        """Get all path configurations"""
        return {
            'model_path': self.config.get('PATHS', 'model_path'),
            'video_path': self.config.get('PATHS', 'video_path'),
            'validation_path': self.config.get('PATHS', 'validation_path'),
            'stack_light_path': self.config.get('PATHS', 'stack_light_path')
        }

    def get_video_source_config(self):
        """Get video source configuration"""
        return {
            'mode': self.config.get('VIDEO_SOURCE', 'mode'),
            'camera_index': self.config.getint('VIDEO_SOURCE', 'camera_index'),
            'camera_width': self.config.getint('VIDEO_SOURCE', 'camera_width'),
            'camera_height': self.config.getint('VIDEO_SOURCE', 'camera_height'),
            'camera_focus': self.config.getint('VIDEO_SOURCE', 'camera_focus'),
            'video_index': self.config.getint('VIDEO_SOURCE', 'video_index'),
            # TIS DFK camera settings
            'tis_dll_path': self.config.get('VIDEO_SOURCE', 'tis_dll_path', fallback='./tisgrabber_x64.dll'),
            'tis_device_name': self.config.get('VIDEO_SOURCE', 'tis_device_name', fallback='DFK 33UX178'),
            'tis_camera_index': self.config.getint('VIDEO_SOURCE', 'tis_camera_index', fallback=-1)
        }

    def get_yolo_config(self):
        """Get YOLO configuration"""
        return {
            'confidence_threshold': self.config.getfloat('YOLO_SETTINGS', 'confidence_threshold'),
            'iou_threshold': self.config.getfloat('YOLO_SETTINGS', 'iou_threshold'),
            'max_detections': self.config.getint('YOLO_SETTINGS', 'max_detections'),
            'use_half_precision': self.config.getboolean('YOLO_SETTINGS', 'use_half_precision'),
            'use_augmentation': self.config.getboolean('YOLO_SETTINGS', 'use_augmentation')
        }

    def get_detection_config(self):
        """Get detection configuration"""
        count_key = ('expected_case_count' if self.config.has_option(
            'DETECTION_SETTINGS', 'expected_case_count') else 'expected_module_count')
        options = {
            'fps': self.config.getint('DETECTION_SETTINGS', 'fps'),
            'stop_threshold_seconds': self.config.getfloat('DETECTION_SETTINGS', 'stop_threshold_seconds'),
            'distance_threshold_pixels': self.config.getint('DETECTION_SETTINGS', 'distance_threshold_pixels'),
            'expected_case_count': self.config.getint(
                'DETECTION_SETTINGS', count_key, fallback=4),
            'kmeans_clusters': self.config.getint('DETECTION_SETTINGS', 'kmeans_clusters'),
            'kmeans_rounds': self.config.getint('DETECTION_SETTINGS', 'kmeans_rounds'),
            # Tolerance mechanism for detection box flickering
            'absent_tolerance_frames': self.config.getint('DETECTION_SETTINGS', 'absent_tolerance_frames', fallback=5),
            'result_hold_min_seconds': self.config.getfloat('DETECTION_SETTINGS', 'result_hold_min_seconds', fallback=2.0)
        }

        for key, default in (('case_pass_consecutive_frames', 10),
                             ('case_fail_consecutive_frames', 5)):
            value = self.config.getint('DETECTION_SETTINGS', key, fallback=default)
            if value <= 0:
                raise ValueError(f'DETECTION_SETTINGS {key} must be a positive integer')
            options[key] = value
        return options

    def get_display_config(self):
        """Get display configuration"""
        return {
            'show_crosshair': self.config.getboolean('DISPLAY_SETTINGS', 'show_crosshair'),
            'show_timestamp': self.config.getboolean('DISPLAY_SETTINGS', 'show_timestamp'),
            'show_status': self.config.getboolean('DISPLAY_SETTINGS', 'show_status'),
            'window_name': self.config.get('DISPLAY_SETTINGS', 'window_name'),
            'auto_fit_screen': self.config.getboolean('DISPLAY_SETTINGS', 'auto_fit_screen', fallback=True),
            'max_scale': self.config.getfloat('DISPLAY_SETTINGS', 'max_scale', fallback=1.0),
            'min_scale': self.config.getfloat('DISPLAY_SETTINGS', 'min_scale', fallback=0.1)
        }

    def get_case_crop_config(self):
        """Optional decision crops; older configurations keep saving disabled."""
        return {
            'enabled': self.config.getboolean('CASE_CROP_SAVING', 'enabled', fallback=False),
            'output_path': self.config.get('CASE_CROP_SAVING', 'output_path', fallback='./case_crops/'),
        }

    def get_camera_health_config(self):
        """Read the allowed carrier-center ROI and scalable visual alert settings."""
        defaults = {
            'roi_x_ratio': 0.25, 'roi_y_ratio': 0.25,
            'roi_w_ratio': 0.50, 'roi_h_ratio': 0.50,
            'alert_blink_interval': 0.5, 'alert_font_scale_ratio': 0.04,
            'alert_border_thickness_ratio': 0.012,
        }
        options = {key: self.config.getfloat('CAMERA_HEALTH', key, fallback=value)
                   for key, value in defaults.items()}
        for key, value in options.items():
            if not math.isfinite(value) or value < 0 or (value == 0 and key not in (
                    'roi_x_ratio', 'roi_y_ratio')):
                raise ValueError(f'CAMERA_HEALTH {key} must be finite and positive (ROI origins may be zero)')
        for axis, size in (('x', 'w'), ('y', 'h')):
            if options[f'roi_{axis}_ratio'] + options[f'roi_{size}_ratio'] > 1:
                raise ValueError('CAMERA_HEALTH ROI must fit inside the frame')
        return options

    def get_device(self):
        """Get device configuration"""
        return self.config.get('DEVICE_SETTINGS', 'device')

    def get_light_tower_config(self):
        """Get light tower configuration.

        ``usb`` keeps the physical PATLITE behavior. ``mock`` only reports
        requested statuses in the terminal, without launching the USB helper.
        The fallback preserves compatibility with existing configuration files.
        """
        options = {
            'mode': self.config.get('LIGHT_TOWER', 'mode', fallback='usb').strip().lower(),
            'command_timeout_seconds': self.config.getfloat('LIGHT_TOWER', 'command_timeout_seconds', fallback=3),
            'retry_initial_seconds': self.config.getfloat('LIGHT_TOWER', 'retry_initial_seconds', fallback=1),
            'retry_max_seconds': self.config.getfloat('LIGHT_TOWER', 'retry_max_seconds', fallback=30),
            'max_attempts': self.config.getint('LIGHT_TOWER', 'max_attempts', fallback=5),
            'refresh_seconds': self.config.getfloat('LIGHT_TOWER', 'refresh_seconds', fallback=30),
            'reset_after_command': self.config.getboolean('LIGHT_TOWER', 'reset_after_command', fallback=True),
            'log_path': self.config.get('LIGHT_TOWER', 'log_path', fallback='./logs/tower.jsonl'),
        }
        if options['mode'] not in ('usb', 'mock'):
            raise ValueError('LIGHT_TOWER mode must be usb or mock')
        import math
        for key in ('command_timeout_seconds', 'retry_initial_seconds', 'retry_max_seconds', 'refresh_seconds', 'max_attempts'):
            if not math.isfinite(options[key]) or options[key] <= 0:
                raise ValueError(f'LIGHT_TOWER {key} must be finite and positive')
        return options

    def get_classes(self):
        """Get model class IDs, accepting sealant as the legacy case name."""
        classes = {}
        for name, index in self.config['CLASSES'].items():
            classes[name] = int(index)
        if 'case' not in classes and 'sealant' in classes:
            classes['case'] = classes['sealant']
        if 'case' not in classes or 'carrier' not in classes:
            raise ValueError('CLASSES must define case (or legacy sealant) and carrier')
        if classes['case'] == classes['carrier']:
            raise ValueError('CLASSES case and carrier must have different IDs')
        return classes

    def get_logging_config(self):
        """Get logging configuration"""
        return {
            'log_level': self.config.get('LOGGING_SETTINGS', 'log_level', fallback='INFO'),
            'save_detection_log': self.config.getboolean('LOGGING_SETTINGS', 'save_detection_log', fallback=True),
            'log_file_path': self.config.get('LOGGING_SETTINGS', 'log_file_path', fallback='./logs/detection.log')
        }

    def get_recording_config(self):
        """Get recording configuration"""
        return {
            'enable_recording': self.config.getboolean('RECORDING_SETTINGS', 'enable_recording', fallback=False),
            'record_type': self.config.get('RECORDING_SETTINGS', 'record_type', fallback='inference'),
            'output_path': self.config.get('RECORDING_SETTINGS', 'output_path', fallback='./recordings/'),
            'video_format': self.config.get('RECORDING_SETTINGS', 'video_format', fallback='mp4'),
            'fps': self.config.getint('RECORDING_SETTINGS', 'fps', fallback=15),
            'quality': self.config.get('RECORDING_SETTINGS', 'quality', fallback='medium')
        }

    def get_rejudge_config(self):
        """Get rejudge/manual review configuration (V6.0)"""
        return {
            'review_window_padding': self.config.getint('REJUDGE_SETTINGS', 'review_window_padding', fallback=50),
            'review_border_thickness': self.config.getint('REJUDGE_SETTINGS', 'review_border_thickness', fallback=3)
        }
