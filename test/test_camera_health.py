"""Camera alignment checks run without a camera, model, USB, or GUI."""
from types import SimpleNamespace
from unittest.mock import Mock

import cv2
import numpy as np
import pytest

from cutsealant.config import ConfigManager


@pytest.mark.parametrize('x,allowed', [(49, False), (50, True), (100, True),
                                      (150, True), (151, False)])
def test_roi_boundary_blocks_inspection_and_production_records(
        processor, frame, detection, monkeypatch, x, allowed):
    boxes, categories, scores = detection
    boxes[0] = [x - 10, 130, x + 10, 170]
    boxes[1:] = [x - 8, 132, x - 2, 140]
    judge = Mock(return_value='Pass')
    save = Mock()
    light = Mock(side_effect=lambda status: status)
    monkeypatch.setattr(processor.inspection, 'observe', judge)
    monkeypatch.setattr(processor, '_save_img_data', save)
    monkeypatch.setattr(processor, '_call_light_once', light)
    processor.statistics_logger = Mock()
    result = processor._check_detection_trigger(
        frame, frame, boxes, categories, list(boxes[1:]), scores,
        'Tracking', 15, [x, 150])
    assert processor.detection_state == ('INSPECTING' if allowed else 'CAMERA_ERROR')
    assert result[0] == ('Tracking' if allowed else 'Standby')
    assert judge.call_count == int(allowed)
    save.assert_not_called()
    processor.statistics_logger.log_detection.assert_not_called()
    assert processor.carrier_count == int(allowed)
    if not allowed:
        processor.logger.log_system_event.assert_called_with('camera_misalignment_detected', {
            'centroid': [x, 150], 'roi': {'x': 50, 'y': 50, 'w': 100, 'h': 100},
        })


def test_error_latches_until_operator_reset_then_rechecks(processor, frame, detection, monkeypatch):
    boxes, categories, scores = detection
    processor.clock = SimpleNamespace(time=lambda: 110.0)
    processor.detection_state = 'CAMERA_ERROR'
    processor.camera_error_start_time = 100
    judge = Mock(return_value='Pass')
    monkeypatch.setattr(processor.inspection, 'observe', judge)
    monkeypatch.setattr(processor, '_save_img_data', Mock())
    monkeypatch.setattr(processor, '_call_light_once', Mock(side_effect=lambda status: status))
    args = (frame, frame, boxes, categories, list(boxes[1:]), scores)
    for _ in range(10):
        processor._handle_objects_absent('Standby', 0, [100, 150])
    processor._handle_objects_present(*args, 'Standby', 0, [100, 150])
    assert processor.detection_state == 'CAMERA_ERROR'
    judge.assert_not_called()
    assert processor.workflow.reset_camera_error('Standby') == 'Standby'
    assert processor.detection_state == 'WAITING'
    processor.logger.log_system_event.assert_called_with('camera_error_reset_by_operator', {
        'error_duration_seconds': 10.0,
    })
    # Reset is only a release, not a bypass of the ROI check.
    processor.camera_health_config = {'roi_x_ratio': 0, 'roi_w_ratio': 0.1}
    processor._check_detection_trigger(*args, 'Standby', 15, [100, 150])
    assert processor.detection_state == 'CAMERA_ERROR'
    judge.assert_not_called()


def test_reset_outside_error_does_not_change_result_or_light(processor, monkeypatch):
    processor.detection_state = 'RESULT_HOLD'
    light = Mock()
    monkeypatch.setattr(processor, '_call_light_once', light)
    assert processor.workflow.reset_camera_error('Pass') == 'Pass'
    assert processor.detection_state == 'RESULT_HOLD'
    light.assert_not_called()


@pytest.mark.parametrize('key', ['c', 'C'])
def test_main_loop_reset_key_requires_new_stop_period(processor, frame, detection, monkeypatch, key):
    boxes, categories, scores = detection
    monkeypatch.setattr(processor, 'initialize', Mock(return_value=True))
    monkeypatch.setattr(processor, 'start_threads', Mock())
    monkeypatch.setattr(processor, '_call_light_once', Mock(side_effect=lambda s: s))
    processor.display_config = {'window_name': 'test'}
    processor.recording_config = {'record_type': 'raw'}
    processor.config_manager = Mock()
    for _ in range(2):
        processor.result_queue.put((frame.copy(), boxes, scores, categories))
    observations = []

    def handle(*args):
        observations.append((processor.detection_state, args[-2], args[-1]))
        if len(observations) == 1:
            processor.detection_state = 'CAMERA_ERROR'
            processor.camera_error_start_time = processor.clock.time()
        return 'Standby', 99, [99, 99]

    monkeypatch.setattr(processor, '_handle_objects_present', handle)
    keys = iter([ord(key), ord('q')])
    monkeypatch.setattr(cv2, 'waitKey', lambda _: next(keys))
    assert processor.run()
    assert observations[1] == ('WAITING', 0, [0, 0])
    assert processor._call_light_once.call_args_list[-1].args == ('Tracking',)


def test_roi_checks_selected_largest_carrier(processor, frame, detection, monkeypatch):
    boxes, categories, scores = detection
    boxes = np.vstack(([0, 130, 20, 170], boxes))
    categories = np.insert(categories, 0, 0)
    scores = np.insert(scores, 0, 0.99)
    judge = Mock(return_value='Pass')
    monkeypatch.setattr(processor.inspection, 'observe', judge)
    monkeypatch.setattr(processor, '_save_img_data', Mock())
    monkeypatch.setattr(processor, '_call_light_once', Mock(return_value='Pass'))
    processor.detection_config['case_pass_consecutive_frames'] = 1
    processor._check_detection_trigger(frame, frame, boxes, categories, list(boxes[2:]),
                                       scores, 'Tracking', 15, [100, 150])
    assert processor.detection_state == 'RESULT_HOLD'
    assert judge.call_count == 2


@pytest.mark.parametrize('size', [(1920, 1080), (800, 600), (200, 200)])
def test_blink_overlay_scales_and_keeps_reset_hint(processor, monkeypatch, size):
    w, h = size
    processor.detection_state = 'CAMERA_ERROR'
    processor.camera_error_start_time = 100
    now = [100.1]
    processor.clock = SimpleNamespace(time=lambda: now[0])
    processor.fit_screen_mode = False
    image = np.zeros((h, w, 3), np.uint8)
    processor.display._draw_camera_error_overlay(image)
    assert np.array_equal(image[0, w // 2], [0, 0, 255])
    assert image[h // 3:2 * h // 3, w // 5:4 * w // 5].any()
    now[0] = 100.6
    image.fill(0)
    processor.display._draw_camera_error_overlay(image)
    assert not image.any()
    text = Mock()
    monkeypatch.setattr(cv2, 'putText', text)
    processor.display._display_frame(image, 'test')
    assert '[C] Reset camera alignment error  |  [Q] Quit' in [c.args[1] for c in text.call_args_list]


def test_older_ini_defaults_and_case_setting_migration(tmp_path):
    path = tmp_path / 'old.ini'
    path.write_text('[DETECTION_SETTINGS]\nexpected_module_count = 6\n')
    config = ConfigManager(path)
    assert config.get_camera_health_config()['roi_w_ratio'] == 0.5
    # Supply the other detection defaults without replacing the old setting.
    for key, value in {'fps': '15', 'stop_threshold_seconds': '1',
                       'distance_threshold_pixels': '10', 'kmeans_clusters': '3',
                       'kmeans_rounds': '10'}.items():
        config.config['DETECTION_SETTINGS'][key] = value
    assert config.get_detection_config()['expected_case_count'] == 6
    config.config['DETECTION_SETTINGS']['expected_case_count'] = '3'
    assert config.get_detection_config()['expected_case_count'] == 3


@pytest.mark.parametrize('key,value', [('alert_blink_interval', '0'),
    ('alert_font_scale_ratio', 'nan'), ('roi_x_ratio', '-0.1'),
    ('roi_w_ratio', '0'), ('roi_y_ratio', '0.9')])
def test_invalid_health_settings_fail_at_startup(tmp_path, key, value):
    config = ConfigManager(tmp_path / 'config.ini')
    config.config['CAMERA_HEALTH'][key] = value
    with pytest.raises(ValueError, match='CAMERA_HEALTH'):
        config.get_camera_health_config()
