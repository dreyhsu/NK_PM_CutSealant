from types import SimpleNamespace
from unittest.mock import Mock

import cv2
import numpy as np
import pytest


def test_review_toggle_reopen_and_cancel(legacy, frame):
    review = legacy.ManualReviewWindow()
    box = np.array([10, 10, 50, 50])
    key = "10_10_50_50"
    assert review.open(frame, [box], {key: False}, {key: .9}) is True
    assert review._get_border_color(0) == (0, 255, 0)
    x, y, _, _ = review.sealant_rects[0]
    review._mouse_callback(cv2.EVENT_LBUTTONDOWN, x + 1, y + 1, 0, None)
    assert review._get_border_color(0) == (0, 255, 255)
    review._on_confirm()
    assert review.update() == "confirm"
    states = review.get_results()
    assert states[0]["user_agrees"] is False
    assert review.open(frame, [box], {key: False}, {}, previous_states=states)
    assert review.sealant_states[0]["user_agrees"] is False
    review._on_cancel()
    assert review.update() == "cancel"
    assert review.get_results()[0]["user_agrees"] is True


@pytest.mark.parametrize("key,expected", [(13, "confirm"), (27, "cancel")])
def test_review_keyboard(legacy, frame, monkeypatch, key, expected):
    review = legacy.ManualReviewWindow()
    review.open(frame, [np.array([0, 0, 20, 20])], {}, {})
    monkeypatch.setattr(cv2, "waitKey", lambda _: key)
    assert review.update() == expected


def test_review_empty_and_invalid_crops(legacy, frame):
    review = legacy.ManualReviewWindow()
    assert review.open(frame, [], {}, {}) is False
    assert review.open(frame, [np.array([0, 0, 0, 0])], {}, {}) is False


def test_review_confirm_updates_correct_source_without_changing_light(processor):
    processor.pending_review = {"carrier_id": 1, "sealants": [{"user_agrees": True}, {"user_agrees": True}]}
    processor.manual_review_window = Mock()
    processor.manual_review_window.get_results.return_value = [{"source_index": 1, "user_agrees": False}]
    processor.last_light_status = "Pass"
    processor._handle_review_complete("confirm")
    assert [s["user_agrees"] for s in processor.pending_review["sealants"]] == [True, False]
    assert processor.last_light_status == "Pass"


def test_open_review_on_departure_resets_agreement_before_write(processor):
    processor.pending_review = {"carrier_id": 1, "sealants": [{"user_agrees": False}]}
    processor.manual_review_window = Mock()
    processor.manual_review_window.is_open = True
    processor.rejudge_logger = Mock()
    processor._write_pending_review_log()
    assert processor.pending_review["sealants"][0]["user_agrees"] is True
    processor.manual_review_window.close.assert_called_once()
    processor.rejudge_logger.write_records.assert_called_once()


def test_review_not_opened_outside_hold(processor):
    processor._open_manual_review()
    assert processor.manual_review_window is None


def test_box_matching_and_frame_colors(processor, frame, monkeypatch):
    box = np.array([10, 10, 40, 40])
    processor.detection_state = "RESULT_HOLD"
    processor.sealant_detection_results = {"10_10_40_40": True}
    draw = Mock()
    monkeypatch.setattr(cv2, "rectangle", draw)
    processor._draw_detections(frame, [box], [1], np.array([0, 0, 100, 100]))
    assert draw.call_args.args[3] == (0, 0, 255)
    assert processor._find_matching_box([11, 10, 40, 40], processor.sealant_detection_results) == "10_10_40_40"
    assert processor._calculate_iou([50, 50, 60, 60], box) == 0


def test_screenshot_and_fit_display(processor, frame, monkeypatch, tmp_path):
    processor._take_screenshot(frame)
    images = list((tmp_path / "screenshots").glob("*.jpg"))
    assert len(images) == 1
    assert cv2.imread(str(images[0])).shape == frame.shape
    processor.cached_screen_size = (100, 100)
    processor.display_config = {"auto_fit_screen": True, "max_scale": 1, "min_scale": .1}
    show = Mock()
    monkeypatch.setattr(cv2, "imshow", show)
    processor._display_frame(frame, "test")
    assert show.call_args.args[1].shape[:2] == (85, 85)


@pytest.mark.parametrize("quality,size", [("low", (100, 50)), ("medium", (142, 72)), ("high", (202, 102))])
def test_recording_dimensions(legacy, quality, size):
    recorder = legacy.VideoRecorder({"enable_recording": False, "quality": quality}, (103, 203, 3), "test")
    assert recorder._calculate_recording_dimensions() == size


@pytest.mark.parametrize("opened", [True, False])
def test_recording_writer_lifecycle(legacy, frame, tmp_path, monkeypatch, opened):
    writer = Mock()
    writer.isOpened.return_value = opened
    monkeypatch.setattr(cv2, "VideoWriter", Mock(return_value=writer))
    logger = Mock()
    recorder = legacy.VideoRecorder({"enable_recording": True, "quality": "high", "video_format": "avi", "fps": 15,
                                    "record_type": "raw", "output_path": str(tmp_path) + "/"}, frame.shape, "test", logger)
    assert recorder.is_recording is opened
    recorder.record_frame(frame)
    assert writer.write.call_count == int(opened)
    recorder.stop_recording()
    assert writer.release.call_count == int(opened)
    if not opened:
        assert logger.log_error.call_args.args[0] == "video_writer_init_error"


def test_heartbeat_boundary_and_fields(processor):
    now = [100.0]
    processor.clock = SimpleNamespace(time=lambda: now[0])
    processor.start_time = processor.last_heartbeat_time = 100
    now[0] = 159.99
    processor._log_heartbeat()
    processor.logger.log_system_event.assert_not_called()
    now[0] = 160
    processor._log_heartbeat()
    event, fields = processor.logger.log_system_event.call_args.args
    assert event == "heartbeat"
    assert fields["uptime_seconds"] == 60
    assert fields["detection_state"] == "WAITING"
    assert "memory_usage" in fields
