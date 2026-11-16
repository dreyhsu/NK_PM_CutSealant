import time
from unittest.mock import Mock

import numpy as np
import pytest


@pytest.mark.parametrize("broken,expected", [(False, "Pass"), (True, "NG")])
def test_trigger_boundary_and_one_inspection(processor, frame, detection, monkeypatch, broken, expected):
    boxes, categories, scores = detection
    monkeypatch.setattr(processor, "_cut_sealant_detection", Mock(return_value=broken))
    monkeypatch.setattr(processor, "_call_light_once", Mock(side_effect=lambda s: s))
    monkeypatch.setattr(processor, "_save_img_data", Mock())
    processor.statistics_logger = Mock()
    args = (frame, frame.copy(), boxes, categories, list(boxes[1:]), scores)
    result = processor._handle_objects_present(*args, "Standby", 14, [100, 150])
    assert processor.detection_state == "WAITING"  # exactly 15 / 15 seconds is not > 1
    assert result[1] == 15
    result = processor._handle_objects_present(*args, *result)
    assert processor.detection_state == "RESULT_HOLD"
    assert processor.detection_result == expected
    processor._call_light_once.assert_called_once_with(expected)
    assert processor.pending_review["carrier_id"] == 1
    assert len(processor.pending_review["sealants"]) == 4
    assert processor.current_raw_frame is not frame
    processor._handle_objects_present(*args, *result)
    processor._cut_sealant_detection.assert_called_once()
    processor.statistics_logger.log_detection.assert_called_once()


@pytest.mark.parametrize("previous,carrier", [([0, 0], [10, 110, 190, 190]), ([100, 50], [10, 10, 190, 90])])
def test_motion_and_upper_half_do_not_trigger(processor, frame, detection, monkeypatch, previous, carrier):
    boxes, categories, scores = detection
    boxes[0] = carrier
    judge = Mock()
    monkeypatch.setattr(processor, "_cut_sealant_detection", judge)
    result = processor._check_detection_trigger(frame, frame, boxes, categories, list(boxes[1:]), scores, "Standby", 20, previous)
    assert result[1] == 0
    assert processor.detection_state == "WAITING"
    judge.assert_not_called()


@pytest.mark.parametrize("absent,elapsed,release", [(3, 5, False), (4, 1.99, False), (4, 2, True)])
def test_absence_and_hold_boundaries(processor, monkeypatch, absent, elapsed, release):
    monkeypatch.setattr(time, "time", lambda: 100 + elapsed)
    processor.detection_state = "RESULT_HOLD"
    processor.result_hold_start_time = 100
    processor.absent_frame_count = absent
    processor.pending_review = {"carrier_id": 1, "sealants": []}
    write = Mock()
    light = Mock(return_value="Standby")
    monkeypatch.setattr(processor, "_write_pending_review_log", write)
    monkeypatch.setattr(processor, "_call_light_once", light)
    processor._handle_objects_absent("Pass", 0, [100, 150])
    assert processor.detection_state == ("WAITING" if release else "RESULT_HOLD")
    assert write.call_count == int(release)
    assert light.call_count == int(release)
    if release:
        assert processor.pending_review is None


def test_reappearance_clears_absence_counter(processor, frame, detection):
    processor.detection_state = "RESULT_HOLD"
    processor.absent_frame_count = 4
    boxes, categories, scores = detection
    processor._handle_objects_present(frame, frame, boxes, categories, list(boxes[1:]), scores, "Pass", 0, [100, 150])
    assert processor.absent_frame_count == 0


def test_pending_review_written_only_once(processor):
    processor.pending_review = {"carrier_id": 1, "sealants": []}
    processor.rejudge_logger = Mock()
    processor._write_pending_review_log()
    processor._write_pending_review_log()
    processor.rejudge_logger.write_records.assert_called_once()


def test_stop_flushes_pending_and_releases_resources(processor, monkeypatch):
    processor.pending_review = {"carrier_id": 1, "sealants": []}
    processor.rejudge_logger = Mock()
    processor.cap = Mock()
    processor.video_recorder = Mock()
    processor.statistics_logger = Mock()
    processor.capture_thread = Mock()
    processor.yolo_thread = Mock()
    processor.stop()
    assert processor.stop_event.is_set()
    processor.rejudge_logger.write_records.assert_called_once()
    processor.cap.release.assert_called_once()
    processor.capture_thread.join.assert_called_once_with(timeout=2.0)
    processor.yolo_thread.join.assert_called_once_with(timeout=2.0)
    processor.video_recorder.stop_recording.assert_called_once()
    processor.statistics_logger.close.assert_called_once()
