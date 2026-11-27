import json
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest


@pytest.mark.parametrize('slots', [(0,), (1, 3), (0, 2, 3), (3, 1, 0, 2)])
def test_each_discovered_case_needs_ten_consecutive_passes(rig, slots):
    for _ in range(9):
        workflow = rig.feed(slots)
    assert not workflow.ready
    assert set(workflow.positions) == set(slots)
    assert all(c.pass_streak == 9 for c in workflow.positions.values())
    assert rig.p.detection_result is None
    assert rig.state[0] == 'Tracking'
    rig.feed(slots)
    assert workflow.ready
    assert rig.p.detection_state == 'RESULT_HOLD'
    assert rig.state[0] == 'Pass'
    assert all(c.decision_source == 'automatic' for c in workflow.positions.values())
    rig.p.statistics_logger.log_detection.assert_not_called()


def test_zero_cases_cannot_pass_or_close(rig):
    for _ in range(12):
        workflow = rig.feed(())
    assert workflow.active and not workflow.ready
    rig.absent(10, 10)
    assert workflow.active
    assert 'Cycle incomplete' in workflow.panel.phase


def test_covering_keeps_pass_and_newly_discovered_position_withdraws_ready(rig):
    for _ in range(10):
        workflow = rig.feed((3,))
    count = rig.judge.call_count
    for _ in range(4):
        rig.feed(())
    assert workflow.ready and rig.judge.call_count == count
    rig.feed((0,))
    assert not workflow.ready and workflow.positions[3].passed
    assert workflow.positions[0].pass_streak == 1
    assert rig.state[0] == 'Tracking'
    for _ in range(9):
        rig.feed((0,))
    assert workflow.ready
    rig.feed((3, 0))
    assert rig.judge.call_count == count + 10


@pytest.mark.parametrize('interruption', ['NG', 'Unavailable', 'missing'])
def test_interruption_resets_pass_streak_without_latching_failure(rig, interruption):
    for _ in range(9):
        workflow = rig.feed((0,))
    rig.judge.return_value = interruption
    rig.feed(()) if interruption == 'missing' else rig.feed((0,))
    case = workflow.positions[0]
    assert case.pass_streak == 0
    assert case.fail_streak == int(interruption == 'NG')
    assert not case.needs_review
    rig.judge.return_value = 'Pass'
    rig.feed((0,))
    assert case.pass_streak == 1


def test_fifth_ng_queues_review_and_cannot_be_cleared_automatically(rig):
    rig.judge.return_value = 'NG'
    for _ in range(4):
        workflow = rig.feed((2,))
    assert not workflow.review_queue
    rig.feed((2,))
    case = workflow.positions[2]
    assert case.status == 'REVIEW' and workflow.review_queue == {2}
    assert rig.state[0] == 'NG'
    evidence = case.crop.copy()
    calls = rig.judge.call_count
    rig.judge.return_value = 'Pass'
    for _ in range(10):
        rig.feed((2,), image=np.full_like(rig.p.current_raw_frame, 255))
    assert case.status == 'REVIEW' and not workflow.ready
    assert rig.judge.call_count == calls
    assert np.array_equal(case.crop, evidence)


def test_thresholds_are_configurable(rig):
    rig.p.detection_config.update(case_pass_consecutive_frames=2, case_fail_consecutive_frames=3)
    workflow = rig.feed((0,))
    assert not workflow.ready
    rig.feed((0,))
    assert workflow.ready
    rig.judge.return_value = 'NG'
    rig.feed((1,))
    rig.feed((1,))
    assert not workflow.positions[1].needs_review
    rig.feed((1,))
    assert workflow.positions[1].needs_review


def test_ng_streak_resets_on_missing_or_pass(rig):
    rig.judge.return_value = 'NG'
    for _ in range(4):
        workflow = rig.feed((0,))
    rig.feed(())
    assert workflow.positions[0].fail_streak == 0
    rig.feed((0,))
    rig.judge.return_value = 'Pass'
    rig.feed((0,))
    assert workflow.positions[0].fail_streak == 0


def test_duplicate_candidates_rank_by_confidence_then_smaller_area(rig):
    extras = ([12, 112, 80, 148], [22, 122, 40, 135], [0, 0, 8, 8])
    workflow = rig.feed((0,), extras=extras, confidences=[.8, .9, .9, .99])
    assert list(workflow.positions) == [0]
    assert workflow.positions[0].bbox == [22, 122, 40, 135]
    assert workflow.positions[0].pass_streak == 1
    rig.judge.assert_called_once()
    assert np.allclose(workflow.positions[0].relative_bbox, [12/180, 12/80, 30/180, 25/80])


def test_relative_assignment_survives_carrier_translation_and_box_jitter(rig):
    workflow = rig.feed((3,))
    rig.feed((3,), shift=2)
    assert list(workflow.positions) == [3]
    assert workflow.positions[3].pass_streak == 2
    rig.feed((), shift=3, extras=([124, 161, 154, 181],))
    assert list(workflow.positions) == [3]
    assert workflow.positions[3].pass_streak == 3


@pytest.mark.parametrize('box,expected', [([45, 45, 55, 55], 3),
    ([10, 10, 20, 20], 0), ([60, 10, 80, 20], 1),
    ([10, 60, 20, 80], 2), ([100, 10, 120, 20], None)])
def test_quadrant_center_boundaries(rig, box, expected):
    assert rig.p.workflow.position_index([0, 0, 100, 100], box) == expected


def test_same_processed_frame_does_not_advance_streak(rig):
    workflow = rig.feed((0,))
    workflow.last_processed_frame = rig.p.frame_count + 1
    rig.feed((0,))
    assert workflow.positions[0].pass_streak == 1
    assert rig.judge.call_count == 1


def test_motion_and_carrier_loss_reset_unfinished_streaks(rig):
    workflow = rig.feed((0,))
    rig.feed((0,))
    rig.feed((0,), shift=15)
    assert workflow.positions[0].pass_streak == 0
    assert workflow.needs_stability
    for _ in range(15):
        rig.feed((0,), shift=15)
    assert workflow.positions[0].pass_streak == 0
    rig.feed((0,), shift=15)
    assert workflow.positions[0].pass_streak == 1
    rig.absent()
    assert workflow.positions[0].pass_streak == 0


@pytest.mark.parametrize('absent,elapsed,release', [(4, 5, False), (5, 1.99, False), (5, 2, True)])
def test_only_ready_departure_closes_after_both_tolerances(rig, absent, elapsed, release):
    for _ in range(10):
        workflow = rig.feed((1, 3))
    rig.absent(absent, elapsed)
    assert workflow.active != release
    assert rig.p.statistics_logger.log_detection.call_count == int(release)
    if release:
        rig.p.statistics_logger.log_detection.assert_called_once_with(False, 2, 0, 2, 2000)
        rows = [json.loads(line) for line in open(workflow.cycle_logger.path)]
        assert len(rows) == 1 and rows[0]['outcome'] == 'Completed'
        assert [c['position_id'] for c in rows[0]['positions']] == ['02', '04']
        assert workflow.panel.complete


def test_early_departure_holds_until_termination_and_restarts_same_carrier(rig):
    workflow = rig.feed((1,))
    rig.absent(8, 10)
    assert workflow.active and workflow.positions[1].pass_streak == 0
    assert 'Cycle incomplete' in workflow.panel.phase
    rig.state[:] = rig.p._handle_key(ord('X'), *rig.state)[:3]
    assert not workflow.active and not workflow.positions
    assert rig.p.detection_state == 'WAITING'
    rows = [json.loads(line) for line in open(workflow.cycle_logger.path)]
    assert len(rows) == 1 and rows[0]['outcome'] == 'Terminated'
    rig.p.statistics_logger.log_detection.assert_not_called()
    for _ in range(16):
        rig.feed((1,))
    assert not workflow.active
    rig.feed((1,))
    assert workflow.active and rig.p.carrier_count == 2
    assert workflow.positions[1].pass_streak == 1


def test_shutdown_finalizes_interrupted_once(rig):
    workflow = rig.feed((0, 2))
    rig.p.stop()
    rig.p.stop()
    workflow.finalize('Interrupted')
    rows = [json.loads(line) for line in open(workflow.cycle_logger.path)]
    assert len(rows) == 1 and rows[0]['outcome'] == 'Interrupted'
    assert len(rows[0]['positions']) == 2
    rig.p.rejudge_logger.write_records.assert_called_once()
    rig.p.statistics_logger.log_detection.assert_not_called()


def test_trigger_boundary_and_upper_half_rejection(rig):
    rig.state[1] = 14
    workflow = rig.feed((0,))
    assert not workflow.active and rig.state[1] == 15
    rig.feed((0,))
    assert workflow.active and workflow.positions[0].pass_streak == 1
    rig.p.workflow.terminate(rig.state[0])
    rig.state[:] = ['Standby', 20, [100, 50]]
    rig.feed((0,))  # Carrier center moved: restart stability rather than trigger.
    assert not workflow.active and rig.state[1] == 0


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


def test_statistics_reset_does_not_skip_next_distinct_frame(rig):
    workflow = rig.feed((0,))
    assert rig.p.frame_count == 1
    rig.p._reset_statistics()
    rig.feed((0,))
    assert rig.p.frame_count == 1
    assert workflow.positions[0].pass_streak == 2


def test_logging_failure_does_not_block_termination_or_duplicate_outcome(rig):
    workflow = rig.feed((0,))
    workflow.cycle_logger = Mock()
    workflow.cycle_logger.write.side_effect = OSError('disk unavailable')
    rig.p._handle_key(ord('x'), *rig.state)
    assert not workflow.active and not workflow.positions
    assert rig.p.detection_state == 'WAITING'
    rig.p.stop()
    workflow.cycle_logger.write.assert_called_once()
    rig.p.rejudge_logger.write_records.assert_called_once()
    rig.p.logger.log_error.assert_called_once()


def test_terminate_does_not_reset_camera_error(rig):
    rig.p.detection_state = 'CAMERA_ERROR'
    rig.p.camera_error_start_time = 90
    result = rig.p._handle_key(ord('X'), 'Standby', 20, [100, 150])
    assert result[:3] == ('Standby', 0, [0, 0])
    assert rig.p.detection_state == 'CAMERA_ERROR'
    assert rig.p.camera_error_start_time == 90
