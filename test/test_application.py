import importlib
from pathlib import Path
import runpy
import subprocess
import sys
from unittest.mock import Mock

import cv2
import numpy as np


def dispatch_synchronously(processor, monkeypatch):
    """Make workflow assertions deterministic; worker behavior has separate tests."""
    request = processor.tower._call_light_once
    monkeypatch.setattr(processor.tower, 'start', Mock())
    def dispatch(status):
        result = request(status)
        processor.tower.service_once()
        return result
    monkeypatch.setattr(processor.tower, '_call_light_once', dispatch)


def test_main_launcher_uses_refactored_entrypoint(monkeypatch):
    import cutsealant.app as app
    main = Mock(return_value=0)
    monkeypatch.setattr(app, "main", main)
    root = Path(__file__).resolve().parents[1]
    for filename in ("main.py",):
        try:
            runpy.run_path(str(root / filename), run_name="__main__")
        except SystemExit as result:
            assert result.code == 0
        else:
            raise AssertionError("Launcher must return the entrypoint's exit code")
    main.assert_called_once_with()


def test_import_has_no_external_side_effects(tmp_path):
    root = Path(__file__).resolve().parents[1]
    script = r'''
import sys, pathlib, importlib, importlib.abc, pkgutil
sys.path.insert(0, sys.argv[1])
import cv2, subprocess, usb.core
def forbidden(*a, **kw):
    raise AssertionError("External I/O on import")
for name in ("VideoCapture", "VideoWriter", "namedWindow", "imshow"):
    setattr(cv2, name, forbidden)
subprocess.call = forbidden
usb.core.find = forbidden
class BlockModel(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, *args):
        if fullname.split(".")[0] in ("ultralytics", "torch"):
            raise AssertionError("Model library imported eagerly")
sys.meta_path.insert(0, BlockModel())
import cutsealant
for item in pkgutil.iter_modules(cutsealant.__path__, "cutsealant."):
    importlib.import_module(item.name)
import main
assert not list(pathlib.Path.cwd().iterdir())
'''
    result = subprocess.run([sys.executable, "-c", script, str(root)], cwd=tmp_path, capture_output=True, text=True, timeout=20)
    assert result.returncode == 0, result.stderr


def test_main_loop_standby_quit_and_cleanup(processor, frame, monkeypatch):
    dispatch_synchronously(processor, monkeypatch)
    monkeypatch.setattr(processor, "initialize", Mock(return_value=True))
    monkeypatch.setattr(processor, "start_threads", Mock())
    processor.display_config = {"window_name": "fake"}
    processor.config_manager = Mock()
    processor.config_manager.get_video_source_config.return_value = {"mode": "video"}
    processor.recording_config = {"record_type": "raw"}
    processor.result_queue.put((frame, np.empty((0, 4), np.int32), np.array([]), np.array([])))
    processor.cap = Mock()
    send = Mock(return_value=0)
    processor.command_runner = send
    monkeypatch.setattr(cv2, "waitKey", lambda _: ord("q"))
    assert processor.run() is True
    assert processor.frame_count == 1
    assert processor.stop_event.is_set()
    assert send.call_args.args[0] == [sys.executable, "LR6main.py", "2", "0", "1", "0", "0", "0"]
    processor.cap.release.assert_called_once()


def test_board_in_changes_static_yellow_to_tracking_blink(processor, frame, detection,
                                                          monkeypatch):
    dispatch_synchronously(processor, monkeypatch)
    boxes, categories, scores = detection
    monkeypatch.setattr(processor, 'initialize', Mock(return_value=True))
    monkeypatch.setattr(processor, 'start_threads', Mock())
    processor.display_config = {'window_name': 'fake'}
    processor.config_manager = Mock()
    processor.config_manager.get_video_source_config.return_value = {'mode': 'video'}
    processor.recording_config = {'record_type': 'raw'}
    processor.result_queue.put((frame, boxes, scores, categories))
    processor.cap = Mock()
    send = Mock(return_value=0)
    processor.command_runner = send
    monkeypatch.setattr(cv2, 'waitKey', lambda _: ord('q'))

    assert processor.run() is True
    assert [call.args[0][2:] for call in send.call_args_list] == [
        ['2', '0', '1', '0', '0', '0'],
        ['2', '0', '2', '0', '0', '0'],
    ]


def test_two_carrier_sequence_pass_then_manual_ng(rig, monkeypatch):
    for _ in range(10):
        workflow = rig.feed((0,))
    rig.absent(5, 2)
    assert not workflow.active
    rig.judge.return_value = 'NG'
    for _ in range(17 + 4):
        rig.feed((3,))
    assert rig.p.carrier_count == 2
    assert workflow.positions[3].needs_review
    rig.p.review.open_queued()
    assert rig.p.manual_review_window.is_open
    assert rig.p.manual_review_window.sealant_states[0]['position_id'] == '04'
    rig.p._handle_key(13, *rig.state)
    assert workflow.positions[3].status == 'NG'
    rig.absent(5, 3)
    assert workflow.active
    rig.p._handle_key(ord('x'), *rig.state)
    assert not workflow.active
    assert rig.p.rejudge_logger.write_records.call_count == 2
    rig.p.statistics_logger.log_detection.assert_called_once()


def test_inspection_snapshot_does_not_alias_mutable_state(processor, frame, monkeypatch):
    monkeypatch.setattr(processor.inspection, "_cut_sealant_detection", Mock(return_value=True))
    processor.sealant_detection_results = {"box": True}
    result = processor.inspection.inspect(frame, [])
    processor.sealant_detection_results.clear()
    assert result.is_ng is True
    assert result.by_box == {"box": True}


def test_entrypoint_exception_still_stops_processor(monkeypatch):
    import cutsealant.app as app
    processor = Mock()
    processor.run.side_effect = RuntimeError("fake main failure")
    monkeypatch.setattr(app, "SafeVideoProcessor", Mock(return_value=processor))
    assert app.main() == 1
    processor.stop.assert_called_once()


def test_window_close_stops_without_frames_and_cleanup_is_idempotent(processor, monkeypatch):
    monkeypatch.setattr(processor, 'initialize', Mock(return_value=True))
    monkeypatch.setattr(processor, 'start_threads', Mock())
    monkeypatch.setattr(cv2, 'getWindowProperty', lambda *args: 0)
    processor.display_config = {'window_name': 'fake'}
    processor.light_tower_config = {'mode': 'mock'}
    processor.cap = Mock()
    assert processor.run()
    processor.stop()
    assert processor.stop_event.is_set()
    assert not processor.tower._thread.is_alive()
    processor.cap.release.assert_called_once()
    processor.logger.close.assert_called_once()


def test_quit_works_without_frames(processor, monkeypatch):
    monkeypatch.setattr(processor, 'initialize', Mock(return_value=True))
    monkeypatch.setattr(processor, 'start_threads', Mock())
    monkeypatch.setattr(cv2, 'waitKey', lambda _: ord('q'))
    processor.display_config = {'window_name': 'fake'}
    processor.light_tower_config = {'mode': 'mock'}
    assert processor.run()
    assert not processor.tower._thread.is_alive()


def test_main_loop_single_key_pump_terminates_open_review_before_queued_frame(processor, frame, monkeypatch):
    import json
    monkeypatch.setattr(processor, 'initialize', Mock(return_value=True))
    monkeypatch.setattr(processor, 'start_threads', Mock())
    monkeypatch.setattr(processor, '_call_light_once', Mock(side_effect=lambda s: s))
    judge = Mock(return_value='NG')
    monkeypatch.setattr(processor.inspection, 'observe', judge)
    processor.display_config = {'window_name': 'fake'}
    processor.config_manager = Mock()
    processor.recording_config = {'record_type': 'raw'}
    processor.statistics_logger = Mock()
    processor.rejudge_logger = Mock()
    boxes = np.array([[10, 110, 190, 190], [120, 160, 150, 180]], np.int32)
    scores, categories = np.array([.99, .9]), np.array([0, 1])
    keys = iter([255]*21 + [ord('X'), ord('q')])
    observed = []

    def pump(_):
        key = next(keys)
        if key in (255, ord('X')):
            processor.result_queue.put((frame.copy(), boxes, scores, categories))
        if key == ord('X'):
            assert processor.manual_review_window.is_open
            processor.manual_review_window.sealant_states[0]['user_agrees'] = False
            processor.manual_review_window._on_confirm()
        observed.append(key)
        return key

    monkeypatch.setattr(cv2, 'waitKey', pump)
    assert processor.run()
    assert len(observed) == 23
    assert processor.frame_count == 21  # X discards the queued 22nd result.
    assert judge.call_count == 5
    assert not processor.manual_review_window.is_open
    assert not processor.manual_review_window.sealant_crops
    rows = [json.loads(line) for line in open(processor.workflow.cycle_logger.path)]
    assert len(rows) == 1 and rows[0]['outcome'] == 'Terminated'
    assert rows[0]['positions'][0]['status'] == 'REVIEW'
    processor.statistics_logger.log_detection.assert_not_called()
