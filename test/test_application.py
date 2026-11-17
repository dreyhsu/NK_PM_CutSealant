import importlib
from pathlib import Path
import runpy
import subprocess
import sys
from unittest.mock import Mock

import cv2
import numpy as np


def test_new_and_legacy_launchers_use_same_entrypoint(monkeypatch):
    import cutsealant.app as app
    main = Mock(return_value=0)
    monkeypatch.setattr(app, "main", main)
    root = Path(__file__).resolve().parents[1]
    for filename in ("main.py", "yoloVideoMain_V6.0.7.py"):
        try:
            runpy.run_path(str(root / filename), run_name="__main__")
        except SystemExit as result:
            assert result.code == 0
        else:
            raise AssertionError("Launcher must return the entrypoint's exit code")
    assert main.call_count == 2


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
    assert send.call_args.args[0] == ["python", "LR6main.py", "2", "0", "1", "0", "0", "0"]
    processor.cap.release.assert_called_once()


def test_board_in_changes_static_yellow_to_tracking_blink(processor, frame, detection,
                                                          monkeypatch):
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


def test_two_carrier_sequence_pass_then_ng(processor, frame, detection, monkeypatch):
    boxes, categories, scores = detection
    send = Mock(return_value=0)
    processor.command_runner = send
    monkeypatch.setattr(processor, "_cut_sealant_detection", Mock(side_effect=[False, True]))
    monkeypatch.setattr(processor, "_save_img_data", Mock())
    processor.rejudge_logger = Mock()
    now = [100.0]
    processor.clock = type("Clock", (), {"time": lambda _: now[0]})()
    current = processor._call_light_once("Standby")
    for expected in ("Pass", "NG"):
        current, _, _ = processor._handle_objects_present(frame, frame, boxes, categories, list(boxes[1:]), scores, current, 15, [100, 150])
        assert processor.detection_result == expected
        now[0] += 2
        for _ in range(5):
            current, _, _ = processor._handle_objects_absent(current, 0, [100, 150])
        assert processor.detection_state == "WAITING"
    assert processor.carrier_count == 2
    assert processor.rejudge_logger.write_records.call_count == 2
    assert [c.args[0][2:] for c in send.call_args_list] == [
        ["2", "0", "1", "0", "0", "0"], ["2", "0", "0", "1", "0", "0"],
        ["2", "0", "1", "0", "0", "0"], ["2", "1", "0", "0", "0", "0"],
        ["4", "3", "1", "1", "14"], ["2", "0", "1", "0", "0", "0"]]


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
