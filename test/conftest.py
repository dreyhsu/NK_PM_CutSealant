"""Shared fixtures for baseline and modular behavior tests. Never access hardware."""
import importlib.util
import os
from pathlib import Path
import subprocess
import sys
import types
from unittest.mock import Mock

import cv2
import numpy as np
import pytest
import usb.core

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def forbidden(*args, **kwargs):
    raise AssertionError("Unexpected external I/O in an automated test")


@pytest.fixture(scope="session")
def legacy():
    if not os.environ.get('CUTSEALANT_BASELINE_REF'):
        # The historical files are standalone snapshots; test main.py's services.
        import cutsealant.app as app
        from cutsealant.review import ManualReviewWindow
        yield types.SimpleNamespace(
            **{name: getattr(app, name) for name in (
                'SafeVideoProcessor', 'ConfigManager', 'JSONLogger',
                'RejudgeLogger', 'StatisticsLogger', 'VideoRecorder')},
            ManualReviewWindow=ManualReviewWindow,
        )
        return
    # Baseline imports ultralytics eagerly; the test suite never needs a model.
    stub = types.ModuleType("ultralytics")
    stub.YOLO = forbidden
    original = sys.modules.get("ultralytics")
    sys.modules["ultralytics"] = stub
    spec = importlib.util.spec_from_file_location("legacy_cutsealant", ROOT / "yoloVideoMain_V6.0.7.py")
    module = importlib.util.module_from_spec(spec)
    baseline = os.environ.get("CUTSEALANT_BASELINE_REF")
    if baseline:
        source = subprocess.check_output(
            ["git", "show", f"{baseline}:yoloVideoMain_V6.0.7.py"], cwd=ROOT, text=True
        )
        exec(compile(source, str(ROOT / "yoloVideoMain_V6.0.7.py"), "exec"), module.__dict__)
    else:
        spec.loader.exec_module(module)
    yield module
    if original is None:
        sys.modules.pop("ultralytics", None)
    else:
        sys.modules["ultralytics"] = original


@pytest.fixture(autouse=True)
def isolated_io(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(subprocess, "call", forbidden)
    monkeypatch.setattr(cv2, "VideoCapture", forbidden)
    monkeypatch.setattr(cv2, "VideoWriter", forbidden)
    monkeypatch.setattr(usb.core, "find", forbidden)
    for name in ("namedWindow", "resizeWindow", "setMouseCallback", "imshow",
                 "destroyWindow", "destroyAllWindows", "setWindowProperty"):
        monkeypatch.setattr(cv2, name, Mock())
    monkeypatch.setattr(cv2, "waitKey", lambda _: -1)
    monkeypatch.setattr(cv2, "getWindowProperty", lambda *args: 1)


@pytest.fixture
def processor(legacy):
    p = legacy.SafeVideoProcessor()
    p.paths = {"stack_light_path": "LR6main.py", "validation_path": "./validation/"}
    p.class_map = {"carrier": 0, "case": 1}
    p.detection_config = {
        "fps": 15, "stop_threshold_seconds": 1,
        "distance_threshold_pixels": 10, "kmeans_clusters": 3,
        "kmeans_rounds": 10, "absent_tolerance_frames": 5,
        "result_hold_min_seconds": 2,
    }
    p.logger = Mock()
    return p


@pytest.fixture
def frame():
    return np.zeros((200, 200, 3), dtype=np.uint8)


@pytest.fixture
def detection():
    carrier = [10, 110, 190, 190]
    regions = [[20 + 35*i, 120, 45 + 35*i, 160] for i in range(4)]
    boxes = np.array([carrier, *regions], dtype=np.int32)
    return boxes, np.array([0, 1, 1, 1, 1]), np.array([.99, .91, .92, .93, .94])


@pytest.fixture
def rig(processor, frame, monkeypatch):
    """Feed distinct processed frames with real quadrant geometry and a fake judge."""
    now = [100.0]
    processor.clock = types.SimpleNamespace(time=lambda: now[0])
    processor.statistics_logger = Mock()
    processor.rejudge_logger = Mock()
    light = Mock(side_effect=lambda status: status)
    monkeypatch.setattr(processor, '_call_light_once', light)
    judge = Mock(return_value='Pass')
    monkeypatch.setattr(processor.inspection, 'observe', judge)
    state = ['Tracking', 15, [100, 150]]
    geometry = [[20, 120, 50, 140], [120, 120, 150, 140],
                [20, 160, 50, 180], [120, 160, 150, 180]]

    def feed(slots=(0, 1, 2, 3), shift=0, extras=(), confidences=None, image=None):
        carrier = [10+shift, 110, 190+shift, 190]
        cases = [[x1+shift, y1, x2+shift, y2] for i in slots
                 for x1, y1, x2, y2 in [geometry[i]]]
        cases += list(extras)
        boxes = np.array([carrier, *cases], dtype=np.int32)
        categories = np.array([0] + [1]*len(cases))
        scores = np.array([.99] + (confidences if confidences is not None else [.9]*len(cases)))
        processor.frame_count += 1
        source = frame if image is None else image
        state[:] = processor._handle_objects_present(source.copy(), source, boxes,
                     categories, list(boxes[1:]), scores, *state)
        return processor.workflow

    def absent(count=1, elapsed=0):
        now[0] += elapsed
        for _ in range(count):
            processor.frame_count += 1
            state[:] = processor._handle_objects_absent(*state)
        return processor.workflow

    return types.SimpleNamespace(p=processor, feed=feed, absent=absent, now=now,
                           state=state, judge=judge, light=light)
