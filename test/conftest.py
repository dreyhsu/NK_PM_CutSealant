"""Shared fixtures for baseline and modular behavior tests. Never access hardware."""
import importlib.util
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
    # Baseline imports ultralytics eagerly; the test suite never needs a model.
    stub = types.ModuleType("ultralytics")
    stub.YOLO = forbidden
    original = sys.modules.get("ultralytics")
    sys.modules["ultralytics"] = stub
    spec = importlib.util.spec_from_file_location("legacy_cutsealant", ROOT / "yoloVideoMain_V6.0.7.py")
    module = importlib.util.module_from_spec(spec)
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


@pytest.fixture
def processor(legacy):
    p = legacy.SafeVideoProcessor()
    p.paths = {"stack_light_path": "LR6main.py", "validation_path": "./validation/"}
    p.class_map = {"carrier": 0, "sealant": 1}
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
