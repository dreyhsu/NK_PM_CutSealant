import queue
from types import SimpleNamespace
from unittest.mock import Mock

import cv2
import numpy as np
import pytest

from cutsealant.app import SafeVideoProcessor
from cutsealant.config import ConfigManager
from cutsealant.camera import CameraSource
from cutsealant.roi import InferenceROI


def test_initialize_with_injected_model_camera_and_no_logs(tmp_path, frame):
    config = ConfigManager()
    config.config["PATHS"]["model_path"] = str(tmp_path / "fake.pt")
    config.config["VIDEO_SOURCE"]["mode"] = "camera"
    config.config["LOGGING_SETTINGS"]["save_detection_log"] = "false"
    with open(config.config_file, "w") as f:
        config.config.write(f)
    (tmp_path / "fake.pt").touch()
    model, cap = Mock(), Mock()
    cap.read.return_value = (True, frame)
    cap.get.side_effect = lambda prop: {
        cv2.CAP_PROP_FRAME_WIDTH: frame.shape[1],
        cv2.CAP_PROP_FRAME_HEIGHT: frame.shape[0],
        cv2.CAP_PROP_FPS: 30.0,
    }.get(prop, 0)
    factory, capture = Mock(return_value=model), Mock(return_value=cap)
    p = SafeVideoProcessor(model_factory=factory, capture_factory=capture)
    try:
        assert p.initialize() is True
        factory.assert_called_once_with(str(tmp_path / "fake.pt"))
        model.to.assert_called_once_with("cpu")
        assert p.model is p.detector.model is model
        assert p.cap is p.camera.capture is cap
        assert p.frame_queue.get_nowait() is frame
        assert p.logger is p.statistics_logger is p.rejudge_logger is None
        assert not (tmp_path / "logs").exists()
    finally:
        p.stop()


def test_missing_model_fails_before_loading_or_opening_camera(tmp_path):
    config = ConfigManager()
    config.config["LOGGING_SETTINGS"]["save_detection_log"] = "false"
    with open(config.config_file, "w") as f:
        config.config.write(f)
    factory, capture = Mock(), Mock()
    p = SafeVideoProcessor(model_factory=factory, capture_factory=capture)
    assert p.initialize() is False
    factory.assert_not_called()
    capture.assert_not_called()


def test_camera_setup_failure_and_invalid_mode(processor):
    processor.config_manager = Mock()
    processor.config_manager.get_video_source_config.return_value = {"mode": "camera", "camera_index": 0}
    cap = Mock()
    cap.isOpened.return_value = False
    processor.capture_factory = Mock(return_value=cap)
    assert processor._setup_video_source() == (None, None)
    assert processor.logger.log_error.call_args.args[0] == "camera_open_error"
    processor.config_manager.get_video_source_config.return_value = {"mode": "invalid"}
    with pytest.raises(ValueError):
        processor._setup_video_source()


@pytest.mark.parametrize('system,expected', [
    ('Windows', cv2.CAP_DSHOW),
    ('Darwin', cv2.CAP_AVFOUNDATION),
    ('Linux', cv2.CAP_V4L2),
    ('Other', cv2.CAP_ANY),
])
def test_camera_backend_matches_operating_system(monkeypatch, system, expected):
    monkeypatch.setattr('cutsealant.camera.platform.system', lambda: system)
    assert CameraSource._opencv_camera_backend() == expected


@pytest.mark.parametrize("opened", [True, False])
def test_reconnect_releases_old_capture_and_resets_on_success(processor, monkeypatch, opened):
    old, new = Mock(), Mock()
    new.isOpened.return_value = opened
    processor.cap = old
    processor.clock = SimpleNamespace(time=lambda: 100, sleep=Mock())
    monkeypatch.setattr(processor, "_setup_video_source", Mock(return_value=(new, "fake")))
    assert processor._attempt_camera_reconnect() is opened
    old.release.assert_called_once()
    processor.clock.sleep.assert_called_once_with(5)
    assert processor.camera_reconnect_attempts == (0 if opened else 1)


def test_reconnect_stops_after_limit(processor, monkeypatch):
    processor.camera_reconnect_attempts = processor.max_camera_reconnect_attempts
    setup = Mock()
    monkeypatch.setattr(processor, "_setup_video_source", setup)
    assert processor._attempt_camera_reconnect() is False
    setup.assert_not_called()


@pytest.mark.parametrize("full", [False, True])
def test_capture_worker_delivers_or_drops_frame(processor, frame, full):
    processor.cap = Mock()
    processor.cap.isOpened.side_effect = [True, False]
    processor.cap.read.return_value = (True, frame)
    processor.config_manager = Mock()
    processor.config_manager.get_video_source_config.return_value = {"mode": "video"}
    processor.frame_queue = Mock()
    if full:
        processor.frame_queue.put.side_effect = queue.Full
    processor._capture_frames()
    processor.frame_queue.put.assert_called_once_with(frame, timeout=.1)


def test_capture_failure_stops_after_threshold_without_waiting(processor):
    processor.cap = Mock()
    processor.cap.read.return_value = (False, None)
    processor.clock = SimpleNamespace(time=lambda: 100, sleep=Mock())
    processor.config_manager = Mock()
    processor.config_manager.get_video_source_config.return_value = {"mode": "video"}
    processor._capture_frames()
    assert processor.cap.read.call_count == 50
    assert processor.logger.log_error.call_args.args[0] == "capture_max_failures"


def test_yolo_outputs_and_prediction_settings(processor, frame):
    processor.inference_roi = InferenceROI((200, 200), (0, 0, 200, 200))
    processor.yolo_config = {"confidence_threshold": .8, "iou_threshold": .45, "max_detections": 100,
                             "use_augmentation": False, "use_half_precision": True}
    def tensor(value):
        result = Mock()
        result.cpu.return_value.numpy.return_value = np.array(value)
        return result
    boxes = SimpleNamespace(xyxy=tensor([[1.8, 2, 30, 40]]), conf=tensor([.9]), cls=tensor([1]))
    processor.model = Mock()
    processor.model.predict.return_value = [SimpleNamespace(boxes=boxes)]
    result, scores, categories = processor._yolo_detect(frame)
    assert result.tolist() == [[1, 2, 30, 40]]
    assert scores.tolist() == [.9]
    assert categories.tolist() == [1]
    assert processor.model.predict.call_args.kwargs == {
        "conf": .8, "iou": .45, "agnostic_nms": False, "max_det": 100,
        "augment": False, "verbose": False, "half": True}


@pytest.mark.parametrize("raises", [False, True])
def test_empty_or_failed_yolo_returns_empty_arrays(processor, frame, raises):
    processor.inference_roi = InferenceROI((200, 200), (0, 0, 200, 200))
    processor.yolo_config = {"confidence_threshold": .8, "iou_threshold": .45, "max_detections": 100,
                             "use_augmentation": False, "use_half_precision": True}
    processor.model = Mock()
    if raises:
        processor.model.predict.side_effect = RuntimeError("fake failure")
    else:
        processor.model.predict.return_value = []
    boxes, scores, classes = processor._yolo_detect(frame)
    assert boxes.shape == (0, 4)
    assert len(scores) == len(classes) == 0
    assert processor.logger.log_error.call_count == int(raises)


@pytest.mark.parametrize("full", [False, True])
def test_yolo_worker_forwards_or_drops_result(processor, frame, monkeypatch, full):
    processor.frame_queue.put(frame)
    processor.frame_queue.put(None)
    prediction = (np.zeros((0, 4)), np.array([]), np.array([]))
    monkeypatch.setattr(processor, "_yolo_detect", Mock(return_value=prediction))
    processor.result_queue = Mock()
    if full:
        processor.result_queue.put.side_effect = queue.Full
    processor._yolo_worker()
    processor.result_queue.put.assert_called_once_with((frame, *prediction), timeout=.1)


@pytest.mark.parametrize("success", [False, True])
def test_model_reload_uses_injected_factory(processor, monkeypatch, success):
    processor.paths["model_path"] = "fake.pt"
    monkeypatch.setattr(processor, "_try_cuda_memory_cleanup", Mock())
    factory = Mock(return_value=Mock()) if success else Mock(side_effect=RuntimeError("reload failed"))
    processor.model_factory = factory
    assert processor._try_reload_model() is success
    factory.assert_called_once_with("fake.pt")


def test_instances_do_not_share_queues_or_carrier_state():
    a, b = SafeVideoProcessor(), SafeVideoProcessor()
    a.frame_queue.put("frame")
    a.sealant_detection_results["box"] = True
    assert a.frame_queue.maxsize == a.result_queue.maxsize == 2
    assert b.frame_queue.empty()
    assert b.sealant_detection_results == {}
    a.detection_state = "RESULT_HOLD"
    assert a.workflow.state.detection_state == "RESULT_HOLD"
    assert b.detection_state == "WAITING"


def test_video_source_uses_configured_file(processor, tmp_path):
    path = tmp_path / "sample.avi"
    path.touch()
    processor.paths["video_path"] = str(path)
    processor.config_manager = Mock()
    processor.config_manager.get_video_source_config.return_value = {"mode": "video", "video_index": 0}
    cap = Mock()
    processor.capture_factory = Mock(return_value=cap)
    assert processor._setup_video_source() == (cap, "sample.avi")
    processor.capture_factory.assert_called_once_with(str(path))


def test_tis_source_uses_wrapper_without_loading_dll(processor, monkeypatch):
    import cutsealant.camera as camera
    processor.config_manager = Mock()
    processor.config_manager.get_video_source_config.return_value = {
        "mode": "dfk_camera", "tis_camera_index": 1, "tis_dll_path": "fake.dll", "tis_device_name": "DFK"}
    wrapper = Mock()
    wrapper.get_device_info.return_value = {"device_name": "DFK_test", "resolution": "200x200"}
    factory = Mock(return_value=wrapper)
    monkeypatch.setattr(camera, "TIS_AVAILABLE", True)
    monkeypatch.setattr(camera, "TISCameraWrapper", factory)
    assert processor._setup_video_source() == (wrapper, "TIS_DFK_test")
    factory.assert_called_once_with(dll_path="fake.dll", device_name="DFK", camera_index=1, logger=processor.logger)


@pytest.mark.parametrize("reload_success", [True, False])
def test_worker_cuda_recovery_threshold(processor, frame, monkeypatch, reload_success):
    processor.frame_queue = Mock()
    processor.frame_queue.get.side_effect = [frame, frame, frame, None]
    processor.clock = SimpleNamespace(time=lambda: 0, sleep=Mock())
    monkeypatch.setattr(processor, "_yolo_detect", Mock(side_effect=RuntimeError("CUDA out of memory")))
    cleanup, reload = Mock(), Mock(return_value=reload_success)
    monkeypatch.setattr(processor, "_try_cuda_memory_cleanup", cleanup)
    monkeypatch.setattr(processor, "_try_reload_model", reload)
    processor._yolo_worker()
    assert cleanup.call_count == 3
    reload.assert_called_once()
    assert processor.frame_queue.get.call_count == (4 if reload_success else 3)


def test_worker_general_failure_limit(processor, frame, monkeypatch):
    processor.frame_queue = Mock()
    processor.frame_queue.get.return_value = frame
    processor.clock = SimpleNamespace(time=lambda: 0, sleep=Mock())
    monkeypatch.setattr(processor, "_yolo_detect", Mock(side_effect=ValueError("fake inference failure")))
    processor._yolo_worker()
    assert processor.frame_queue.get.call_count == 30
    assert processor.logger.log_error.call_args.args[0] == "yolo_worker_fatal_error"
