"""Crop-only inference, original-coordinate restoration and placement overlay."""
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import cv2
import numpy as np
import pytest

from cutsealant.app import SafeVideoProcessor
from cutsealant.config import ConfigManager
from cutsealant.roi import DEFAULT_INFERENCE_ROI, InferenceROI


@pytest.fixture
def cropped_processor(processor):
    path = Path(__file__).resolve().parents[1] / 'config' / 'inference_roi.json'
    processor.inference_roi = InferenceROI.load(path)
    processor.yolo_config = {'confidence_threshold': .8, 'iou_threshold': .45,
                             'max_detections': 100, 'use_augmentation': False,
                             'use_half_precision': False}
    processor.model = Mock()
    return processor


def prediction(boxes, scores, classes):
    def tensor(value, shape=None):
        array = np.array(value, dtype=np.float32)
        if shape is not None:
            array = array.reshape(shape)
        result = Mock()
        result.cpu.return_value.numpy.return_value = array
        return result
    return [SimpleNamespace(boxes=SimpleNamespace(
        xyxy=tensor(boxes, (-1, 4)), conf=tensor(scores), cls=tensor(classes)))]


def test_saved_roi_matches_requested_coordinates():
    path = Path(__file__).resolve().parents[1] / 'config' / 'inference_roi.json'
    assert json.loads(path.read_text()) == {
        'version': 1, 'reference_size': [1344, 756], 'roi_xyxy': [454, 312, 850, 670]}


@pytest.mark.parametrize('width,height,bounds', [
    (1344, 756, (454, 312, 850, 670)),
    (2688, 1512, (908, 624, 1700, 1340)),
    (4096, 2160, (1383, 891, 2591, 1915)),
])
def test_predict_receives_only_scaled_crop_and_returns_global_boxes(cropped_processor, width, height, bounds):
    p = cropped_processor
    image = np.zeros((height, width, 3), np.uint8)
    x1, y1, x2, y2 = bounds
    image[y1:y2, x1:x2] = [20, 60, 100]
    before = image.copy()
    p.model.predict.return_value = prediction([[5.9, 10.2, 100.9, 200.9]], [.91], [1])
    boxes, scores, classes = p._yolo_detect(image)
    crop = p.model.predict.call_args.args[0]
    assert crop.shape == (y2-y1, x2-x1, 3)
    assert np.array_equal(crop, before[y1:y2, x1:x2])
    assert not np.shares_memory(crop, image)
    assert np.array_equal(image, before)
    assert boxes.tolist() == [[x1+5, y1+10, x1+100, y1+200]]
    assert np.allclose(scores, [.91]) and classes.tolist() == [1]


def test_crop_boundary_clipping_discards_empty_boxes_and_preserves_metadata(cropped_processor):
    p = cropped_processor
    p.model.predict.return_value = prediction([
        [-20, -10, 30, 40], [-10, -10, 10000, 10000],
        [500, 10, 600, 50], [-10, -10, -2, -1]], [.9, .8, .7, .6], [1, 0, 1, 0])
    boxes, scores, classes = p._yolo_detect(np.zeros((756, 1344, 3), np.uint8))
    assert boxes.tolist() == [[454, 312, 484, 352], [454, 312, 850, 670]]
    assert np.allclose(scores, [.9, .8]) and classes.tolist() == [1, 0]


def test_empty_prediction_boxes_keep_expected_shape(cropped_processor):
    p = cropped_processor
    p.model.predict.return_value = prediction([], [], [])
    boxes, scores, classes = p._yolo_detect(np.zeros((756, 1344, 3), np.uint8))
    assert boxes.shape == (0, 4)
    assert not len(scores) and not len(classes)
    p.logger.log_error.assert_not_called()


def test_worker_keeps_original_frame_for_quality_judgment(cropped_processor):
    p = cropped_processor
    image = np.zeros((756, 1344, 3), np.uint8)
    p.model.predict.return_value = prediction([[10, 10, 100, 100]], [.9], [1])
    p.frame_queue.put(image)
    p.frame_queue.put(None)
    p._yolo_worker()
    original, boxes, scores, classes = p.result_queue.get_nowait()
    assert original is image
    assert original.shape == (756, 1344, 3)
    assert p.model.predict.call_args.args[0].shape == (358, 396, 3)
    assert boxes.tolist() == [[464, 322, 554, 412]]


@pytest.mark.parametrize('width,height', [(1344, 756), (4096, 2160)])
def test_thick_yellow_outline_matches_inference_bounds(cropped_processor, monkeypatch, width, height):
    p = cropped_processor
    image = np.zeros((height, width, 3), np.uint8)
    rectangle = Mock(wraps=cv2.rectangle)
    monkeypatch.setattr(cv2, 'rectangle', rectangle)
    p.display._draw_inference_roi(image)
    x1, y1, x2, y2 = p.inference_roi.scaled_bounds(image.shape)
    args = rectangle.call_args.args
    assert args[1:4] == ((x1, y1), (x2-1, y2-1), (0, 255, 255))
    assert args[4] >= 4
    assert tuple(image[y1, (x1+x2)//2]) == (0, 255, 255)
    assert tuple(image[(y1+y2)//2, x1]) == (0, 255, 255)
    assert not image[(y1+y2)//2, (x1+x2)//2].any()
    assert not image[0, 0].any()


def test_main_overlay_draws_roi_after_panel(cropped_processor, monkeypatch):
    p = cropped_processor
    p.display_config = {'show_crosshair': False}
    draw_roi = Mock()
    monkeypatch.setattr(p.display, '_draw_inference_roi', draw_roi)
    image = np.zeros((756, 1344, 3), np.uint8)
    p.display._add_ui_elements(image, 'Standby')
    draw_roi.assert_called_once_with(image)


def test_default_config_creates_roi_next_to_ini_without_overwriting_it(tmp_path):
    path = tmp_path / 'custom' / 'config.ini'
    config = ConfigManager(path)
    roi_path = path.with_name('inference_roi.json')
    assert json.loads(roi_path.read_text()) == DEFAULT_INFERENCE_ROI
    edited = {**DEFAULT_INFERENCE_ROI, 'roi_xyxy': [100, 100, 200, 200]}
    roi_path.write_text(json.dumps(edited))
    path.unlink()
    config = ConfigManager(path)
    assert config.get_inference_roi().roi_xyxy == (100, 100, 200, 200)


@pytest.mark.parametrize('update', [
    {'version': 2}, {'version': True}, {'reference_size': [0, 756]},
    {'reference_size': [1344]}, {'reference_size': [1344.0, 756]},
    {'roi_xyxy': [454, 312, 454, 670]}, {'roi_xyxy': [-1, 312, 850, 670]},
    {'roi_xyxy': [454, 312, 1400, 670]}, {'roi_xyxy': [454, 312, 850, 800]},
    {'roi_xyxy': [454, 312, 850]}, {'roi_xyxy': [454, True, 850, 670]},
])
def test_invalid_roi_is_rejected(tmp_path, update):
    path = tmp_path / 'roi.json'
    path.write_text(json.dumps({**DEFAULT_INFERENCE_ROI, **update}))
    with pytest.raises(ValueError):
        InferenceROI.load(path)


@pytest.mark.parametrize('roi_content', [None, '{bad json', '[]'])
def test_invalid_or_missing_config_stops_before_model_or_camera(tmp_path, roi_content):
    config = ConfigManager()
    roi_path = Path(config.config_file).with_name('inference_roi.json')
    if roi_content is None:
        roi_path.unlink()
    else:
        roi_path.write_text(roi_content)
    factory, capture = Mock(), Mock()
    p = SafeVideoProcessor(model_factory=factory, capture_factory=capture)
    assert p.initialize() is False
    factory.assert_not_called()
    capture.assert_not_called()
    p.stop()


def test_unloaded_roi_never_falls_back_to_whole_frame(processor, frame):
    processor.model = Mock()
    boxes, scores, classes = processor._yolo_detect(frame)
    processor.model.predict.assert_not_called()
    assert boxes.shape == (0, 4) and not len(scores) and not len(classes)
