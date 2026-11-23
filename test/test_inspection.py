from unittest.mock import Mock

import cv2
import numpy as np
import pytest


@pytest.mark.parametrize("points,expected", [
    ([[0, 0], [20, 0], [20, 20], [0, 20]], True),
    ([[0, 0], [20, 0], [10, 10], [20, 20], [0, 20]], False),
    ([[0, 0], [1, 1]], False),
])
def test_contour_decision(processor, points, expected):
    assert bool(processor._loop_closed_identify(np.array(points, np.int32).reshape(-1, 1, 2))) is expected


@pytest.mark.parametrize("outcomes", [[True]*4, [True, False, True, True]])
def test_region_aggregation(processor, frame, detection, monkeypatch, outcomes):
    boxes, _, _ = detection
    contour = np.array([[[0, 0]], [[10, 0]], [[10, 10]]], np.int32)
    monkeypatch.setattr(processor, "_img_kmeans", lambda img: (img, np.array([[0], [100], [255]], np.uint8)))
    monkeypatch.setattr(cv2, "findContours", lambda *a: ([contour], None))
    judge = Mock(side_effect=outcomes)
    monkeypatch.setattr(processor, "_loop_closed_identify", judge)
    assert processor._cut_sealant_detection(frame, boxes[1:]) == (not all(outcomes))
    assert list(processor.sealant_detection_results.values()) == [not x for x in outcomes]
    assert judge.call_count == 4


def test_kmeans_uses_brightest_cluster_values(processor):
    image = np.repeat(np.array([[0, 120, 250]], np.uint8), 30, axis=0)
    cv2.setRNGSeed(42)
    segmented, centers = processor._img_kmeans(image)
    assert segmented.shape == image.shape
    assert sorted(centers.ravel()) == [0, 120, 250]
    assert np.array_equal(segmented, image)


def test_missing_fourth_region_is_ng(processor, frame, detection, monkeypatch):
    boxes, _, _ = detection
    monkeypatch.setattr(processor, "_loop_closed_identify", lambda _: True)
    frame[:] = 255
    cv2.setRNGSeed(42)
    assert processor._cut_sealant_detection(frame, boxes[1:4]) is True
    assert len(processor.sealant_detection_results) == 3


def test_extra_region_is_ng(processor, frame, detection, monkeypatch):
    boxes, _, _ = detection
    extra = np.array([[5, 5, 25, 25]], np.int32)
    case_boxes = np.concatenate((boxes[1:], extra))
    monkeypatch.setattr(processor, '_loop_closed_identify', lambda _: True)
    frame[:] = 255
    cv2.setRNGSeed(42)
    assert processor._cut_sealant_detection(frame, case_boxes) is True
    assert len(processor.sealant_detection_results) == 5


@pytest.mark.parametrize("mode", ["empty", "no_contour", "exception"])
def test_unanalysed_region_is_ng(processor, frame, monkeypatch, mode):
    box = np.array([[0, 0, 20, 20]], np.int32)
    if mode == "empty":
        box[:] = 0
    elif mode == "no_contour":
        monkeypatch.setattr(cv2, "findContours", lambda *a: ([], None))
    else:
        monkeypatch.setattr(processor, "_img_kmeans", Mock(side_effect=ValueError("bad crop")))
    assert processor._cut_sealant_detection(frame, box) is True
    assert list(processor.sealant_detection_results.values()) == [True]


@pytest.mark.known_issue
def test_large_box_analyses_only_largest_contour(processor, frame, monkeypatch):
    contours = [np.array([[[0, 0]], [[s, 0]], [[s, s]], [[0, s]]], np.int32) for s in [3, 12, 5, 8]]
    monkeypatch.setattr(cv2, "findContours", lambda *a: (contours, None))
    judge = Mock(return_value=True)
    monkeypatch.setattr(processor, "_loop_closed_identify", judge)
    processor._cut_sealant_detection(frame, np.array([[0, 0, 50, 50]]))
    judge.assert_called_once()
    assert np.array_equal(judge.call_args.args[0], contours[1])
