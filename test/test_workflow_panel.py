from types import SimpleNamespace

import cv2
import numpy as np

from cutsealant.display import Display
from cutsealant.workflow import WORKFLOW_STEPS, WorkflowPanelState


def test_five_step_workflow_completes_and_stays_visible_until_next_board():
    workflow = WorkflowPanelState()

    assert workflow.status('board_in') == 'CURRENT'
    workflow.update(True, 0, 'WAITING')
    assert workflow.status('board_in') == 'DONE'
    assert workflow.status('modules_detected') == 'CURRENT'

    workflow.update(True, 4, 'WAITING')
    assert workflow.label('modules_detected') == '4 modules have been detected'
    assert workflow.status('inspecting') == 'CURRENT'

    # A lower one-frame count must not make the displayed count move backwards.
    workflow.update(True, 3, 'RESULT_HOLD')
    assert workflow.module_count == 4
    assert workflow.status('inspection_finished') == 'DONE'
    assert workflow.status('board_out') == 'CURRENT'

    workflow.update(False, 0, 'WAITING', board_released=True)
    assert workflow.complete is True
    assert workflow.completed_count == len(WORKFLOW_STEPS)
    assert workflow.phase == 'Cycle complete'

    workflow.update(False, 0, 'WAITING', board_released=True)
    assert workflow.complete is True

    workflow.update(True, 2, 'WAITING')
    assert workflow.complete is False
    assert workflow.module_count == 2
    assert workflow.status('modules_detected') == 'DONE'
    assert workflow.status('inspecting') == 'CURRENT'


def test_incomplete_board_departure_resets_cycle():
    workflow = WorkflowPanelState()
    workflow.update(True, 3, 'WAITING')
    workflow.update(False, 0, 'WAITING', board_released=True)
    assert workflow.completed_count == 0
    assert workflow.status('board_in') == 'CURRENT'


def test_workflow_panel_is_drawn_at_top_right(monkeypatch):
    workflow = WorkflowPanelState()
    workflow.update(True, 4, 'WAITING')
    context = SimpleNamespace(workflow=SimpleNamespace(panel=workflow))
    display = Display(context)
    frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
    labels = []
    rectangles = []

    original_rectangle = cv2.rectangle

    def capture_text(_frame, value, *_args, **_kwargs):
        labels.append(value)
        return _frame

    def capture_rectangle(image, point1, point2, *args, **kwargs):
        rectangles.append((point1, point2))
        return original_rectangle(image, point1, point2, *args, **kwargs)

    monkeypatch.setattr(cv2, 'putText', capture_text)
    monkeypatch.setattr(cv2, 'rectangle', capture_rectangle)
    display._draw_workflow_panel(frame)

    assert {'WORKFLOW', 'Board-In', '4 modules have been detected',
            'Inspecting', 'Inspection Finished', 'Board-Out'} <= set(labels)
    panel_top_left, _ = rectangles[0]
    assert panel_top_left[0] > frame.shape[1] // 2
    assert panel_top_left[1] == 10
