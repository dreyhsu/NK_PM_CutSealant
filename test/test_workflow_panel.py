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
    assert workflow.status('inspecting_modules') == 'CURRENT'

    workflow.update(True, 4, 'WAITING')
    assert workflow.label('inspecting_modules') == 'Inspecting 4 modules'
    assert workflow.status('inspecting_modules') == 'CURRENT'

    # The OSD follows the current count so an obstruction is visible.
    workflow.update(True, 3, 'RESULT_HOLD')
    assert workflow.module_count == 3
    assert workflow.status('inspecting_modules') == 'CURRENT'
    assert workflow.status('board_out') == 'PENDING'

    workflow.update(False, 0, 'WAITING', board_released=True)
    assert workflow.complete is True
    assert workflow.completed_count == len(WORKFLOW_STEPS)
    assert workflow.phase == 'Cycle complete'

    workflow.update(False, 0, 'WAITING', board_released=True)
    assert workflow.complete is True

    workflow.update(True, 2, 'WAITING')
    assert workflow.complete is False
    assert workflow.module_count == 2
    assert workflow.status('inspecting_modules') == 'CURRENT'


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
    display._draw_workflow_panel(frame, 'Standby')

    assert {'WORKFLOW', 'Board-In', 'Inspecting 4 modules', 'Board-Out',
            'REQUEST: STANDBY (STATIC)', 'USB: UNKNOWN'} <= set(labels)
    assert any(label.startswith('TIME ') for label in labels)
    panel_top_left, _ = rectangles[0]
    assert panel_top_left[0] > frame.shape[1] // 2
    assert panel_top_left[1] == 10


def test_top_left_osd_is_removed(monkeypatch):
    workflow = WorkflowPanelState()
    context = SimpleNamespace(
        workflow=SimpleNamespace(panel=workflow),
        display_config={'show_crosshair': False, 'show_timestamp': True,
                        'show_status': True},
        detection_state='RESULT_HOLD',
        detection_result='NG',
        sealant_detection_results={'box': True},
        last_light_status='NG',
    )
    display = Display(context)
    frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
    labels = []

    def capture_text(_frame, value, *_args, **_kwargs):
        labels.append(value)
        return _frame

    monkeypatch.setattr(cv2, 'putText', capture_text)
    display._add_ui_elements(frame, 'NG')

    assert 'REQUEST: NG' in labels
    assert not any(label.startswith(('Status:', 'State:', 'Result:', 'Broken:'))
                   for label in labels)
    assert 'All sealant OK' not in labels


def test_tracking_osd_identifies_blinking_yellow(monkeypatch):
    workflow = WorkflowPanelState()
    workflow.update(True, 4, 'WAITING')
    context = SimpleNamespace(workflow=SimpleNamespace(panel=workflow))
    display = Display(context)
    frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
    labels = []

    def capture_text(_frame, value, *_args, **_kwargs):
        labels.append(value)
        return _frame

    monkeypatch.setattr(cv2, 'putText', capture_text)
    display._draw_workflow_panel(frame, 'Tracking')

    assert 'REQUEST: TRACKING (BLINK)' in labels
