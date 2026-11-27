from types import SimpleNamespace

import cv2
import numpy as np

from cutsealant.display import Display
from cutsealant.workflow import WORKFLOW_STEPS, WorkflowPanelState


def test_three_step_workflow_completes_and_stays_visible_until_next_carrier():
    workflow = WorkflowPanelState()

    assert workflow.status('carrier_in') == 'CURRENT'
    workflow.update(True, 0, 'WAITING')
    assert workflow.status('carrier_in') == 'DONE'
    assert workflow.status('inspecting_cases') == 'CURRENT'

    workflow.update(True, 4, 'WAITING')
    assert workflow.label('inspecting_cases') == 'Inspecting 4 cases on carrier'
    assert workflow.status('inspecting_cases') == 'CURRENT'

    # The OSD remembers discovered cases while individual detections disappear.
    workflow.update(True, 3, 'RESULT_HOLD')
    assert workflow.case_count == 4
    assert workflow.status('inspecting_cases') == 'DONE'
    assert workflow.status('carrier_out') == 'CURRENT'

    workflow.update(False, 0, 'WAITING', carrier_released=True)
    assert workflow.complete is True
    assert workflow.completed_count == len(WORKFLOW_STEPS)
    assert workflow.phase == 'Cycle complete'

    workflow.update(False, 0, 'WAITING', carrier_released=True)
    assert workflow.complete is True

    workflow.update(True, 2, 'WAITING')
    assert workflow.complete is False
    assert workflow.case_count == 2
    assert workflow.status('inspecting_cases') == 'CURRENT'


def test_incomplete_carrier_departure_preserves_panel():
    workflow = WorkflowPanelState()
    workflow.update(True, 3, 'WAITING')
    workflow.update(False, 0, 'WAITING', carrier_released=True)
    assert workflow.completed_count == 1
    assert workflow.status('inspecting_cases') == 'CURRENT'


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

    assert {'WORKFLOW', 'Carrier-In', 'Inspecting 4 cases on carrier', 'Carrier-Out',
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


def test_panel_displays_all_position_statuses(rig, monkeypatch):
    from cutsealant.workflow import CasePosition
    workflow = rig.feed(())
    for index, status in enumerate(['CHECKING', 'REVIEW', 'NG', 'PASS']):
        workflow.positions[index] = CasePosition(index, [0, 0, .2, .2], [0, 0, 20, 20],
                                                 status=status, decision_source='manual' if index == 3 else None)
    workflow.update_panel(True)
    labels = []
    monkeypatch.setattr(cv2, 'putText', lambda image, text, *args, **kwargs: labels.append(text))
    rig.p.display._draw_workflow_panel(np.zeros((1080, 1920, 3), np.uint8), 'NG')
    assert {'Position 01: Checking 0/10', 'Position 02: Manual check required',
            'Position 03: Confirmed NG', 'Position 04: Manual Pass'} <= set(labels)
