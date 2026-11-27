"""Carrier cycles with persistent, carrier-relative case positions."""
from dataclasses import dataclass, field
from datetime import datetime

import numpy as np

from .logs import CycleLogger

WORKFLOW_STEPS = ('carrier_in', 'inspecting_cases', 'carrier_out')


@dataclass
class WorkflowPanelState:
    completed: set = field(default_factory=set)
    case_count: int = 0
    cycle_started: bool = False
    cycle_complete: bool = False
    inspection_started: bool = False
    message: str = ''
    case_labels: list = field(default_factory=lambda: ['Not discovered'] * 4)

    def reset(self):
        self.completed.clear()
        self.case_count = 0
        self.cycle_started = self.cycle_complete = self.inspection_started = False
        self.message = ''
        self.case_labels = ['Not discovered'] * 4

    def update(self, carrier_present, case_count, detection_state, carrier_released=False):
        """Compatibility panel entry point; only readiness can complete inspection."""
        if carrier_present and self.cycle_complete:
            self.reset()
        if carrier_present:
            self.cycle_started = True
            self.completed.add('carrier_in')
            self.case_count = max(self.case_count, int(case_count))
        if detection_state == 'RESULT_HOLD' and self.case_count:
            self.inspection_started = True
            self.completed.add('inspecting_cases')
        if carrier_released and 'inspecting_cases' in self.completed:
            self.completed.add('carrier_out')
            self.cycle_complete = True

    @property
    def completed_count(self):
        return len(self.completed)

    @property
    def complete(self):
        return self.cycle_complete

    @property
    def phase(self):
        if self.message:
            return self.message
        if self.complete:
            return 'Cycle complete'
        current = next((step for step in WORKFLOW_STEPS if step not in self.completed), None)
        return self.label(current) if current else 'Cycle complete'

    def label(self, step):
        return {'carrier_in': 'Carrier-In',
                'inspecting_cases': f'Inspecting {self.case_count} cases on carrier',
                'carrier_out': 'Carrier-Out'}.get(step, '')

    def status(self, step):
        if step in self.completed:
            return 'DONE'
        current = next((name for name in WORKFLOW_STEPS if name not in self.completed), None)
        return 'CURRENT' if step == current else 'PENDING'


@dataclass
class InspectionState:
    """Compatibility fields owned by the workflow/main thread."""
    detection_state: str = 'WAITING'
    detection_result: object = None
    current_sealant_result: object = None
    sealant_detection_results: dict = field(default_factory=dict)
    detection_triggered_time: float = 0
    absent_frame_count: int = 0
    result_hold_start_time: float = 0
    detection_start_time: float = 0
    camera_error_start_time: float = 0
    pending_review: object = None
    carrier_count: int = 0
    user_reviewed_flag: bool = False
    current_raw_frame: object = None
    current_sealant_boxes: list = field(default_factory=list)
    current_scores: dict = field(default_factory=dict)
    review_log_written: bool = False


@dataclass
class CasePosition:
    index: int
    relative_bbox: list
    bbox: list
    status: str = 'CHECKING'
    pass_streak: int = 0
    fail_streak: int = 0
    decision_source: object = None
    observation: str = 'Unavailable'
    crop: object = None
    confidence: float = 0
    evidence_frame: int = 0
    user_reviewed: bool = False

    @property
    def passed(self):
        return self.status == 'PASS'

    @property
    def needs_review(self):
        return self.status in ('REVIEW', 'NG')

    def reset_streaks(self):
        if self.status == 'CHECKING':
            self.pass_streak = self.fail_streak = 0

    def label(self, threshold):
        if self.passed:
            return 'Manual Pass' if self.decision_source == 'manual' else 'Automatic Pass'
        return {'REVIEW': 'Manual check required', 'NG': 'Confirmed NG'}.get(
            self.status, f'Checking {self.pass_streak}/{threshold}')

    def record(self):
        return {'position_id': f'{self.index + 1:02}', 'index': self.index,
                'relative_bbox': self.relative_bbox, 'bbox': self.bbox,
                'status': self.status, 'decision_source': self.decision_source,
                'pass_streak': self.pass_streak, 'fail_streak': self.fail_streak,
                'last_observation': self.observation, 'confidence': self.confidence,
                'evidence_frame': self.evidence_frame, 'user_reviewed': self.user_reviewed}


class InspectionWorkflow:
    def __init__(self, context):
        self.context = context
        self.state = InspectionState()
        self.panel = WorkflowPanelState()
        self.positions = {}
        self.active = False
        self.last_processed_frame = None
        self.needs_stability = False
        self.carrier_missing = False
        self.cycle_logger = None
        self.audit_session_id = datetime.now().strftime('%Y%m%d_%H%M%S_%f')

    @property
    def ready(self):
        return bool(self.positions) and all(case.passed for case in self.positions.values())

    @staticmethod
    def position_index(carrier, box):
        x1, y1, x2, y2 = map(float, carrier)
        cx, cy = (float(box[0]) + float(box[2])) / 2, (float(box[1]) + float(box[3])) / 2
        if x2 <= x1 or y2 <= y1 or not (x1 <= cx < x2 and y1 <= cy < y2):
            return None
        return int(cx >= (x1 + x2) / 2) + 2 * int(cy >= (y1 + y2) / 2)

    def update_panel(self, carrier_present, case_count=0, carrier_released=False):
        if self.active:
            self.panel.cycle_started = self.panel.inspection_started = True
            self.panel.completed.add('carrier_in')
            self.panel.case_count = len(self.positions)
            if self.ready:
                self.panel.completed.add('inspecting_cases')
            else:
                self.panel.completed.discard('inspecting_cases')
            threshold = self.context.detection_config.get('case_pass_consecutive_frames', 10)
            self.panel.case_labels = [self.positions[i].label(threshold) if i in self.positions
                                      else 'Not discovered' for i in range(4)]
            self.panel.message = ('Cycle incomplete - restore carrier or press X'
                                  if self.carrier_missing and not self.ready else
                                  'Ready - remove carrier' if self.ready else
                                  'Manual check required - press M' if any(
                                      c.needs_review for c in self.positions.values()) else '')
        elif carrier_present and self.context.detection_state != 'CAMERA_ERROR':
            self.panel.update(True, 0, 'WAITING')

    def reset_camera_error(self, current_status):
        ctx = self.context
        if ctx.detection_state != 'CAMERA_ERROR':
            return current_status
        if ctx.logger:
            ctx.logger.log_system_event('camera_error_reset_by_operator', {
                'error_duration_seconds': round(ctx.clock.time() - ctx.camera_error_start_time, 1)})
            ctx.logger.log_state_change('CAMERA_ERROR', 'WAITING', 'operator_reset_c_key')
        ctx.detection_state = 'WAITING'
        ctx.camera_error_start_time = ctx.absent_frame_count = 0
        self.panel.reset()
        return ctx._call_light_once('Standby')

    def _reset_streaks(self):
        for case in self.positions.values():
            case.reset_streaks()

    def _handle_objects_present(self, frame, raw_frame, boxes, categories, sealant_boxes, scores,
                                current_status, stop_time, previous_centroid):
        ctx = self.context
        ctx.absent_frame_count = 0
        if ctx.detection_state == 'CAMERA_ERROR':
            return current_status, 0, previous_centroid
        if not self.active:
            return self._check_detection_trigger(frame, raw_frame, boxes, categories,
                                                 sealant_boxes, scores, current_status,
                                                 stop_time, previous_centroid)
        if self.last_processed_frame == ctx.frame_count:
            return current_status, stop_time, previous_centroid
        carrier = ctx._check_correct_carrier(categories, boxes)
        centroid = [(int(carrier[0]) + int(carrier[2])) // 2,
                    (int(carrier[1]) + int(carrier[3])) // 2]
        moving = np.linalg.norm(np.array(centroid) - previous_centroid) >= ctx.detection_config['distance_threshold_pixels']
        self.carrier_missing = False
        if moving or self.needs_stability:
            self._reset_streaks()
            self.needs_stability = True
            stop_time = 0 if moving else stop_time + 1
            if stop_time / ctx.detection_config['fps'] <= ctx.detection_config['stop_threshold_seconds']:
                self.last_processed_frame = ctx.frame_count
                self.update_panel(True)
                return current_status, stop_time, centroid
            self.needs_stability = False
        self._process_cases(raw_frame, carrier, boxes, categories, scores)
        return self.refresh_status(current_status), 0, centroid

    def _check_detection_trigger(self, frame, raw_frame, boxes, categories, sealant_boxes, scores,
                                 current_status, stop_time, previous_centroid):
        ctx = self.context
        carrier = ctx._check_correct_carrier(categories, boxes)
        if carrier is None or ctx.detection_state == 'CAMERA_ERROR':
            return current_status, 0, previous_centroid
        centroid = [(int(carrier[0]) + int(carrier[2])) // 2,
                    (int(carrier[1]) + int(carrier[3])) // 2]
        h, w = frame.shape[:2]
        distance = np.linalg.norm(np.array(centroid) - previous_centroid)
        if centroid[1] <= h / 2 or distance >= ctx.detection_config['distance_threshold_pixels']:
            return current_status, 0, centroid
        stop_time += 1
        if stop_time / ctx.detection_config['fps'] <= ctx.detection_config['stop_threshold_seconds']:
            return current_status, stop_time, centroid
        cfg = ctx.camera_health_config
        roi = [int(cfg.get('roi_x_ratio', .25) * w), int(cfg.get('roi_y_ratio', .25) * h),
               int(cfg.get('roi_w_ratio', .5) * w), int(cfg.get('roi_h_ratio', .5) * h)]
        if not (roi[0] <= centroid[0] <= roi[0] + roi[2] and roi[1] <= centroid[1] <= roi[1] + roi[3]):
            ctx.detection_state = 'CAMERA_ERROR'
            ctx.camera_error_start_time = ctx.clock.time()
            if ctx.logger:
                ctx.logger.log_state_change('WAITING', 'CAMERA_ERROR', 'carrier_centroid_outside_roi')
                ctx.logger.log_system_event('camera_misalignment_detected', {
                    'centroid': centroid, 'roi': dict(zip(('x', 'y', 'w', 'h'), roi))})
            return ctx._call_light_once('Standby'), 0, centroid
        self.active = True
        self.positions.clear()
        self.panel.reset()
        self.carrier_missing = self.needs_stability = False
        ctx.carrier_count += 1
        ctx.detection_start_time = ctx.clock.time()
        ctx.detection_state = 'INSPECTING'
        ctx.review_log_written = False
        ctx.user_reviewed_flag = False
        if ctx.logger:
            ctx.logger.log_detection_session_start()
            ctx.logger.log_state_change('WAITING', 'INSPECTING', 'carrier_stable_and_aligned')
        self._process_cases(raw_frame, carrier, boxes, categories, scores)
        return self.refresh_status(current_status), 0, centroid

    def _process_cases(self, frame, carrier, boxes, categories, scores):
        ctx = self.context
        if self.last_processed_frame == ctx.frame_count:
            return
        self.last_processed_frame = ctx.frame_count
        candidates = {}
        for n, (box, category) in enumerate(zip(boxes, categories)):
            if category != ctx.class_map['sealant']:
                continue
            index = self.position_index(carrier, box)
            if index is None or box[2] <= box[0] or box[3] <= box[1]:
                continue
            confidence = float(scores[n]) if n < len(scores) else 0.0
            area = float(box[2] - box[0]) * float(box[3] - box[1])
            rank = (confidence, -area)
            if index not in candidates or rank > candidates[index][0]:
                candidates[index] = (rank, box, confidence)
        for index in sorted(candidates):
            _, box, confidence = candidates[index]
            bbox = list(map(int, box))
            relative = [(bbox[0] - int(carrier[0])) / float(carrier[2] - carrier[0]),
                        (bbox[1] - int(carrier[1])) / float(carrier[3] - carrier[1]),
                        (bbox[2] - int(carrier[0])) / float(carrier[2] - carrier[0]),
                        (bbox[3] - int(carrier[1])) / float(carrier[3] - carrier[1])]
            if index not in self.positions:
                self.positions[index] = CasePosition(index, relative, bbox)
            case = self.positions[index]
            # Preserve the frame that caused manual checking, and all final evidence.
            if case.status != 'CHECKING':
                continue
            observation = ctx.inspection.observe(frame, bbox)
            case.observation = observation
            if observation == 'Unavailable':
                case.reset_streaks()
                continue
            case.relative_bbox, case.bbox, case.confidence = relative, bbox, confidence
            case.crop = frame[bbox[1]:bbox[3], bbox[0]:bbox[2]].copy()
            case.evidence_frame = ctx.frame_count
            if observation == 'Pass':
                case.pass_streak += 1
                case.fail_streak = 0
                if case.pass_streak >= ctx.detection_config.get('case_pass_consecutive_frames', 10):
                    case.status, case.decision_source = 'PASS', 'automatic'
            else:
                case.fail_streak += 1
                case.pass_streak = 0
                if case.fail_streak >= ctx.detection_config.get('case_fail_consecutive_frames', 5):
                    case.status = 'REVIEW'
        for index, case in self.positions.items():
            if index not in candidates:
                case.reset_streaks()
        ctx.current_raw_frame = frame.copy()
        self._sync_pending_review()

    def _sync_pending_review(self):
        ctx = self.context
        sealants = []
        for index, case in sorted(self.positions.items()):
            x1, y1, x2, y2 = case.bbox
            ai_result = 'NG' if case.needs_review or case.decision_source == 'manual' else case.observation
            final = ('REPASS' if case.decision_source == 'manual' and case.passed else
                     'Pass' if case.passed else 'NG' if case.status == 'NG' else
                     'Manual check required' if case.status == 'REVIEW' else 'Pending')
            sealants.append({'index': index, 'bbox': [x1, y1, x2-x1, y2-y1],
                             'ai_result': ai_result, 'confidence': case.confidence,
                             'user_reviewed': case.user_reviewed,
                             'user_agrees': not (case.passed and case.decision_source == 'manual'),
                             'final_result': final})
        ctx.pending_review = {'carrier_id': ctx.carrier_count, 'frame_number': ctx.frame_count,
                              'user_reviewed': any(c.user_reviewed for c in self.positions.values()),
                              'sealants': sealants}
        ctx.sealant_detection_results = {'_'.join(map(str, c.bbox)): not c.passed
                                         for c in self.positions.values()}

    def refresh_status(self, current_status):
        ctx = self.context
        if not self.active:
            return current_status
        if self.ready:
            if ctx.detection_state != 'RESULT_HOLD':
                ctx.result_hold_start_time = ctx.clock.time()
            ctx.detection_state, ctx.detection_result = 'RESULT_HOLD', 'Pass'
            requested = 'Pass'
        else:
            ctx.detection_state = 'INSPECTING'
            requested = 'NG' if any(c.needs_review for c in self.positions.values()) else 'Tracking'
            ctx.detection_result = 'NG' if requested == 'NG' else None
        ctx.current_sealant_result = requested == 'NG'
        self.update_panel(not self.carrier_missing)
        return ctx._call_light_once(requested) if requested != current_status else current_status

    def _handle_objects_absent(self, current_status, stop_time, previous_centroid):
        ctx = self.context
        ctx.absent_frame_count += 1
        self._reset_streaks()
        self.needs_stability = self.active
        tolerance = ctx.detection_config.get('absent_tolerance_frames', 5)
        if self.active:
            if ctx.absent_frame_count >= tolerance:
                self.carrier_missing = True
                if self.ready and ctx.clock.time() - ctx.result_hold_start_time >= ctx.detection_config.get('result_hold_min_seconds', 2):
                    self.finalize('Completed')
                    return ctx._call_light_once('Standby'), 0, [0, 0]
                self.update_panel(False)
        elif ctx.detection_state != 'CAMERA_ERROR' and ctx.absent_frame_count >= tolerance:
            current_status = ctx._call_light_once('Standby') if current_status != 'Standby' else current_status
        return current_status, 0, [0, 0]

    def finalize(self, outcome):
        """Emit one terminal record, then release this cycle's mutable state."""
        if not self.active:
            return
        ctx = self.context
        self._sync_pending_review()
        duration = max(0, int((ctx.clock.time() - ctx.detection_start_time) * 1000))
        record = {'carrier_id': ctx.carrier_count, 'outcome': outcome,
                  'cycle_time_ms': duration, 'source': ctx.source_name,
                  'positions': [c.record() for _, c in sorted(self.positions.items())]}
        if self.cycle_logger is None:
            self.cycle_logger = CycleLogger('./logs', self.audit_session_id)
        # Claim finalization before I/O, so cleanup cannot emit a second outcome.
        # A logging failure must never prevent the operator from terminating.
        self.active = False
        outputs = [('cycle_audit', lambda: self.cycle_logger.write(record)),
                   ('rejudge', ctx._write_pending_review_log)]
        if outcome == 'Completed' and ctx.statistics_logger:
            count = len(self.positions)
            outputs.append(('statistics', lambda: ctx.statistics_logger.log_detection(
                False, count, 0, count, duration)))
        if ctx.logger:
            outputs.extend([('cycle_event', lambda: ctx.logger.log_system_event('carrier_cycle_finalized', record)),
                            ('session_end', ctx.logger.log_detection_session_end)])
        for name, write in outputs:
            try:
                write()
            except Exception as exc:
                print(f'Cycle {ctx.carrier_count} {name} write failed: {exc}')
                if ctx.logger:
                    ctx.logger.log_error('cycle_record_error', exc,
                                         {'carrier_id': ctx.carrier_count, 'output': name})
        labels = list(self.panel.case_labels)
        count = len(self.positions)
        ctx.review.close()
        self.positions.clear()
        self.last_processed_frame = None
        self.needs_stability = self.carrier_missing = False
        self.state = InspectionState(carrier_count=ctx.carrier_count)
        self.panel.reset()
        self.panel.message = f'Cycle {outcome.lower()}'
        if outcome == 'Completed':
            self.panel.case_labels = labels
            self.panel.case_count = count
            self.panel.completed.update(WORKFLOW_STEPS)
            self.panel.cycle_started = self.panel.cycle_complete = True

    def terminate(self, current_status):
        self.finalize('Terminated')
        if self.context.detection_state == 'CAMERA_ERROR':
            return current_status
        self._reset_streaks()
        return self.context._call_light_once('Standby')
