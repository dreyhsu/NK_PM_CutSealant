"""Workflow functionality extracted from CutSealant V6.0.7."""
from dataclasses import dataclass, field

import cv2
import numpy as np

WORKFLOW_STEPS = (
    'carrier_in',
    'inspecting_cases',
    'carrier_out',
)


@dataclass
class WorkflowPanelState:
    """Persistent carrier cycle, with cases inspected on the carrier."""

    completed: set = field(default_factory=set)
    case_count: int = 0
    cycle_started: bool = False
    cycle_complete: bool = False
    inspection_started: bool = False

    def reset(self):
        """Prepare the panel for a carrier that has not arrived yet."""
        self.completed.clear()
        self.case_count = 0
        self.cycle_started = False
        self.cycle_complete = False
        self.inspection_started = False

    def update(self, carrier_present, case_count, detection_state, carrier_released=False):
        """Advance the display state from observations made for the current frame."""
        case_count = max(0, int(case_count))

        # Keep a completed cycle on screen until the next carrier starts a new one.
        if carrier_present and self.cycle_complete:
            self.reset()

        if carrier_present:
            self.cycle_started = True
            self.completed.add('carrier_in')
            # Show the current count so a blocked case is visible in the OSD.
            self.case_count = case_count

        if detection_state == 'RESULT_HOLD' and self.cycle_started:
            self.inspection_started = True

        if carrier_released:
            if self.inspection_started:
                self.completed.update(('inspecting_cases', 'carrier_out'))
                self.cycle_complete = True
            else:
                # An incomplete carrier departure is not a completed cycle.
                self.reset()

    @property
    def completed_count(self):
        return len(self.completed)

    @property
    def complete(self):
        return self.cycle_complete

    @property
    def phase(self):
        if self.complete:
            return 'Cycle complete'
        current = next((step for step in WORKFLOW_STEPS if step not in self.completed), None)
        return self.label(current) if current else 'Cycle complete'

    def label(self, step):
        labels = {
            'carrier_in': 'Carrier-In',
            'inspecting_cases': f'Inspecting {self.case_count} cases on carrier',
            'carrier_out': 'Carrier-Out',
        }
        return labels.get(step, '')

    def status(self, step):
        if step in self.completed:
            return 'DONE'
        current = next((name for name in WORKFLOW_STEPS if name not in self.completed), None)
        return 'CURRENT' if step == current else 'PENDING'


@dataclass
class InspectionState:
    """Carrier state, changed by the main/UI thread only."""
    detection_state: object = 'WAITING'
    detection_result: object = None
    current_sealant_result: object = None
    sealant_detection_results: object = field(default_factory=dict)
    detection_triggered_time: object = 0
    absent_frame_count: object = 0
    result_hold_start_time: object = 0
    detection_start_time: object = 0
    camera_error_start_time: float = 0
    pending_review: object = None
    carrier_count: object = 0
    user_reviewed_flag: object = False
    current_raw_frame: object = None
    current_sealant_boxes: object = field(default_factory=list)
    current_scores: object = field(default_factory=dict)
    review_log_written: object = False


class InspectionWorkflow:
    """Composed service; context supplies shared application interfaces."""

    def __init__(self, context):
        self.context = context
        self.state = InspectionState()
        self.panel = WorkflowPanelState()

    def update_panel(self, carrier_present, case_count, carrier_released=False):
        """Update the persistent overlay model after processing a frame."""
        self.panel.update(
            carrier_present,
            case_count,
            self.context.detection_state,
            carrier_released,
        )

    def reset_camera_error(self, current_status):
        """Release the latched error; the next stop must pass the ROI check again."""
        ctx = self.context
        if ctx.detection_state != 'CAMERA_ERROR':
            return current_status
        duration = round(ctx.clock.time() - ctx.camera_error_start_time, 1)
        if ctx.logger:
            ctx.logger.log_system_event('camera_error_reset_by_operator', {
                'error_duration_seconds': duration,
            })
            ctx.logger.log_state_change('CAMERA_ERROR', 'WAITING', 'operator_reset_c_key')
        ctx.detection_state = 'WAITING'
        ctx.camera_error_start_time = 0
        ctx.absent_frame_count = 0
        self.panel.reset()
        return ctx._call_light_once('Standby')

    def _handle_objects_present(self, frame, raw_frame, boxes, categories, sealant_boxes, scores,
                               current_status, stop_time, previous_centroid):
        """Handle the case when carrier and sealant are detected"""
        ctx = self.context

        # === NEW: Reset absent frame counter when objects are present ===
        ctx.absent_frame_count = 0

        if ctx.detection_state == 'RESULT_HOLD':
            # The result remains live while the carrier is present. A temporary
            # obstruction can make it red, then all four passing cases turn it
            # green without requiring a new carrier cycle.
            sealant_result = ctx._cut_sealant_detection(raw_frame, sealant_boxes)
            new_result = 'NG' if sealant_result else 'Pass'
            ctx.current_sealant_result = sealant_result
            if new_result != ctx.detection_result:
                old_result = ctx.detection_result
                ctx.detection_result = new_result
                current_status = ctx._call_light_once(new_result)
                self._update_review_snapshot(
                    raw_frame, boxes, categories, sealant_boxes, scores)
                print(f"Detection result updated: {old_result} → {new_result}")
                if ctx.logger:
                    ctx.logger.log_system_event('detection_result_updated', {
                        'old_result': old_result,
                        'new_result': new_result,
                        'case_count': len(sealant_boxes),
                    })
            return current_status, stop_time, previous_centroid

        elif ctx.detection_state == 'WAITING':
            # Check if we should trigger detection
            # V6.0: Pass scores for confidence tracking
            return ctx._check_detection_trigger(
                frame, raw_frame, boxes, categories, sealant_boxes, scores,
                current_status, stop_time, previous_centroid
            )

        elif ctx.detection_state == 'DETECTED':
            # Detection already triggered, maintain status
            return current_status, stop_time, previous_centroid

        return current_status, stop_time, previous_centroid

    def _update_review_snapshot(self, raw_frame, boxes, categories, sealant_boxes, scores):
        """Store the latest live result so Carrier-Out logs the final status."""
        ctx = self.context
        ctx.current_raw_frame = raw_frame.copy()
        ctx.current_sealant_boxes = sealant_boxes.copy() if sealant_boxes else []
        ctx.current_scores = {}
        for box in sealant_boxes:
            box_key = f"{box[0]}_{box[1]}_{box[2]}_{box[3]}"
            for index, (detected_box, category) in enumerate(zip(boxes, categories)):
                if (category == ctx.class_map.get('sealant', 1)
                        and np.array_equal(detected_box, box)):
                    ctx.current_scores[box_key] = (
                        float(scores[index]) if index < len(scores) else 0.0)
                    break

        if ctx.pending_review is None:
            return
        ctx.pending_review['frame_number'] = ctx.frame_count
        ctx.pending_review['user_reviewed'] = False
        ctx.pending_review['sealants'] = []
        for index, box in enumerate(sealant_boxes):
            box_key = f"{box[0]}_{box[1]}_{box[2]}_{box[3]}"
            is_broken = ctx.sealant_detection_results.get(box_key, True)
            ctx.pending_review['sealants'].append({
                'index': index,
                'bbox': [int(box[0]), int(box[1]),
                         int(box[2] - box[0]), int(box[3] - box[1])],
                'ai_result': 'NG' if is_broken else 'Pass',
                'confidence': ctx.current_scores.get(box_key, 0.0),
                'user_agrees': True,
            })

    def _handle_objects_absent(self, current_status, stop_time, previous_centroid):
        """Handle the case when carrier or sealant are not detected"""
        ctx = self.context

        # === NEW: Increment absent frame counter ===
        ctx.absent_frame_count += 1

        # Get tolerance settings from config
        absent_tolerance = ctx.detection_config.get('absent_tolerance_frames', 5)
        result_hold_min = ctx.detection_config.get('result_hold_min_seconds', 2.0)

        if ctx.detection_state == 'RESULT_HOLD':
            # === NEW: Check tolerance before resetting ===
            # Check 1: Are we still within the absent tolerance?
            if ctx.absent_frame_count < absent_tolerance:
                # Still within tolerance, don't reset
                return current_status, stop_time, previous_centroid

            # Check 2: Are we still within the protection period?
            time_in_result_hold = ctx.clock.time() - ctx.result_hold_start_time
            if time_in_result_hold < result_hold_min:
                # Still within protection period, don't reset
                return current_status, stop_time, previous_centroid

            # V6.0: Write pending review log before leaving RESULT_HOLD
            ctx._write_pending_review_log()

            # V6.0: Close manual review window if open
            if ctx.manual_review_window and ctx.manual_review_window.is_open:
                ctx.manual_review_window.close()

            # Both checks passed, now reset
            old_state = ctx.detection_state
            ctx.detection_state = 'WAITING'
            ctx.detection_result = None

            print(f"Objects removed (absent {ctx.absent_frame_count} frames), state: {old_state} → {ctx.detection_state}")
            if ctx.logger:
                ctx.logger.log_state_change(old_state, ctx.detection_state,
                    f"objects_removed_after_{ctx.absent_frame_count}_frames")

            # Reset sealant detection state
            ctx.current_sealant_result = None
            ctx.sealant_detection_results = {}

            # V6.0: Reset review state for next carrier
            ctx.user_reviewed_flag = False
            ctx.pending_review = None
            ctx.current_raw_frame = None
            ctx.current_sealant_boxes = []
            ctx.current_scores = {}
            ctx.review_log_written = False

            # End detection session
            if ctx.logger and ctx.logger.detection_session_active:
                ctx.logger.log_detection_session_end()

            # Reset to standby light (yellow)
            current_status = ctx._call_light_once('Standby')

            # Reset timing
            stop_time = 0
            previous_centroid = [0, 0]

        elif ctx.detection_state in ['DETECTED', 'WAITING']:
            # === NEW: Check tolerance before resetting ===
            if ctx.absent_frame_count < absent_tolerance:
                # Still within tolerance, maintain stop_time (don't reset)
                return current_status, stop_time, previous_centroid

            # Objects disappeared beyond tolerance
            if ctx.detection_state == 'DETECTED':
                old_state = ctx.detection_state
                ctx.detection_state = 'WAITING'
                print(f"Objects disappeared during detection (absent {ctx.absent_frame_count} frames), state: {old_state} → {ctx.detection_state}")
                if ctx.logger:
                    ctx.logger.log_state_change(old_state, ctx.detection_state,
                        f"objects_disappeared_during_detection_after_{ctx.absent_frame_count}_frames")

            # Ensure light is on standby if not already
            if current_status != 'Standby':
                current_status = ctx._call_light_once('Standby')

            # Reset timing when truly absent
            stop_time = 0
            previous_centroid = [0, 0]

        return current_status, stop_time, previous_centroid

    def _check_detection_trigger(self, frame, raw_frame, boxes, categories, sealant_boxes, scores,
                               current_status, stop_time, previous_centroid):
        """Check if detection should be triggered (only called in WAITING state)"""
        ctx = self.context

        h, w = frame.shape[:2]

        carrier_pos = ctx._check_correct_carrier(categories, boxes)
        centroid = [
            int((carrier_pos[2] - carrier_pos[0]) / 2) + carrier_pos[0],
            int((carrier_pos[3] - carrier_pos[1]) / 2) + carrier_pos[1]
        ]

        cv2.circle(frame, centroid, 5, (0, 255, 0), 2)

        distance = np.linalg.norm(np.array(centroid) - np.array(previous_centroid))

        if centroid[1] > h/2 and distance < ctx.detection_config['distance_threshold_pixels']:
            stop_time += 1

            if stop_time / ctx.detection_config['fps'] > ctx.detection_config['stop_threshold_seconds']:
                print(f'Detection triggered after {round(stop_time/ctx.detection_config["fps"], 2)} seconds')

                cfg = ctx.camera_health_config
                roi_x = int(cfg.get('roi_x_ratio', 0.25) * w)
                roi_y = int(cfg.get('roi_y_ratio', 0.25) * h)
                roi_w = int(cfg.get('roi_w_ratio', 0.50) * w)
                roi_h = int(cfg.get('roi_h_ratio', 0.50) * h)
                if not (roi_x <= centroid[0] <= roi_x + roi_w
                        and roi_y <= centroid[1] <= roi_y + roi_h):
                    old_state = ctx.detection_state
                    ctx.detection_state = 'CAMERA_ERROR'
                    ctx.camera_error_start_time = ctx.clock.time()
                    if ctx.logger:
                        ctx.logger.log_state_change(old_state, 'CAMERA_ERROR',
                                                   'carrier_centroid_outside_roi')
                        ctx.logger.log_system_event('camera_misalignment_detected', {
                            'centroid': centroid,
                            'roi': {'x': roi_x, 'y': roi_y, 'w': roi_w, 'h': roi_h},
                        })
                    print('[CAMERA_HEALTH] Carrier stopped outside ROI; inspect alignment and press C')
                    return ctx._call_light_once('Standby'), 0, centroid

                # === NEW: Record detection start time for cycle time calculation ===
                ctx.detection_start_time = ctx.clock.time()

                # === STATE TRANSITION: WAITING → DETECTED ===
                old_state = ctx.detection_state
                ctx.detection_state = 'DETECTED'
                if ctx.logger:
                    ctx.logger.log_state_change(old_state, ctx.detection_state, "trigger_conditions_met")

                # Start detection session
                if ctx.logger and not ctx.logger.detection_session_active:
                    ctx.logger.log_detection_session_start()

                # Initial evaluation; RESULT_HOLD continues live evaluation.
                sealant_result = ctx._cut_sealant_detection(raw_frame, sealant_boxes)
                ctx.current_sealant_result = sealant_result

                # Log detection results
                ctx._log_detection_results(sealant_result, frame.shape, centroid)

                # === STATE TRANSITION: DETECTED → RESULT_HOLD ===
                old_state = ctx.detection_state
                ctx.detection_state = 'RESULT_HOLD'

                # === NEW: Record time entering RESULT_HOLD for protection period ===
                ctx.result_hold_start_time = ctx.clock.time()

                if ctx.logger:
                    ctx.logger.log_state_change(old_state, ctx.detection_state, "detection_completed")

                # Set result and update lights ONCE
                ctx.carrier_count += 1
                if not sealant_result:
                    ctx.detection_result = 'Pass'
                    current_status = ctx._call_light_once('Pass')
                    print("Detection result: PASS - GREEN requested")
                else:
                    ctx.detection_result = 'NG'
                    current_status = ctx._call_light_once('NG')
                    print("Detection result: NG - RED requested")

                # Save detection image
                ctx._save_img_data(raw_frame, sealant_boxes, sealant_result)

                # === NEW: Write statistics log (CSV) ===
                if ctx.statistics_logger:
                    cycle_time_ms = int((ctx.clock.time() - ctx.detection_start_time) * 1000)
                    total_count = len(ctx.sealant_detection_results)
                    ng_count = sum(1 for is_broken in ctx.sealant_detection_results.values() if is_broken)
                    ok_count = total_count - ng_count
                    ctx.statistics_logger.log_detection(
                        sealant_result, ok_count, ng_count, total_count, cycle_time_ms
                    )

                # === V6.0: Build pending_review data for rejudge log ===
                ctx.user_reviewed_flag = False  # Reset for new carrier

                # Build pending_review structure
                ctx.pending_review = {
                    'carrier_id': ctx.carrier_count,
                    'frame_number': ctx.frame_count,
                    'user_reviewed': False,
                    'sealants': []
                }
                self._update_review_snapshot(
                    raw_frame, boxes, categories, sealant_boxes, scores)

                print(f"[V6.0] Carrier #{ctx.carrier_count} pending review created with {len(ctx.pending_review['sealants'])} sealants")

                stop_time = 0
        else:
            stop_time = 0

        return current_status, stop_time, centroid
