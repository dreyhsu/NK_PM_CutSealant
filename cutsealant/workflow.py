"""Workflow functionality extracted from CutSealant V6.0.7."""
from dataclasses import dataclass, field

import cv2
import numpy as np

WORKFLOW_STEPS = (
    'board_in',
    'modules_detected',
    'inspecting',
    'inspection_finished',
    'board_out',
)


@dataclass
class WorkflowPanelState:
    """Persistent five-step cycle shown on every displayed frame."""

    completed: set = field(default_factory=set)
    module_count: int = 0
    cycle_started: bool = False
    cycle_complete: bool = False

    def reset(self):
        """Prepare the panel for a board that has not arrived yet."""
        self.completed.clear()
        self.module_count = 0
        self.cycle_started = False
        self.cycle_complete = False

    def update(self, board_present, module_count, detection_state, board_released=False):
        """Advance the display state from observations made for the current frame."""
        module_count = max(0, int(module_count))

        # Keep a completed cycle on screen until the next board starts a new one.
        if board_present and self.cycle_complete:
            self.reset()

        if board_present:
            self.cycle_started = True
            self.completed.add('board_in')
            # Retain the highest count so a one-frame detector flicker does not
            # make the displayed count move backwards.
            self.module_count = max(self.module_count, module_count)
            if self.module_count:
                self.completed.add('modules_detected')

        if detection_state == 'RESULT_HOLD' and self.cycle_started:
            self.completed.update(('inspecting', 'inspection_finished'))

        if board_released:
            if 'inspection_finished' in self.completed:
                self.completed.add('board_out')
                self.cycle_complete = True
            else:
                # An incomplete board departure is not a completed cycle.
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
            'board_in': 'Board-In',
            'modules_detected': f'{self.module_count} modules have been detected',
            'inspecting': 'Inspecting',
            'inspection_finished': 'Inspection Finished',
            'board_out': 'Board-Out',
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

    def update_panel(self, board_present, module_count, board_released=False):
        """Update the persistent overlay model after processing a frame."""
        self.panel.update(
            board_present,
            module_count,
            self.context.detection_state,
            board_released,
        )

    def _handle_objects_present(self, frame, raw_frame, boxes, categories, sealant_boxes, scores,
                               current_status, stop_time, previous_centroid):
        """Handle the case when carrier and sealant are detected"""
        ctx = self.context

        # === NEW: Reset absent frame counter when objects are present ===
        ctx.absent_frame_count = 0

        if ctx.detection_state == 'RESULT_HOLD':
            # 🔑 KEY LOGIC: Already have detection result, maintain state
            # No further detection, no light changes, just return current status
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

        carrier_idx = list(categories).index(ctx.class_map['carrier'])
        carrier_pos = boxes[carrier_idx]
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

                # 🔑 EXECUTE DETECTION ONLY ONCE
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
                if not sealant_result:
                    ctx.detection_result = 'Pass'
                    current_status = ctx._call_light_once('Pass')
                    print("Detection result: PASS - Light changed to GREEN")
                else:
                    ctx.detection_result = 'NG'
                    current_status = ctx._call_light_once('NG')
                    print("Detection result: NG - Light changed to RED")

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
                ctx.carrier_count += 1
                ctx.user_reviewed_flag = False  # Reset for new carrier
                ctx.current_raw_frame = raw_frame.copy()
                ctx.current_sealant_boxes = sealant_boxes.copy() if sealant_boxes else []

                # Build confidence scores mapping
                ctx.current_scores = {}
                for i, box in enumerate(sealant_boxes):
                    box_key = f"{box[0]}_{box[1]}_{box[2]}_{box[3]}"
                    # Find corresponding score from YOLO detection
                    # scores array corresponds to boxes array, need to find sealant indices
                    for j, (b, cat) in enumerate(zip(boxes, categories)):
                        if cat == ctx.class_map.get('sealant', 1):
                            if np.array_equal(b, box):
                                ctx.current_scores[box_key] = float(scores[j]) if j < len(scores) else 0.0
                                break

                # Build pending_review structure
                ctx.pending_review = {
                    'carrier_id': ctx.carrier_count,
                    'frame_number': ctx.frame_count,
                    'user_reviewed': False,
                    'sealants': []
                }

                for i, box in enumerate(sealant_boxes):
                    box_key = f"{box[0]}_{box[1]}_{box[2]}_{box[3]}"
                    is_broken = ctx.sealant_detection_results.get(box_key, False)
                    ai_result = 'NG' if is_broken else 'Pass'
                    confidence = ctx.current_scores.get(box_key, 0.0)

                    ctx.pending_review['sealants'].append({
                        'index': i,
                        'bbox': [int(box[0]), int(box[1]), int(box[2] - box[0]), int(box[3] - box[1])],
                        'ai_result': ai_result,
                        'confidence': confidence,
                        'user_agrees': True  # Default: agree with AI
                    })

                print(f"[V6.0] Carrier #{ctx.carrier_count} pending review created with {len(ctx.pending_review['sealants'])} sealants")

                stop_time = 0
        else:
            stop_time = 0

        return current_status, stop_time, centroid
