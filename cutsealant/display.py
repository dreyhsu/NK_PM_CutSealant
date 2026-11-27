"""Display functionality extracted from CutSealant V6.0.7."""
import os
import tkinter as tk
from datetime import datetime

import cv2
import numpy as np

from .workflow import WORKFLOW_STEPS

PANEL_BG = (9, 15, 26)
PANEL_WHITE = (255, 242, 235)
PANEL_MUTED = (189, 163, 145)
PANEL_GREEN = (138, 212, 54)
PANEL_AMBER = (84, 196, 245)

class Display:
    """Composed service; context supplies shared application interfaces."""

    def __init__(self, context):
        self.context = context

    def _check_correct_carrier(self, categories, boxes):
        """Check for correct carrier detection"""
        ctx = self.context
        carrier_indices = [i for i, cat in enumerate(categories) if cat == ctx.class_map['carrier']]
        if len(carrier_indices) == 1:
            return boxes[carrier_indices[0]]
        elif len(carrier_indices) > 1:
            # Multiple carriers detected, choose the largest one
            areas = []
            for idx in carrier_indices:
                box = boxes[idx]
                area = (box[2] - box[0]) * (box[3] - box[1])
                areas.append(area)
            largest_idx = carrier_indices[np.argmax(areas)]
            return boxes[largest_idx]
        return None

    def _calculate_iou(self, box1, box2):
        """
        Calculate Intersection over Union (IoU) between two boxes
        Args:
            box1: [x1, y1, x2, y2] format
            box2: [x1, y1, x2, y2] format or box_key string "x1_y1_x2_y2"
        Returns:
            IoU value (0.0 to 1.0)
        """
        ctx = self.context
        # Handle box_key string format
        if isinstance(box2, str):
            coords = box2.split('_')
            box2 = [int(coords[0]), int(coords[1]), int(coords[2]), int(coords[3])]

        # Calculate intersection area
        x1_inter = max(box1[0], box2[0])
        y1_inter = max(box1[1], box2[1])
        x2_inter = min(box1[2], box2[2])
        y2_inter = min(box1[3], box2[3])

        # Check if there is intersection
        if x2_inter < x1_inter or y2_inter < y1_inter:
            return 0.0

        intersection_area = (x2_inter - x1_inter) * (y2_inter - y1_inter)

        # Calculate union area
        box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
        box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union_area = box1_area + box2_area - intersection_area

        # Calculate IoU
        if union_area == 0:
            return 0.0

        iou = intersection_area / union_area
        return iou

    def _find_matching_box(self, current_box, stored_box_keys, iou_threshold=0.75):
        """
        Find matching box from stored results using IoU matching
        Args:
            current_box: Current detection box [x1, y1, x2, y2]
            stored_box_keys: Dictionary keys of stored detection results
            iou_threshold: Minimum IoU to consider as a match (default: 0.75)
        Returns:
            Matching box_key if found, None otherwise
        """
        ctx = self.context
        best_iou = 0.0
        best_match = None

        for box_key in stored_box_keys:
            iou = ctx._calculate_iou(current_box, box_key)
            if iou > best_iou and iou >= iou_threshold:
                best_iou = iou
                best_match = box_key

        return best_match

    def _draw_detections(self, frame, boxes, categories, correct_carrier_box):
        """Draw detection boxes and return sealant boxes"""
        ctx = self.context
        sealant_boxes = []

        for i, (box, category) in enumerate(zip(boxes, categories)):
            color = (0, 255, 0) if category == ctx.class_map['carrier'] else (255, 0, 0)

            # Only draw carrier if it's the correct one
            if category == ctx.class_map['carrier']:
                if np.array_equal(box, correct_carrier_box):
                    cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), color, 2)
                    #cv2.putText(frame, 'carrier', (box[0], box[1]-10),
                    #           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
            elif category == ctx.class_map.get('sealant', 1):
                index = ctx.workflow.position_index(correct_carrier_box, box)
                if index is None:
                    continue
                sealant_boxes.append(box)
                case = ctx.workflow.positions.get(index)
                color = ((0, 255, 0) if case and case.passed else
                         (0, 0, 255) if case and case.needs_review else (0, 255, 255))
                cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), color, 2)
                cv2.putText(frame, f'Position {index + 1:02}', (box[0], max(20, box[1]-8)),
                            cv2.FONT_HERSHEY_SIMPLEX, .6, color, 2, cv2.LINE_AA)

        return sealant_boxes

    def _get_screen_size(self):
        """Get actual screen size using tkinter (cached)"""
        ctx = self.context
        if ctx.cached_screen_size is not None:
            return ctx.cached_screen_size

        try:
            root = tk.Tk()
            root.withdraw()  # Hide the window
            screen_width = root.winfo_screenwidth()
            screen_height = root.winfo_screenheight()
            root.destroy()
            ctx.cached_screen_size = (screen_width, screen_height)
            return screen_width, screen_height
        except Exception:
            # Fallback to default values
            ctx.cached_screen_size = (1920, 1080)
            return 1920, 1080

    def _create_display_window(self, window_name):
        """Create display window with proper flags for fullscreen support"""
        ctx = self.context
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, 800, 600)  # Initial size

    def _display_frame(self, frame, window_name):
        """Display frame with auto-fit capability"""
        ctx = self.context
        try:
            display_frame = frame.copy()

            if ctx.fit_screen_mode and ctx.display_config.get('auto_fit_screen', True):
                # Get actual screen size (cached)
                screen_width, screen_height = ctx._get_screen_size()

                # Get frame dimensions
                frame_height, frame_width = display_frame.shape[:2]

                # Calculate scale to fit screen
                scale_x = screen_width * 0.85 / frame_width  # Use 85% of screen width
                scale_y = screen_height * 0.85 / frame_height  # Use 85% of screen height
                scale = min(scale_x, scale_y)

                # Apply min/max scale limits
                max_scale = ctx.display_config.get('max_scale', 2.0)
                min_scale = ctx.display_config.get('min_scale', 0.2)
                scale = max(min_scale, min(max_scale, scale))

                if scale != 1.0:
                    new_width = int(frame_width * scale)
                    new_height = int(frame_height * scale)
                    display_frame = cv2.resize(display_frame, (new_width, new_height), interpolation=cv2.INTER_AREA)

            # === Draw shortcut hints after scaling (fixed position/size regardless of zoom) ===
            dh, dw = display_frame.shape[:2]
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.55
            thickness = 1

            if ctx.detection_state == 'CAMERA_ERROR':
                self._draw_camera_error_overlay(display_frame)

            # --- Bottom bar: general shortcuts (white, always visible) ---
            shortcut_text = "[Q] Quit  [S] Screenshot  [F] Fit Screen  [G] Fullscreen  [P] Statistics  [R] Reset  [V] Recording  [M] Recheck  [X] Terminate"
            if ctx.detection_state == 'CAMERA_ERROR':
                shortcut_text = '[C] Reset camera alignment error  |  [Q] Quit'
            text_size = cv2.getTextSize(shortcut_text, font, font_scale, thickness)[0]
            bar_x = (dw - text_size[0]) // 2
            bar_y = dh - 12
            overlay = display_frame.copy()
            cv2.rectangle(overlay, (0, dh - 30), (dw, dh), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.5, display_frame, 0.5, 0, display_frame)
            cv2.putText(display_frame, shortcut_text, (bar_x, bar_y),
                        font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)

            # --- Manual recheck remains available during an active cycle. ---
            if ctx.workflow.active:
                m_text = "[M] Manual Review"
                m_size = cv2.getTextSize(m_text, font, 0.75, 2)[0]
                m_x = dw - m_size[0] - 30
                m_y = dh // 2
                overlay2 = display_frame.copy()
                pad = 8
                cv2.rectangle(overlay2,
                              (m_x - pad, m_y - m_size[1] - pad),
                              (m_x + m_size[0] + pad, m_y + pad),
                              (0, 0, 0), -1)
                cv2.addWeighted(overlay2, 0.5, display_frame, 0.5, 0, display_frame)
                cv2.putText(display_frame, m_text, (m_x, m_y),
                            font, 0.75, (255, 80, 0), 2, cv2.LINE_AA)

            cv2.imshow(window_name, display_frame)

        except Exception as e:
            error_msg = f"Display error: {e}"
            print(error_msg)
            if ctx.logger:
                ctx.logger.log_error("display_error", e)

    def _draw_camera_error_overlay(self, frame):
        """Blink a proportional red border and centered warning after resizing."""
        ctx = self.context
        cfg = ctx.camera_health_config
        interval = cfg.get('alert_blink_interval', 0.5)
        elapsed = ctx.clock.time() - ctx.camera_error_start_time
        if elapsed % (interval * 2) >= interval:
            return
        h, w = frame.shape[:2]
        border = max(1, int(min(w, h) * cfg.get('alert_border_thickness_ratio', 0.012)))
        cv2.rectangle(frame, (0, 0), (w - 1, h - 1), (0, 0, 255), border)
        font = cv2.FONT_HERSHEY_SIMPLEX
        text = 'Camera Out of Alignment!'
        scale = max(0.6, np.hypot(w, h) / 2203 * cfg.get('alert_font_scale_ratio', 0.04) * 25)
        # Keep the complete alert readable even in a narrow display window.
        text_width = cv2.getTextSize(text, font, scale, 2)[0][0]
        scale = min(scale, scale * max(1, w - 28) / max(1, text_width))
        thickness = max(1, int(scale * 2))
        (tw, th), baseline = cv2.getTextSize(text, font, scale, thickness)
        x, y = (w - tw) // 2, (h + th) // 2
        overlay = frame.copy()
        cv2.rectangle(overlay, (max(0, x - 14), max(0, y - th - 14)),
                      (min(w - 1, x + tw + 14), min(h - 1, y + baseline + 14)),
                      (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)
        cv2.putText(frame, text, (x, y), font, scale, (0, 0, 0), thickness + 2, cv2.LINE_AA)
        cv2.putText(frame, text, (x, y), font, scale, (255, 255, 255), thickness, cv2.LINE_AA)

    def _toggle_fit_screen(self):
        """Toggle fit screen mode on/off"""
        ctx = self.context
        ctx.fit_screen_mode = not ctx.fit_screen_mode
        mode_str = "ON" if ctx.fit_screen_mode else "OFF"
        print(f"Fit screen mode: {mode_str}")
        if ctx.logger:
            ctx.logger.log_system_event("fit_screen_toggled", {"mode": mode_str})

    def _take_screenshot(self, frame):
        """Take a screenshot and save to file"""
        ctx = self.context
        try:
            # Create screenshot directory if it doesn't exist
            if not os.path.exists(ctx.screenshot_path):
                os.makedirs(ctx.screenshot_path)

            # Generate filename with timestamp
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')[:-3]
            filename = f"{ctx.screenshot_path}screenshot_{timestamp}.jpg"

            # Save the frame
            cv2.imwrite(filename, frame)

            print(f"Screenshot saved: {filename}")
            if ctx.logger:
                ctx.logger.log_system_event("screenshot_saved", {"filename": filename})

        except Exception as e:
            error_msg = f"Screenshot error: {e}"
            print(error_msg)
            if ctx.logger:
                ctx.logger.log_error("screenshot_error", e)

    def _toggle_fullscreen(self, window_name):
        """Toggle fullscreen mode"""
        ctx = self.context
        try:
            ctx.is_fullscreen = not ctx.is_fullscreen
            if ctx.is_fullscreen:
                cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
                print("Fullscreen mode: ON")
            else:
                cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)
                print("Fullscreen mode: OFF")

            if ctx.logger:
                ctx.logger.log_system_event("fullscreen_toggled", {"fullscreen": ctx.is_fullscreen})
        except Exception as e:
            print(f"Fullscreen toggle error: {e}")

    def _draw_workflow_panel(self, frame, current_status=None):
        """Draw tower/time information above the top-right workflow card."""
        workflow = self.context.workflow.panel
        height, width = frame.shape[:2]
        scale = max(0.40, min(2.0, width / 1920, height / 1080))
        panel_width = min(width - 20, max(180, round(570 * scale)))
        row_height = max(24, round(48 * scale))
        header_height = max(58, round(92 * scale))
        status_height = max(62, round(88 * scale))
        gap = max(5, round(8 * scale))
        x = width - panel_width - 10
        status_y = 10
        y = status_y + status_height + gap
        panel_height = min(
            max(0, height - y - 10),
            header_height + row_height * (len(WORKFLOW_STEPS) + 4) + 8,
        )

        tower_status = (
            current_status
            or getattr(self.context, 'last_light_status', None)
            or 'Unknown'
        )
        tower_color = {
            'Standby': (0, 255, 255),
            'Tracking': (0, 255, 255),
            'Pass': (0, 255, 0),
            'NG': (0, 0, 255),
        }.get(tower_status, PANEL_WHITE)
        now = datetime.now()

        status_overlay = frame.copy()
        cv2.rectangle(
            status_overlay,
            (x, status_y),
            (x + panel_width, status_y + status_height),
            PANEL_BG,
            -1,
        )
        cv2.addWeighted(status_overlay, 0.90, frame, 0.10, 0, frame)

        status_font = max(0.31, 0.60 * scale)
        status_thickness = max(1, round(2 * scale))
        status_baseline = status_y + max(27, round(38 * scale))
        time_text = f"TIME {now.strftime('%H:%M:%S')}"
        light_mode = {
            'Standby': ' (STATIC)',
            'Tracking': ' (BLINK)',
        }.get(tower_status, '')
        tower = getattr(self.context, 'tower', None)
        health = getattr(tower, 'health', 'unknown')
        light_text = f'REQUEST: {str(tower_status).upper()}{light_mode}'
        status_font = min(status_font, (panel_width - 30) / max(1, cv2.getTextSize(
            light_text, cv2.FONT_HERSHEY_SIMPLEX, 1, status_thickness)[0][0]))
        light_width = cv2.getTextSize(
            light_text,
            cv2.FONT_HERSHEY_SIMPLEX,
            status_font,
            status_thickness,
        )[0][0]
        light_x = x + panel_width - light_width - max(13, round(16 * scale))
        dot_x = light_x - max(13, round(18 * scale))
        dot_y = status_baseline - max(4, round(6 * scale))
        pulse = 1.0 - abs(2.0 * now.microsecond / 1_000_000.0 - 1.0)
        dot_radius = max(3, round(6.0 * scale))
        dot_color = tower_color
        if tower_status == 'Tracking':
            brightness = 0.15 + 0.85 * pulse
            dot_color = tuple(round(channel * brightness) for channel in tower_color)
        cv2.circle(frame, (dot_x, dot_y), dot_radius + max(1, round(scale)),
                   PANEL_BG, -1, cv2.LINE_AA)
        cv2.circle(frame, (dot_x, dot_y), dot_radius, dot_color, -1, cv2.LINE_AA)
        transport_text = f'USB: {health.upper()}'
        if getattr(tower, 'buzzer_outcome', None) == 'failed_or_unknown':
            transport_text += ' BUZZER: ERROR'
        cv2.putText(frame, transport_text, (x + 12, status_y + status_height - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, max(0.25, status_font * .8),
                    PANEL_WHITE if health in ('connected', 'mock') else (0, 0, 255),
                    1, cv2.LINE_AA)
        cv2.putText(frame, time_text, (x + max(13, round(16 * scale)), status_baseline),
                    cv2.FONT_HERSHEY_SIMPLEX, status_font, PANEL_WHITE,
                    status_thickness, cv2.LINE_AA)
        cv2.putText(frame, light_text, (light_x, status_baseline),
                    cv2.FONT_HERSHEY_SIMPLEX, status_font, tower_color,
                    status_thickness, cv2.LINE_AA)

        overlay = frame.copy()
        cv2.rectangle(
            overlay,
            (x, y),
            (x + panel_width, y + panel_height),
            PANEL_BG,
            -1,
        )
        cv2.addWeighted(overlay, 0.90, frame, 0.10, 0, frame)

        font_scale = max(0.32, 0.68 * scale)
        thickness = max(1, round(2 * scale))

        def text(value, tx, ty, color=PANEL_WHITE, size=font_scale):
            available_width = max(1, x + panel_width - 12 - tx)
            text_width = cv2.getTextSize(value, cv2.FONT_HERSHEY_SIMPLEX, size, thickness)[0][0]
            size *= min(1, available_width / max(1, text_width))
            cv2.putText(
                frame,
                value,
                (int(tx), int(ty)),
                cv2.FONT_HERSHEY_SIMPLEX,
                size,
                color,
                thickness,
                cv2.LINE_AA,
            )

        phase_color = PANEL_GREEN if workflow.complete else PANEL_AMBER
        cv2.rectangle(
            frame,
            (x, y),
            (x + max(3, round(5 * scale)), y + panel_height),
            phase_color,
            -1,
        )
        text('WORKFLOW', x + 14, y + max(21, round(35 * scale)))
        progress = f'{workflow.completed_count} / {len(WORKFLOW_STEPS)}'
        progress_width = cv2.getTextSize(
            progress,
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            thickness,
        )[0][0]
        text(progress, x + panel_width - progress_width - 14,
             y + max(21, round(35 * scale)), phase_color)
        text(workflow.phase, x + 14, y + max(45, round(70 * scale)),
             phase_color, max(0.30, 0.59 * scale))

        rows_y = y + header_height
        for index, step in enumerate(WORKFLOW_STEPS):
            top = rows_y + index * row_height
            if top + row_height > y + panel_height:
                break
            status = workflow.status(step)
            color = (PANEL_GREEN if status == 'DONE'
                     else PANEL_AMBER if status == 'CURRENT'
                     else PANEL_MUTED)
            if status == 'CURRENT':
                row_overlay = frame.copy()
                cv2.rectangle(
                    row_overlay,
                    (x + 7, top),
                    (x + panel_width - 7, top + row_height),
                    (26, 43, 51),
                    -1,
                )
                cv2.addWeighted(row_overlay, 0.90, frame, 0.10, 0, frame)
            baseline = top + max(18, round(32 * scale))
            marker = 'OK' if status == 'DONE' else f'{index + 1:02}'
            text(marker, x + 14, baseline, color, max(0.30, 0.59 * scale))
            text(workflow.label(step), x + max(49, round(75 * scale)), baseline,
                 color, max(0.29, 0.56 * scale))

        for index, label in enumerate(workflow.case_labels):
            top = rows_y + (len(WORKFLOW_STEPS) + index) * row_height
            if top + row_height > y + panel_height:
                break
            color = (PANEL_GREEN if label in ('Automatic Pass', 'Manual Pass') else
                     (0, 0, 255) if label in ('Manual check required', 'Confirmed NG') else PANEL_MUTED)
            text(f'Position {index + 1:02}: {label}', x + 14,
                 top + max(18, round(32 * scale)), color, max(.29, .56 * scale))

    def _add_ui_elements(self, frame, current_status):
        """Add UI elements to frame"""
        ctx = self.context
        h, w = frame.shape[:2]

        # Add crosshair if enabled
        if ctx.display_config.get('show_crosshair', True):
            cv2.line(frame, (w//2-20, h//2), (w//2+20, h//2), (255, 255, 255), 1)
            cv2.line(frame, (w//2, h//2-20), (w//2, h//2+20), (255, 255, 255), 1)

        self._draw_workflow_panel(frame, current_status)
        self._draw_inference_roi(frame)

    def _draw_inference_roi(self, frame):
        """Use exactly the detector's scaled crop as the yellow placement outline."""
        roi = getattr(self.context, 'inference_roi', None)
        if roi is None:
            return
        x1, y1, x2, y2 = roi.scaled_bounds(frame.shape)
        height, width = frame.shape[:2]
        thickness = max(4, round(6 * min(width / roi.reference_size[0],
                                       height / roi.reference_size[1])))
        cv2.rectangle(frame, (x1, y1), (x2 - 1, y2 - 1), (0, 255, 255), thickness)
