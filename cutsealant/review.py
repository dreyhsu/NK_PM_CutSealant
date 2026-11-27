"""Review functionality extracted from CutSealant V6.0.7."""
import cv2
import numpy as np

class ManualReviewWindow:
    """Interactive manual review window for sealant inspection (V6.0)"""

    def __init__(self, padding=50, border_thickness=3):
        self.padding = padding
        self.border_thickness = border_thickness
        self.window_name = "Manual Review - Press Enter to Confirm, ESC to Cancel"
        self.is_open = False

        # Review data
        self.sealant_crops = []      # List of cropped images
        self.sealant_rects = []      # List of (x, y, w, h) in display coordinates
        self.sealant_states = []     # List of {'ai_result': str, 'user_agrees': bool}
        self.original_positions = [] # Original positions for layout

        # Display canvas
        self.canvas = None
        self.canvas_width = 0
        self.canvas_height = 0

        # Button areas
        self.confirm_btn_rect = None
        self.cancel_btn_rect = None

        # Pending action from mouse click (set by _on_confirm/_on_cancel, consumed by update())
        self.pending_action = None

    def open(self, frame, sealant_boxes, sealant_results, confidences, previous_states=None):
        """
        Open the review window with sealant crops

        Args:
            frame: Original frame (numpy array)
            sealant_boxes: List of [x1, y1, x2, y2] boxes
            sealant_results: Dict mapping box_key to is_broken (True=NG, False=Pass)
            confidences: Dict mapping box_key to confidence score
            previous_states: List of previous sealant states from pending_review (optional)
                             Used to restore user_agrees when re-opening the window
        """
        if len(sealant_boxes) == 0:
            print("[ManualReview] No sealants to review")
            return False

        self.pending_action = None
        self.sealant_crops = []
        self.sealant_states = []
        self.original_positions = []

        # Extract crops and initialize states
        for i, box in enumerate(sealant_boxes):
            x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])

            # Ensure valid crop area
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(frame.shape[1], x2)
            y2 = min(frame.shape[0], y2)

            if x2 <= x1 or y2 <= y1:
                continue

            crop = frame[y1:y2, x1:x2].copy()
            self.sealant_crops.append(crop)

            # Get AI result for this sealant
            box_key = f"{box[0]}_{box[1]}_{box[2]}_{box[3]}"
            is_broken = sealant_results.get(box_key, False)
            ai_result = 'NG' if is_broken else 'Pass'
            confidence = confidences.get(box_key, 0.0)

            # Restore previous user_agrees if available, otherwise default to True
            if previous_states and i < len(previous_states):
                user_agrees = previous_states[i].get('user_agrees', True)
            else:
                user_agrees = True

            self.sealant_states.append({
                'ai_result': ai_result,
                'user_agrees': user_agrees,
                'confidence': confidence,
                'bbox': [x1, y1, x2 - x1, y2 - y1],  # x, y, w, h
                'source_index': i,
                'position_id': f'{i + 1:02}'
            })

            # Store original center position for layout
            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2
            self.original_positions.append((center_x, center_y, x2 - x1, y2 - y1))

        if len(self.sealant_crops) == 0:
            print("[ManualReview] No valid crops extracted")
            return False

        # Create layout and draw window
        self._create_layout()
        self._draw_window()

        self.is_open = True
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(self.window_name, self._mouse_callback)

        return True

    def open_cases(self, cases):
        """Open immutable per-position evidence captured on different frames."""
        if not cases:
            return False
        self.pending_action = None
        self.sealant_crops = []
        self.sealant_states = []
        self.original_positions = []
        reference = cases[0]
        rx1, ry1, rx2, ry2 = reference['relative_bbox']
        reference_width = reference['crop'].shape[1] / (rx2 - rx1)
        reference_height = reference['crop'].shape[0] / (ry2 - ry1)
        for case in cases:
            crop = case['crop']
            if crop is None or crop.size == 0:
                continue
            index = case['index']
            x1, y1, x2, y2 = case['relative_bbox']
            self.sealant_crops.append(crop.copy())
            self.sealant_states.append({'source_index': index, 'position_id': f'{index + 1:02}',
                                       'ai_result': 'NG', 'user_agrees': True,
                                       'confidence': case['confidence']})
            self.original_positions.append(((x1+x2)*reference_width/2,
                                            (y1+y2)*reference_height/2,
                                            crop.shape[1], crop.shape[0]))
        if not self.sealant_crops:
            return False
        self._create_layout()
        self._draw_window()
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(self.window_name, self._mouse_callback)
        self.is_open = True
        return True

    def _create_layout(self):
        """Create layout preserving relative spatial positions"""
        if len(self.sealant_crops) == 0:
            return

        # Find bounding box of all original positions
        min_x = min(p[0] - p[2]/2 for p in self.original_positions)
        max_x = max(p[0] + p[2]/2 for p in self.original_positions)
        min_y = min(p[1] - p[3]/2 for p in self.original_positions)
        max_y = max(p[1] + p[3]/2 for p in self.original_positions)

        orig_width = max_x - min_x
        orig_height = max_y - min_y

        # Calculate scale to fit in reasonable window size
        max_canvas_width = 1200
        max_canvas_height = 800
        button_area_height = 105

        scale = min(
            (max_canvas_width - 2 * self.padding) / max(orig_width, 1),
            (max_canvas_height - 2 * self.padding - button_area_height) / max(orig_height, 1),
            2.0  # Max scale
        )
        scale = max(scale, 0.001)

        # Calculate canvas size
        self.canvas_width = int(orig_width * scale + 2 * self.padding)
        self.canvas_height = int(orig_height * scale + 2 * self.padding + button_area_height)

        # Ensure minimum size
        self.canvas_width = max(self.canvas_width, 400)
        self.canvas_height = max(self.canvas_height, 300)

        # Calculate display rectangles for each sealant
        self.sealant_rects = []
        for i, (cx, cy, w, h) in enumerate(self.original_positions):
            # Transform to canvas coordinates
            display_cx = (cx - min_x) * scale + self.padding
            display_cy = (cy - min_y) * scale + self.padding
            display_w = w * scale
            display_h = h * scale

            x = int(display_cx - display_w / 2)
            y = int(display_cy - display_h / 2)

            self.sealant_rects.append((x, y, int(display_w), int(display_h)))

    def _get_border_color(self, index):
        """
        Get border color based on AI result and user agreement

        Rules (BGR format):
        - AI=Pass, Agree=True  -> Green (0, 255, 0)
        - AI=NG,   Agree=True  -> Red (0, 0, 255)
        - Any,     Agree=False -> Yellow (0, 255, 255)
        """
        state = self.sealant_states[index]

        if not state['user_agrees']:
            return (0, 255, 255)  # Yellow - disagreed
        elif state['ai_result'] == 'Pass':
            return (0, 255, 0)    # Green - Pass agreed
        else:
            return (0, 0, 255)    # Red - NG agreed

    def _draw_window(self):
        """Draw the review window canvas"""
        # Create black canvas
        self.canvas = np.zeros((self.canvas_height, self.canvas_width, 3), dtype=np.uint8)

        # Draw each sealant crop with border
        for i, (crop, rect) in enumerate(zip(self.sealant_crops, self.sealant_rects)):
            x, y, w, h = rect

            # Ensure we don't exceed canvas bounds
            if x < 0 or y < 0 or x + w > self.canvas_width or y + h > self.canvas_height:
                continue

            # Resize crop to fit rectangle
            try:
                resized_crop = cv2.resize(crop, (w, h), interpolation=cv2.INTER_LINEAR)
                self.canvas[y:y+h, x:x+w] = resized_crop
            except Exception as e:
                print(f"[ManualReview] Error resizing crop {i}: {e}")
                continue

            # Draw border
            color = self._get_border_color(i)
            cv2.rectangle(self.canvas, (x, y), (x + w, y + h), color, self.border_thickness)

            # Draw label (S1, S2, ...)
            label = f"Position {self.sealant_states[i]['position_id']}"
            label_pos = (x + 5, y + 20)
            cv2.putText(self.canvas, label, label_pos, cv2.FONT_HERSHEY_SIMPLEX,
                       0.6, (255, 255, 255), 2, cv2.LINE_AA)

        # Draw buttons at bottom
        button_y = self.canvas_height - 85
        button_height = 40
        button_width = 120

        # Confirm button (green)
        confirm_x = self.canvas_width // 2 - button_width - 20
        self.confirm_btn_rect = (confirm_x, button_y, button_width, button_height)
        cv2.rectangle(self.canvas,
                     (confirm_x, button_y),
                     (confirm_x + button_width, button_y + button_height),
                     (0, 180, 0), -1)
        cv2.putText(self.canvas, "Confirm", (confirm_x + 15, button_y + 28),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # Cancel button (gray)
        cancel_x = self.canvas_width // 2 + 20
        self.cancel_btn_rect = (cancel_x, button_y, button_width, button_height)
        cv2.rectangle(self.canvas,
                     (cancel_x, button_y),
                     (cancel_x + button_width, button_y + button_height),
                     (100, 100, 100), -1)
        cv2.putText(self.canvas, "Cancel", (cancel_x + 20, button_y + 28),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        for instruction, baseline in (
                ('Click crop: agree NG / disagree Pass', self.canvas_height - 23),
                ('Enter: Confirm | Esc: Cancel | X: Terminate', self.canvas_height - 7)):
            cv2.putText(self.canvas, instruction, (10, baseline),
                        cv2.FONT_HERSHEY_SIMPLEX, .4, (200, 200, 200), 1)

    def _mouse_callback(self, event, x, y, flags, param):
        """Handle mouse click events"""
        if not self.is_open or event != cv2.EVENT_LBUTTONDOWN:
            return

        # Check if clicked on a sealant
        for i, rect in enumerate(self.sealant_rects):
            rx, ry, rw, rh = rect
            if rx <= x <= rx + rw and ry <= y <= ry + rh:
                # Toggle user_agrees
                self.sealant_states[i]['user_agrees'] = not self.sealant_states[i]['user_agrees']
                state_str = "Agree" if self.sealant_states[i]['user_agrees'] else "Disagree"
                print(f"[ManualReview] S{i+1} toggled to: {state_str}")
                self._draw_window()
                return

        # Check if clicked on Confirm button
        if self.confirm_btn_rect:
            bx, by, bw, bh = self.confirm_btn_rect
            if bx <= x <= bx + bw and by <= y <= by + bh:
                self._on_confirm()
                return

        # Check if clicked on Cancel button
        if self.cancel_btn_rect:
            bx, by, bw, bh = self.cancel_btn_rect
            if bx <= x <= bx + bw and by <= y <= by + bh:
                self._on_cancel()
                return

    def _on_confirm(self):
        """Handle confirm action"""
        print("[ManualReview] Confirmed - saving user modifications")
        self.pending_action = 'confirm'

    def _on_cancel(self):
        """Handle cancel action - reset all to agree"""
        print("[ManualReview] Cancelled - resetting to auto-agree")
        self.pending_action = 'cancel'

    def update(self, key=None):
        """Update display using the main loop's single keyboard event. Returns: 'confirm', 'cancel', or None"""
        if not self.is_open or self.canvas is None:
            return None

        window_closed = False
        try:
            if cv2.getWindowProperty(self.window_name, cv2.WND_PROP_VISIBLE) < 1:
                self.pending_action = 'cancel'
                window_closed = True
            else:
                cv2.imshow(self.window_name, self.canvas)
        except cv2.error:
            self.pending_action = 'cancel'
            window_closed = True

        # Keyboard input sets pending_action
        if key == 13 and not window_closed:   # Enter
            self.pending_action = 'confirm'
        elif key == 27: # ESC
            self.pending_action = 'cancel'

        # Consume pending_action (from mouse click or keyboard)
        if self.pending_action is not None:
            action = self.pending_action
            self.pending_action = None

            # Handle cancel: reset all user_agrees
            if action == 'cancel':
                for state in self.sealant_states:
                    state['user_agrees'] = True

            # Close window
            self.is_open = False
            try:
                cv2.destroyWindow(self.window_name)
            except Exception:
                pass

            return action

        return None

    def get_results(self):
        """Get review results as list of sealant states"""
        return self.sealant_states

    def close(self):
        """Close the review window"""
        self.is_open = False
        self.pending_action = None
        self.sealant_crops = []
        self.sealant_states = []
        self.sealant_rects = []
        self.original_positions = []
        self.canvas = None
        self.confirm_btn_rect = self.cancel_btn_rect = None
        try:
            cv2.destroyWindow(self.window_name)
        except Exception:
            pass


class ReviewController:
    """Queue frozen failure evidence and apply decisions to stable position IDs."""

    def __init__(self, context):
        self.context = context
        self.review_cycle_id = None
        self.review_indices = []

    def open_queued(self):
        workflow = self.context.workflow
        if workflow.review_queue:
            self._open_manual_review(automatic=True)

    def _open_manual_review(self, automatic=False):
        ctx = self.context
        workflow = ctx.workflow
        if not workflow.active or (ctx.manual_review_window and ctx.manual_review_window.is_open):
            return
        indices = sorted(i for i, case in workflow.positions.items()
                         if case.needs_review and (not automatic or i in workflow.review_queue))
        cases = [workflow.positions[i] for i in indices if workflow.positions[i].crop is not None]
        if not cases:
            return
        if ctx.is_fullscreen:
            ctx._toggle_fullscreen(ctx.display_config['window_name'])
            ctx.fullscreen_paused_for_review = True
        if ctx.manual_review_window is None:
            ctx.manual_review_window = ManualReviewWindow(
                padding=ctx.rejudge_config.get('review_window_padding', 50),
                border_thickness=ctx.rejudge_config.get('review_border_thickness', 3))
        records = [{'index': case.index, 'crop': case.crop.copy(),
                    'confidence': case.confidence, 'relative_bbox': case.relative_bbox}
                   for case in cases]
        if ctx.manual_review_window.open_cases(records):
            self.review_cycle_id = ctx.carrier_count
            self.review_indices = [case.index for case in cases]
            workflow.review_queue.difference_update(self.review_indices)
            if ctx.logger:
                ctx.logger.log_system_event('manual_review_opened', {
                    'carrier_id': ctx.carrier_count,
                    'position_ids': [f'{i + 1:02}' for i in self.review_indices]})

    def _handle_review_complete(self, result):
        ctx = self.context
        workflow = ctx.workflow
        if not workflow.active or self.review_cycle_id != ctx.carrier_count:
            self.close()
            return
        if result == 'confirm':
            for state in ctx.manual_review_window.get_results():
                index = state['source_index']
                if index not in self.review_indices or index not in workflow.positions:
                    continue
                case = workflow.positions[index]
                if not case.needs_review:
                    continue
                case.status = 'NG' if state.get('user_agrees', True) else 'PASS'
                case.decision_source = 'manual'
                case.user_reviewed = True
            ctx.user_reviewed_flag = True
        if ctx.logger:
            ctx.logger.log_system_event('manual_review_' + ('confirmed' if result == 'confirm' else 'cancelled'), {
                'carrier_id': ctx.carrier_count,
                'position_ids': [f'{i + 1:02}' for i in self.review_indices]})
        workflow._sync_pending_review()
        self.close()

    def close(self):
        ctx = self.context
        if ctx.manual_review_window:
            ctx.manual_review_window.close()
        self.review_cycle_id = None
        self.review_indices = []
        if ctx.fullscreen_paused_for_review:
            ctx._toggle_fullscreen(ctx.display_config['window_name'])
            ctx.fullscreen_paused_for_review = False

    def _write_pending_review_log(self):
        ctx = self.context
        if ctx.pending_review is None or ctx.review_log_written:
            return
        # Unconfirmed edits never overwrite already committed position decisions.
        if ctx.manual_review_window and ctx.manual_review_window.is_open:
            self.close()
        if ctx.rejudge_logger:
            ctx.rejudge_logger.write_records(ctx.pending_review, ctx.source_name)
        ctx.review_log_written = True
