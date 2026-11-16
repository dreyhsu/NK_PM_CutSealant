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
                'source_index': i  # Original index in pending_review['sealants']
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
        button_area_height = 80

        scale = min(
            (max_canvas_width - 2 * self.padding) / max(orig_width, 1),
            (max_canvas_height - 2 * self.padding - button_area_height) / max(orig_height, 1),
            2.0  # Max scale
        )
        scale = max(scale, 0.5)  # Min scale

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
            label = f"S{i + 1}"
            label_pos = (x + 5, y + 20)
            cv2.putText(self.canvas, label, label_pos, cv2.FONT_HERSHEY_SIMPLEX,
                       0.6, (255, 255, 255), 2, cv2.LINE_AA)

        # Draw buttons at bottom
        button_y = self.canvas_height - 60
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

        # Draw instruction text
        instruction = "Click sealant to toggle agree/disagree | Enter=Confirm | ESC=Cancel"
        cv2.putText(self.canvas, instruction, (10, self.canvas_height - 5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)

    def _mouse_callback(self, event, x, y, flags, param):
        """Handle mouse click events"""
        if event != cv2.EVENT_LBUTTONDOWN:
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

    def update(self):
        """Update display and handle keyboard input. Returns: 'confirm', 'cancel', or None"""
        if not self.is_open or self.canvas is None:
            return None

        cv2.imshow(self.window_name, self.canvas)
        key = cv2.waitKey(1) & 0xFF

        # Keyboard input sets pending_action
        if key == 13:   # Enter
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
        try:
            cv2.destroyWindow(self.window_name)
        except Exception:
            pass


class ReviewController:
    """Composed service; context supplies shared application interfaces."""

    def __init__(self, context):
        self.context = context

    def _open_manual_review(self):
        """Open manual review window (V6.0)"""
        ctx = self.context
        if ctx.detection_state != 'RESULT_HOLD':
            print("[V6.0] Cannot open review: not in RESULT_HOLD state")
            return

        if not ctx.pending_review or ctx.current_raw_frame is None:
            print("[V6.0] Cannot open review: no pending data")
            return

        # Mark that user pressed M key
        ctx.user_reviewed_flag = True
        ctx.pending_review['user_reviewed'] = True

        print(f"[V6.0] Opening manual review for Carrier #{ctx.pending_review['carrier_id']}")

        # If fullscreen is active, exit it so the review window can appear on top
        if ctx.is_fullscreen:
            ctx._toggle_fullscreen(ctx.display_config['window_name'])
            ctx.fullscreen_paused_for_review = True
            print("[V6.0] Fullscreen paused for manual review")

        # Create manual review window if needed
        padding = ctx.rejudge_config.get('review_window_padding', 50)
        thickness = ctx.rejudge_config.get('review_border_thickness', 3)

        if not ctx.manual_review_window:
            ctx.manual_review_window = ManualReviewWindow(padding=padding, border_thickness=thickness)

        # Prepare sealant boxes as list of arrays
        sealant_boxes_list = []
        for sealant in ctx.pending_review['sealants']:
            bbox = sealant['bbox']
            # Convert [x, y, w, h] back to [x1, y1, x2, y2]
            x1, y1 = bbox[0], bbox[1]
            x2, y2 = bbox[0] + bbox[2], bbox[1] + bbox[3]
            sealant_boxes_list.append(np.array([x1, y1, x2, y2]))

        # Build confidence dict
        confidences = {}
        for sealant in ctx.pending_review['sealants']:
            bbox = sealant['bbox']
            x1, y1 = bbox[0], bbox[1]
            x2, y2 = bbox[0] + bbox[2], bbox[1] + bbox[3]
            box_key = f"{x1}_{y1}_{x2}_{y2}"
            confidences[box_key] = sealant.get('confidence', 0.0)

        # Open review window, passing current pending_review states to restore previous user_agrees
        previous_states = ctx.pending_review.get('sealants', None)
        success = ctx.manual_review_window.open(
            ctx.current_raw_frame,
            sealant_boxes_list,
            ctx.sealant_detection_results,
            confidences,
            previous_states=previous_states
        )

        if success:
            print("[V6.0] Review window opened")
            if ctx.logger:
                ctx.logger.log_system_event("manual_review_opened", {
                    "carrier_id": ctx.pending_review['carrier_id'],
                    "sealant_count": len(ctx.pending_review['sealants'])
                })
        else:
            print("[V6.0] Failed to open review window")

    def _handle_review_complete(self, result):
        """Handle manual review completion (V6.0)"""
        ctx = self.context
        if not ctx.manual_review_window or not ctx.pending_review:
            return

        if result == 'confirm':
            # Get user modifications from review window
            review_results = ctx.manual_review_window.get_results()

            # Update pending_review using source_index to handle skipped crops correctly
            for state in review_results:
                source_index = state.get('source_index')
                if source_index is not None and source_index < len(ctx.pending_review['sealants']):
                    ctx.pending_review['sealants'][source_index]['user_agrees'] = state.get('user_agrees', True)

            # Count modifications
            disagree_count = sum(1 for s in ctx.pending_review['sealants'] if not s.get('user_agrees', True))
            print(f"[V6.0] Review confirmed: {disagree_count} modifications")

            if ctx.logger:
                ctx.logger.log_system_event("manual_review_confirmed", {
                    "carrier_id": ctx.pending_review['carrier_id'],
                    "modifications": disagree_count
                })

        elif result == 'cancel':
            # Reset all to agree (auto-agree)
            for sealant in ctx.pending_review['sealants']:
                sealant['user_agrees'] = True

            print("[V6.0] Review cancelled - auto-agree applied")

            if ctx.logger:
                ctx.logger.log_system_event("manual_review_cancelled", {
                    "carrier_id": ctx.pending_review['carrier_id']
                })

        # Close the review window
        ctx.manual_review_window.close()

        # Restore fullscreen if it was paused for review
        if ctx.fullscreen_paused_for_review:
            ctx._toggle_fullscreen(ctx.display_config['window_name'])
            ctx.fullscreen_paused_for_review = False
            print("[V6.0] Fullscreen restored after manual review")

    def _write_pending_review_log(self):
        """Write pending review data to rejudge CSV log (V6.0)"""
        ctx = self.context
        if not ctx.pending_review:
            return

        if ctx.review_log_written:
            return

        if not ctx.rejudge_logger:
            print("[V6.0] Warning: rejudge_logger not initialized")
            return

        # Close review window if still open
        if ctx.manual_review_window and ctx.manual_review_window.is_open:
            # Treat as cancel - auto agree
            for sealant in ctx.pending_review['sealants']:
                sealant['user_agrees'] = True
            ctx.manual_review_window.close()

        # Write to CSV
        ctx.rejudge_logger.write_records(ctx.pending_review, ctx.source_name)
        ctx.review_log_written = True

        if ctx.logger:
            ctx.logger.log_system_event("rejudge_log_written", {
                "carrier_id": ctx.pending_review['carrier_id'],
                "user_reviewed": ctx.pending_review.get('user_reviewed', False),
                "sealant_count": len(ctx.pending_review['sealants'])
            })
