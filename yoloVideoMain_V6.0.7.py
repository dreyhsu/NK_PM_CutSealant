# -*- coding: utf-8 -*-
"""
Optimized version with detection state machine and reduced USB light calls
Version 6.0 - On-Demand Manual Review (按需人工複判)

Features:
- Detection only triggers once per object
- Light status maintained until objects are removed
- Significantly reduced USB communication
- Fixed: Light command failure now allows retry on next cycle
- Tolerance for detection box flickering (absent_tolerance_frames)
- RESULT_HOLD protection period (result_hold_min_seconds)
- CSV statistics log for production reporting
- TIS DFK industrial camera support (dfk_camera mode)
- 24/7 monitoring mode with enhanced fault tolerance
- Automatic camera reconnection
- Heartbeat logging for system health monitoring
- NEW (V6.0): On-demand manual review with M key
- NEW (V6.0): Interactive review window for sealant inspection
- NEW (V6.0): Rejudge CSV log with REPASS/REFAIL tracking

Keyboard shortcuts:
- Q: Quit program
- S: Take screenshot (saved to ./screenshots/)
- F: Toggle fit screen mode (auto-scale to screen size)
- G: Toggle fullscreen mode
- P: Print statistics
- R: Reset statistics
- V: Toggle video recording (if enabled in config)
- M: Open manual review window (only in RESULT_HOLD state)
"""

import threading
import queue
import configparser
import os
import signal
import sys
import csv
import tkinter as tk
from ultralytics import YOLO
from glob import glob
import cv2
import numpy as np
import subprocess
import time
import json
import logging
from datetime import datetime

# TIS Camera support (optional - only loaded when dfk_camera mode is used)
try:
    from tis_camera_wrapper import TISCameraWrapper
    TIS_AVAILABLE = True
except ImportError:
    TIS_AVAILABLE = False
    print("[INFO] TIS camera support not available (tis_camera_wrapper.py not found)")

# Global variables
frame_queue = queue.Queue(maxsize=2)
result_queue = queue.Queue(maxsize=2)


class JSONLogger:
    """JSON logger for detection results and system events"""
    
    def __init__(self, log_file_path, log_level='INFO'):
        self.log_file_path = log_file_path
        self.log_level = log_level.upper()
        
        # Create logs directory if it doesn't exist
        log_dir = os.path.dirname(log_file_path)
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir)
        
        # Setup standard logger for error messages
        self.setup_standard_logger()
        
        # Detection session tracking
        self.session_id = self._generate_session_id()
        self.detection_session_active = False
        
        # Log session start
        self.log_system_event("session_start", {
                              "session_id": self.session_id,
                              "log_level": self.log_level})
        
        
    def setup_standard_logger(self):
        """Setup standard Python logger for error messages"""
        # Create logger
        self.logger = logging.getLogger('EPS_Detection')
        self.logger.setLevel(getattr(logging, self.log_level))
        
        # Remove existing handlers to avoid duplicates
        self.logger.handlers.clear()
        
        # Create file handler for errors
        error_log_path = self.log_file_path.replace('.log', '_errors.log')
        file_handler = logging.FileHandler(error_log_path)
        file_handler.setLevel(logging.ERROR)
        
        # Create console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(getattr(logging, self.log_level))
        
        # Create formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        # Add handlers to logger
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)
    
    def _generate_session_id(self):
        """Generate unique session ID"""
        return datetime.now().strftime('%Y%m%d_%H%M%S')
    
    def _write_json_log(self, log_entry):
        """Write JSON log entry to file"""
        try:
            with open(self.log_file_path, 'a', encoding='utf-8') as f:
                json.dump(log_entry, f, ensure_ascii=False)
                f.write('\n')
        except Exception as e:
            self.logger.error(f"Failed to write JSON log: {e}")
    
    def log_system_event(self, event_type, data=None):
        """Log system events (startup, shutdown, errors)"""
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "session_id": self.session_id,
            "log_type": "system",
            "event_type": event_type,
            "data": data or {}
        }
        self._write_json_log(log_entry)
        
        # Also log to standard logger if it's an error
        if event_type == "error":
            self.logger.error(f"System Error: {data}")
    
    def log_detection_session_start(self):
        """Log start of detection session"""
        self.detection_session_active = True
        self.log_system_event("detection_session_start", {
            "message": "Entered sealant detection mode"
        })
    
    def log_detection_session_end(self):
        """Log end of detection session"""
        self.detection_session_active = False
        self.log_system_event("detection_session_end", {
            "message": "Exited sealant detection mode"
        })
    
    def log_detection_result(self, total_sealants, ok_count, ng_count, 
                           detection_details=None, frame_info=None):
        """Log sealant detection results"""
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "session_id": self.session_id,
            "log_type": "detection_result",
            "summary": {
                "total_sealants": total_sealants,
                "ok_count": ok_count,
                "ng_count": ng_count,
                "overall_result": "NG" if ng_count > 0 else "OK"
            },
            "detection_details": detection_details or [],
            "frame_info": frame_info or {}
        }
        self._write_json_log(log_entry)
        
        # Also log summary to standard logger
        self.logger.info(
            f"Detection Result - Total: {total_sealants}, "
            f"OK: {ok_count}, NG: {ng_count}"
        )
    
    def log_performance_stats(self, stats):
        """Log performance statistics"""
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "session_id": self.session_id,
            "log_type": "performance",
            "stats": stats
        }
        self._write_json_log(log_entry)
    
    def log_configuration(self, config_data):
        """Log system configuration"""
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "session_id": self.session_id,
            "log_type": "configuration",
            "config": config_data
        }
        self._write_json_log(log_entry)
    
    def log_error(self, error_type, error_message, context=None):
        """Log error events"""
        error_data = {
            "error_type": error_type,
            "error_message": str(error_message),
            "context": context or {}
        }
        self.log_system_event("error", error_data)
    
    def log_state_change(self, old_state, new_state, trigger_reason=None):
        """Log detection state changes"""
        self.log_system_event("state_change", {
            "old_state": old_state,
            "new_state": new_state,
            "trigger_reason": trigger_reason
        })
    
    def close(self):
        """Close logger and log session end"""
        self.log_system_event("session_end", {
            "session_id": self.session_id
        })


class StatisticsLogger:
    """CSV statistics logger for detection results (production reporting)"""
    
    def __init__(self, log_dir, session_id):
        self.log_dir = log_dir
        self.session_id = session_id
        self.csv_file_path = None
        self.file_handle = None
        
        # Statistics counters
        self.total_inspections = 0
        self.total_ok = 0
        self.total_ng = 0
        
        # Setup CSV file
        self._setup_csv()
    
    def _setup_csv(self):
        """Setup CSV file with headers"""
        try:
            # Create logs directory if it doesn't exist
            if self.log_dir and not os.path.exists(self.log_dir):
                os.makedirs(self.log_dir)
            
            # Generate filename with session_id
            self.csv_file_path = os.path.join(
                self.log_dir, 
                f"statistics_{self.session_id}.csv"
            )
            
            # Create file and write header
            self.file_handle = open(self.csv_file_path, 'w', encoding='utf-8', newline='')
            self.file_handle.write("timestamp,result,ok_count,ng_count,total,cycle_time_ms\n")
            self.file_handle.flush()
            
            print(f"Statistics log created: {self.csv_file_path}")
            
        except Exception as e:
            print(f"Failed to create statistics log: {e}")
            self.file_handle = None
    
    def log_detection(self, result, ok_count, ng_count, total_count, cycle_time_ms):
        """Log a single detection result to CSV"""
        if self.file_handle is None:
            return
        
        try:
            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            result_str = "NG" if result else "OK"
            
            # Write CSV line
            line = f"{timestamp},{result_str},{ok_count},{ng_count},{total_count},{cycle_time_ms}\n"
            self.file_handle.write(line)
            self.file_handle.flush()
            
            # Update counters
            self.total_inspections += 1
            self.total_ok += ok_count
            self.total_ng += ng_count
            
        except Exception as e:
            print(f"Failed to write statistics: {e}")
    
    def get_summary(self):
        """Get current statistics summary"""
        yield_rate = 0.0
        if self.total_ok + self.total_ng > 0:
            yield_rate = (self.total_ok / (self.total_ok + self.total_ng)) * 100
        
        return {
            "total_inspections": self.total_inspections,
            "total_ok": self.total_ok,
            "total_ng": self.total_ng,
            "yield_rate_percent": round(yield_rate, 2)
        }
    
    def close(self):
        """Close CSV file and print summary"""
        if self.file_handle:
            try:
                # Write summary as comment at end of file
                summary = self.get_summary()
                self.file_handle.write(f"\n# Summary: Inspections={summary['total_inspections']}, ")
                self.file_handle.write(f"OK={summary['total_ok']}, NG={summary['total_ng']}, ")
                self.file_handle.write(f"Yield={summary['yield_rate_percent']}%\n")
                
                self.file_handle.close()
                print(f"Statistics log closed: {self.csv_file_path}")
                print(f"  Total inspections: {summary['total_inspections']}")
                print(f"  Yield rate: {summary['yield_rate_percent']}%")
                
            except Exception as e:
                print(f"Error closing statistics log: {e}")


class RejudgeLogger:
    """CSV logger for manual review / rejudge records (V6.0)"""
    
    # CSV Header columns
    CSV_HEADER = [
        'timestamp', 'session_id', 'video_source', 'frame_number',
        'carrier_id', 'sealant_index', 'bbox_x', 'bbox_y', 'bbox_w', 'bbox_h',
        'ai_result', 'confidence', 'user_reviewed', 'user_agrees', 'final_result'
    ]
    
    def __init__(self, log_dir, session_id):
        self.log_dir = os.path.join(log_dir, 'rejudge')
        self.session_id = session_id
        
        # Create rejudge log directory if it doesn't exist
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)
            print(f"[RejudgeLogger] Created directory: {self.log_dir}")
    
    def _get_csv_path(self):
        """Get today's CSV file path"""
        date_str = datetime.now().strftime('%Y%m%d')
        return os.path.join(self.log_dir, f"rejudge_{date_str}.csv")
    
    def _ensure_csv_header(self, csv_path):
        """Ensure CSV file has header (create if needed)"""
        if not os.path.exists(csv_path):
            try:
                with open(csv_path, 'w', newline='', encoding='utf-8') as f:
                    writer = csv.writer(f)
                    writer.writerow(self.CSV_HEADER)
                print(f"[RejudgeLogger] Created CSV: {csv_path}")
            except Exception as e:
                print(f"[RejudgeLogger] Error creating CSV: {e}")
    
    def _calculate_final_result(self, ai_result, user_agrees):
        """
        Calculate final_result based on ai_result and user_agrees
        
        Rules:
        - ai_result=Pass, user_agrees=True  -> Pass
        - ai_result=Pass, user_agrees=False -> REFAIL
        - ai_result=NG,   user_agrees=True  -> NG
        - ai_result=NG,   user_agrees=False -> REPASS
        """
        if user_agrees:
            return ai_result
        else:
            if ai_result == 'Pass':
                return 'REFAIL'
            else:
                return 'REPASS'
    
    def write_records(self, pending_review, video_source):
        """
        Write all sealant records for a carrier to CSV
        
        Args:
            pending_review: dict containing carrier data and sealants list
            video_source: str, camera/video source name
        """
        if not pending_review or 'sealants' not in pending_review:
            return
        
        csv_path = self._get_csv_path()
        self._ensure_csv_header(csv_path)
        
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        carrier_id = pending_review.get('carrier_id', 0)
        frame_number = pending_review.get('frame_number', 0)
        user_reviewed = pending_review.get('user_reviewed', False)
        
        try:
            with open(csv_path, 'a', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                
                for sealant in pending_review['sealants']:
                    ai_result = sealant.get('ai_result', 'Pass')
                    user_agrees = sealant.get('user_agrees', True)
                    final_result = self._calculate_final_result(ai_result, user_agrees)
                    
                    bbox = sealant.get('bbox', [0, 0, 0, 0])
                    
                    row = [
                        timestamp,
                        self.session_id,
                        video_source,
                        frame_number,
                        carrier_id,
                        sealant.get('index', 0),
                        bbox[0],  # bbox_x
                        bbox[1],  # bbox_y
                        bbox[2],  # bbox_w
                        bbox[3],  # bbox_h
                        ai_result,
                        round(sealant.get('confidence', 0.0), 4),
                        user_reviewed,
                        user_agrees,
                        final_result
                    ]
                    writer.writerow(row)
            
            # Log summary
            sealant_count = len(pending_review['sealants'])
            disagree_count = sum(1 for s in pending_review['sealants'] if not s.get('user_agrees', True))
            print(f"[RejudgeLogger] Written {sealant_count} records for Carrier #{carrier_id}, "
                  f"reviewed={user_reviewed}, disagreed={disagree_count}")
                  
        except Exception as e:
            print(f"[RejudgeLogger] Error writing records: {e}")


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


class VideoRecorder:
    """Video recorder for saving raw or inference frames"""
    
    def __init__(self, recording_config, frame_shape, session_id, logger=None):
        self.recording_config = recording_config
        self.frame_shape = frame_shape
        self.session_id = session_id
        self.logger = logger
        self.video_writer = None
        self.is_recording = False
        self.output_filename = None
        
        # Store calculated recording dimensions
        self.recording_width = None
        self.recording_height = None
        
        if recording_config['enable_recording']:
            self._setup_recording()
    
    def _calculate_recording_dimensions(self):
        """Calculate and store recording dimensions based on quality setting"""
        height, width = self.frame_shape[:2]
        
        if self.recording_config['quality'] == 'low':
            width = int(width * 0.5)
            height = int(height * 0.5)
        elif self.recording_config['quality'] == 'medium':
            width = int(width * 0.7)
            height = int(height * 0.7)
        
        # Ensure even dimensions for codec compatibility
        width = width - (width % 2)
        height = height - (height % 2)
        
        self.recording_width = width
        self.recording_height = height
        
        return width, height
    
    def _setup_recording(self):
        """Setup video recording"""
        try:
            # Create recordings directory
            output_dir = self.recording_config['output_path']
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            
            # Generate filename with timestamp (only if not already set)
            if self.output_filename is None:
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                record_type = self.recording_config['record_type']
                video_format = self.recording_config['video_format']
                self.output_filename = f"{output_dir}{self.session_id}_{record_type}_{timestamp}.{video_format}"
            
            video_format = self.recording_config['video_format']
            
            # Set codec based on format - optimized for macOS
            if video_format.lower() == 'mp4':
                # Try macOS-friendly codecs in order
                codecs_to_try = ['avc1', 'mp4v', 'h264']
                fourcc = None
                
                for codec in codecs_to_try:
                    test_fourcc = cv2.VideoWriter_fourcc(*codec)
                    test_writer = cv2.VideoWriter('test.tmp', test_fourcc, 1, (100, 100))
                    if test_writer.isOpened():
                        test_writer.release()
                        if os.path.exists('test.tmp'):
                            os.remove('test.tmp')
                        fourcc = test_fourcc
                        print(f"Using codec: {codec}")
                        break
                    test_writer.release()
                
                if fourcc is None:
                    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # fallback
                    
            elif video_format.lower() == 'avi':
                fourcc = cv2.VideoWriter_fourcc(*'MJPG')  # Better compatibility than XVID
            else:
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            
            # Calculate recording dimensions
            width, height = self._calculate_recording_dimensions()
            
            # Initialize video writer
            self.video_writer = cv2.VideoWriter(
                self.output_filename,
                fourcc,
                self.recording_config['fps'],
                (width, height)
            )
            
            if self.video_writer.isOpened():
                self.is_recording = True
                record_type = self.recording_config['record_type']
                print(f"Recording started: {self.output_filename}")
                print(f"Record type: {record_type}")
                print(f"Quality: {self.recording_config['quality']} ({width}x{height})")
                
                if self.logger:
                    self.logger.log_system_event("recording_started", {
                        "filename": self.output_filename,
                        "record_type": record_type,
                        "quality": self.recording_config['quality'],
                        "resolution": f"{width}x{height}"
                    })
            else:
                error_msg = f"Failed to initialize video writer: {self.output_filename}"
                print(error_msg)
                if self.logger:
                    self.logger.log_error("video_writer_init_error", error_msg)
                    
        except Exception as e:
            error_msg = f"Recording setup error: {e}"
            print(error_msg)
            if self.logger:
                self.logger.log_error("recording_setup_error", e)
    
    def record_frame(self, frame):
        """Record a frame to video file"""
        if not self.is_recording or self.video_writer is None:
            return
        
        try:
            # Use pre-calculated recording dimensions for consistent sizing
            if self.recording_width is None or self.recording_height is None:
                self._calculate_recording_dimensions()
            
            # Resize frame to match video writer dimensions
            frame_resized = cv2.resize(
                frame, 
                (self.recording_width, self.recording_height), 
                interpolation=cv2.INTER_AREA
            )
            
            self.video_writer.write(frame_resized)
            
        except Exception as e:
            error_msg = f"Frame recording error: {e}"
            print(error_msg)
            if self.logger:
                self.logger.log_error("frame_recording_error", e)
    
    def start_new_recording(self):
        """Start a new recording session (for toggle functionality)"""
        # Reset filename to generate new one
        self.output_filename = None
        self._setup_recording()
    
    def stop_recording(self):
        """Stop recording and release resources"""
        if self.is_recording and self.video_writer is not None:
            try:
                self.video_writer.release()
                self.is_recording = False
                
                # Get file size
                file_size = 0
                if os.path.exists(self.output_filename):
                    file_size = os.path.getsize(self.output_filename)
                
                print(f"Recording stopped: {self.output_filename}")
                print(f"File size: {file_size / (1024*1024):.2f} MB")
                
                if self.logger:
                    self.logger.log_system_event("recording_stopped", {
                        "filename": self.output_filename,
                        "file_size_mb": round(file_size / (1024*1024), 2)
                    })
                    
            except Exception as e:
                error_msg = f"Recording stop error: {e}"
                print(error_msg)
                if self.logger:
                    self.logger.log_error("recording_stop_error", e)


class SafeVideoProcessor:
    """Enhanced video processor with detection state machine"""
    
    def __init__(self):
        self.running = False
        self.stop_event = threading.Event()
        self.cap = None
        self.model = None
        self.logger = None
        self.video_recorder = None
        self.statistics_logger = None  # NEW: CSV statistics logger
        
        # Performance tracking
        self.frame_count = 0
        self.detection_count = 0
        self.start_time = time.time()
        
        # Configuration
        self.config_manager = None
        self.paths = {}
        self.yolo_config = {}
        self.detection_config = {}
        self.display_config = {}
        self.device = 'cpu'
        self.class_map = {}
        self.recording_config = {}
        
        # Video source info
        self.source_name = "Unknown"
        self.frame_shape = None
        
        # Detection results storage
        self.current_sealant_result = None
        self.sealant_detection_results = {}
        
        # Worker threads
        self.capture_thread = None
        self.yolo_thread = None
        
        # === NEW: Detection State Machine ===
        self.detection_state = 'WAITING'  # WAITING, DETECTED, RESULT_HOLD
        self.detection_result = None      # None, 'Pass', 'NG'
        self.last_light_status = None     # Track last light status to avoid duplicate calls
        self.detection_triggered_time = 0 # Track when detection was last triggered
        
        # === NEW: Tolerance mechanism for detection box flickering ===
        self.absent_frame_count = 0       # Count consecutive frames with missing objects
        self.result_hold_start_time = 0   # Timestamp when entering RESULT_HOLD state
        
        # === NEW: Detection cycle time tracking ===
        self.detection_start_time = 0     # Timestamp when detection cycle starts
        
        # === Display mode control ===
        self.fit_screen_mode = True       # True = fit screen, False = original size
        self.is_fullscreen = False        # Track fullscreen state
        self.cached_screen_size = None    # Cache screen size to avoid repeated tkinter calls
        
        # === Screenshot settings ===
        self.screenshot_path = './screenshots/'
        
        # === 24/7 Monitoring Mode - Enhanced tolerance settings ===
        self.last_heartbeat_time = time.time()
        self.heartbeat_interval = 60  # Log heartbeat every 60 seconds
        self.last_frame_received_time = time.time()
        self.idle_warning_threshold = 300  # Warn after 5 minutes of no frames
        self.idle_warning_logged = False
        
        # Camera reconnection settings
        self.camera_reconnect_attempts = 0
        self.max_camera_reconnect_attempts = 5
        self.camera_reconnect_delay = 5  # seconds between reconnect attempts
        
        # === V6.0: Manual Review / Rejudge settings ===
        self.rejudge_logger = None           # RejudgeLogger instance
        self.manual_review_window = None     # ManualReviewWindow instance
        self.pending_review = None           # Pending review data for current carrier
        self.carrier_count = 0               # Carrier sequence number
        self.user_reviewed_flag = False      # Whether user pressed M key in current RESULT_HOLD
        self.current_raw_frame = None        # Store raw frame for review window
        self.current_sealant_boxes = []      # Store sealant boxes for review window
        self.current_scores = {}             # Store confidence scores for sealants
        self.rejudge_config = {}             # Rejudge configuration
        self.review_log_written = False      # Whether rejudge log has been written for current carrier
        self.fullscreen_paused_for_review = False  # Whether fullscreen was exited to show review window
        
    def initialize(self):
        """Initialize all components with enhanced error handling"""
        try:
            print("Initializing video processor...")
            
            # Load configuration
            self.config_manager = ConfigManager()
            self.paths = self.config_manager.get_paths()
            self.yolo_config = self.config_manager.get_yolo_config()
            self.detection_config = self.config_manager.get_detection_config()
            self.display_config = self.config_manager.get_display_config()
            self.device = self.config_manager.get_device()
            self.class_map = self.config_manager.get_classes()
            self.recording_config = self.config_manager.get_recording_config()
            
            # Initialize logger
            logging_config = self.config_manager.get_logging_config()
            if logging_config['save_detection_log']:
                self.logger = JSONLogger(
                    logging_config['log_file_path'],
                    logging_config['log_level']
                )
                
                # Log configuration
                config_data = {
                    "paths": self.paths,
                    "yolo_config": self.yolo_config,
                    "detection_config": self.detection_config,
                    "display_config": self.display_config,
                    "device": self.device,
                    "classes": self.class_map,
                    "recording_config": self.recording_config
                }
                self.logger.log_configuration(config_data)
                
                # Initialize statistics logger (CSV)
                log_dir = os.path.dirname(logging_config['log_file_path'])
                self.statistics_logger = StatisticsLogger(log_dir, self.logger.session_id)
                
                # V6.0: Initialize rejudge logger
                self.rejudge_logger = RejudgeLogger(log_dir, self.logger.session_id)
            
            # V6.0: Load rejudge configuration
            self.rejudge_config = self.config_manager.get_rejudge_config()
            
            # Load YOLO model
            print(f"Loading YOLO model: {self.paths['model_path']}")
            if not os.path.exists(self.paths['model_path']):
                error_msg = f"Model file not found: {self.paths['model_path']}"
                print(error_msg)
                if self.logger:
                    self.logger.log_error("model_file_not_found", error_msg)
                return False
            
            self.model = YOLO(self.paths['model_path'])
            self.model.to(self.device)
            print(f"Model loaded successfully on device: {self.device}")
            
            # Initialize video source
            self.cap, self.source_name = self._setup_video_source()
            if self.cap is None:
                return False
            
            # Get frame shape for recording
            ret, test_frame = self.cap.read()
            if ret:
                self.frame_shape = test_frame.shape
                # Put frame back for processing
                try:
                    frame_queue.put(test_frame, timeout=0.1)
                except queue.Full:
                    pass
            
            # Initialize video recorder if enabled
            if self.recording_config['enable_recording'] and self.logger:
                self.video_recorder = VideoRecorder(
                    self.recording_config,
                    self.frame_shape,
                    self.logger.session_id,
                    self.logger
                )
            
            print("Video processor initialized successfully")
            if self.logger:
                self.logger.log_system_event("initialization_complete")
            
            return True
            
        except Exception as e:
            error_msg = f"Initialization failed: {e}"
            print(error_msg)
            if self.logger:
                self.logger.log_error("initialization_error", e)
            import traceback
            traceback.print_exc()
            return False
    
    def _setup_video_source(self):
        """Setup video source (camera, dfk_camera, or file) with error handling"""
        try:
            video_config = self.config_manager.get_video_source_config()
            
            if video_config['mode'] == 'camera':
                # 一般 USB/UVC 相機
                print(f"Initializing camera {video_config['camera_index']}")
                cap = cv2.VideoCapture(video_config['camera_index'], cv2.CAP_DSHOW)
                
                if not cap.isOpened():
                    error_msg = f"Failed to open camera {video_config['camera_index']}"
                    print(error_msg)
                    if self.logger:
                        self.logger.log_error("camera_open_error", error_msg)
                    return None, None
                
                # Set camera properties
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, video_config['camera_width'])
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, video_config['camera_height'])
                cap.set(cv2.CAP_PROP_AUTOFOCUS, video_config['camera_focus'])
                cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)

                # 設定手動焦距值 (範圍通常是 0-255)
                cap.set(cv2.CAP_PROP_FOCUS, 0)
                source_name = f"Camera_{video_config['camera_index']}"
                print(f"Camera initialized: {video_config['camera_width']}x{video_config['camera_height']}")
            
            elif video_config['mode'] == 'dfk_camera':
                # TIS DFK 工業相機
                if not TIS_AVAILABLE:
                    error_msg = "TIS camera support not available. Please ensure tis_camera_wrapper.py and TIS_API.py are in the project directory."
                    print(error_msg)
                    if self.logger:
                        self.logger.log_error("tis_not_available", error_msg)
                    return None, None
                
                print(f"Initializing TIS DFK camera...")
                
                # 決定使用索引還是名稱搜尋
                camera_index = None
                tis_camera_idx = video_config.get('tis_camera_index', -1)
                if tis_camera_idx >= 0:
                    camera_index = tis_camera_idx
                
                # 建立 TIS 相機包裝器
                cap = TISCameraWrapper(
                    dll_path=video_config.get('tis_dll_path', './tisgrabber_x64.dll'),
                    device_name=video_config.get('tis_device_name', 'DFK 33UX178'),
                    camera_index=camera_index,
                    logger=self.logger
                )
                
                if not cap.isOpened():
                    error_msg = "Failed to open TIS DFK camera"
                    print(error_msg)
                    if self.logger:
                        self.logger.log_error("dfk_camera_open_error", error_msg)
                    return None, None
                
                # 取得設備資訊
                device_info = cap.get_device_info()
                source_name = f"TIS_{device_info['device_name']}"
                print(f"TIS DFK camera initialized: {device_info['resolution']}")
                
                if self.logger:
                    self.logger.log_system_event("dfk_camera_initialized", device_info)
                
            elif video_config['mode'] == 'video':
                video_files = glob(self.paths['video_path'])
                if not video_files:
                    error_msg = f"No video files found matching: {self.paths['video_path']}"
                    print(error_msg)
                    if self.logger:
                        self.logger.log_error("video_file_not_found", error_msg)
                    return None, None
                
                video_file = video_files[video_config['video_index']]
                print(f"Opening video file: {video_file}")
                cap = cv2.VideoCapture(video_file)
                
                if not cap.isOpened():
                    error_msg = f"Failed to open video file: {video_file}"
                    print(error_msg)
                    if self.logger:
                        self.logger.log_error("video_open_error", error_msg)
                    return None, None
                
                source_name = os.path.basename(video_file)
                print(f"Video file opened: {video_file}")
            
            else:
                error_msg = f"Invalid video source mode: {video_config['mode']}. Supported modes: 'camera', 'dfk_camera', 'video'"
                print(error_msg)
                if self.logger:
                    self.logger.log_error("config_error", error_msg)
                raise ValueError(error_msg)
            
            return cap, source_name
        
        except Exception as e:
            if self.logger:
                self.logger.log_error("video_source_init_error", e)
            raise
    
    def start_threads(self):
        """Start worker threads"""
        try:
            self.capture_thread = threading.Thread(
                target=self._capture_frames, 
                name="CaptureThread"
            )
            self.yolo_thread = threading.Thread(
                target=self._yolo_worker, 
                name="YOLOThread"
            )
            
            self.capture_thread.daemon = True
            self.yolo_thread.daemon = True
            
            self.capture_thread.start()
            self.yolo_thread.start()
            
            print("Worker threads started")
            if self.logger:
                self.logger.log_system_event("threads_started")
                
        except Exception as e:
            error_msg = f"Failed to start threads: {e}"
            print(error_msg)
            if self.logger:
                self.logger.log_error("thread_start_error", e)
            raise
    
    def _capture_frames(self):
        """Capture frames from video source with enhanced fault tolerance for 24/7 operation"""
        consecutive_failures = 0
        max_failures = 50  # Increased from 10 to 50 for better tolerance (~5 seconds)
        
        try:
            while not self.stop_event.is_set():
                # Check if camera is still open
                if not self.cap or not self.cap.isOpened():
                    # Attempt reconnection for camera modes
                    video_config = self.config_manager.get_video_source_config()
                    if video_config['mode'] in ['camera', 'dfk_camera']:
                        if self._attempt_camera_reconnect():
                            consecutive_failures = 0
                            continue
                        else:
                            # Reconnection failed, wait and retry
                            time.sleep(1.0)
                            continue
                    else:
                        # Video file mode - no reconnection
                        break
                
                try:
                    ret, frame = self.cap.read()
                    if not ret:
                        consecutive_failures += 1
                        
                        # Log warning periodically instead of immediately stopping
                        if consecutive_failures == 10:
                            print(f"[WARNING] Camera read failures: {consecutive_failures}")
                            if self.logger:
                                self.logger.log_system_event("capture_warning", {
                                    "consecutive_failures": consecutive_failures,
                                    "message": "Multiple frame read failures detected"
                                })
                        
                        if consecutive_failures >= max_failures:
                            error_msg = f"Camera read failures reached {max_failures}, attempting reconnection..."
                            print(f"[WARNING] {error_msg}")
                            if self.logger:
                                self.logger.log_error("capture_max_failures", error_msg)
                            
                            # Try to reconnect instead of stopping
                            video_config = self.config_manager.get_video_source_config()
                            if video_config['mode'] in ['camera', 'dfk_camera']:
                                if self._attempt_camera_reconnect():
                                    consecutive_failures = 0
                                    print("[INFO] Camera reconnected successfully")
                                    continue
                            
                            # If not camera mode or reconnection failed, break
                            break
                        
                        time.sleep(0.1)
                        continue
                    
                    # Successful frame read
                    consecutive_failures = 0
                    self.last_frame_received_time = time.time()
                    self.idle_warning_logged = False
                    
                    try:
                        frame_queue.put(frame, timeout=0.1)
                    except queue.Full:
                        pass
                        
                except Exception as e:
                    error_msg = f"Capture thread error: {e}"
                    print(f"[ERROR] {error_msg}")
                    if self.logger:
                        self.logger.log_error("capture_thread_error", str(e))
                    
                    # Don't break immediately, try to recover
                    consecutive_failures += 1
                    if consecutive_failures >= max_failures:
                        break
                    time.sleep(0.5)
        
        except Exception as e:
            if self.logger:
                self.logger.log_error("capture_thread_fatal_error", str(e))
        
        print("[INFO] Capture thread stopped")
    
    def _attempt_camera_reconnect(self):
        """Attempt to reconnect to camera"""
        self.camera_reconnect_attempts += 1
        
        if self.camera_reconnect_attempts > self.max_camera_reconnect_attempts:
            error_msg = f"Max camera reconnection attempts ({self.max_camera_reconnect_attempts}) reached"
            print(f"[ERROR] {error_msg}")
            if self.logger:
                self.logger.log_error("camera_reconnect_max_attempts", error_msg)
            return False
        
        print(f"[INFO] Attempting camera reconnection (attempt {self.camera_reconnect_attempts}/{self.max_camera_reconnect_attempts})...")
        
        try:
            # Release old capture if exists
            if self.cap:
                try:
                    self.cap.release()
                except:
                    pass
            
            time.sleep(self.camera_reconnect_delay)
            
            # Reinitialize camera
            self.cap, self.source_name = self._setup_video_source()
            
            if self.cap and self.cap.isOpened():
                print(f"[INFO] Camera reconnected successfully: {self.source_name}")
                if self.logger:
                    self.logger.log_system_event("camera_reconnected", {
                        "attempt": self.camera_reconnect_attempts,
                        "source": self.source_name
                    })
                self.camera_reconnect_attempts = 0  # Reset counter on success
                return True
            else:
                print(f"[WARNING] Camera reconnection attempt {self.camera_reconnect_attempts} failed")
                return False
                
        except Exception as e:
            print(f"[ERROR] Camera reconnection error: {e}")
            if self.logger:
                self.logger.log_error("camera_reconnect_error", str(e))
            return False
    
    def _yolo_worker(self):
        """YOLO inference worker thread with enhanced fault tolerance for 24/7 operation"""
        consecutive_failures = 0
        max_failures = 30  # Increased from 5 to 30 for better tolerance
        cuda_error_count = 0
        max_cuda_errors = 3
        
        try:
            while not self.stop_event.is_set():
                try:
                    frame = frame_queue.get(timeout=1.0)
                    if frame is None:
                        break
                    
                    boxes, scores, categories = self._yolo_detect(frame)
                    consecutive_failures = 0
                    
                    try:
                        result_queue.put((frame, boxes, scores, categories), timeout=0.1)
                    except queue.Full:
                        pass
                        
                except queue.Empty:
                    continue
                except RuntimeError as e:
                    # Handle CUDA errors specifically
                    error_str = str(e).lower()
                    if 'cuda' in error_str or 'out of memory' in error_str:
                        cuda_error_count += 1
                        print(f"[WARNING] CUDA error detected ({cuda_error_count}/{max_cuda_errors}): {e}")
                        
                        if self.logger:
                            self.logger.log_error("cuda_error", str(e), {
                                "cuda_error_count": cuda_error_count
                            })
                        
                        # Try to recover from CUDA OOM
                        if 'out of memory' in error_str:
                            self._try_cuda_memory_cleanup()
                        
                        if cuda_error_count >= max_cuda_errors:
                            print("[ERROR] Too many CUDA errors, attempting model reload...")
                            if self._try_reload_model():
                                cuda_error_count = 0
                                print("[INFO] Model reloaded successfully")
                            else:
                                print("[ERROR] Model reload failed")
                                break
                        
                        time.sleep(1.0)
                        continue
                    else:
                        # Other RuntimeError
                        consecutive_failures += 1
                        raise
                        
                except Exception as e:
                    consecutive_failures += 1
                    error_msg = f"YOLO worker error: {e}"
                    
                    # Log warning at intervals instead of every error
                    if consecutive_failures in [1, 5, 10, 20, 30]:
                        print(f"[WARNING] {error_msg} (failure {consecutive_failures}/{max_failures})")
                        if self.logger:
                            self.logger.log_error("yolo_worker_error", str(e), {
                                "consecutive_failures": consecutive_failures
                            })
                    
                    if consecutive_failures >= max_failures:
                        error_msg = f"YOLO worker: {max_failures} consecutive failures, stopping"
                        print(f"[ERROR] {error_msg}")
                        if self.logger:
                            self.logger.log_error("yolo_worker_fatal_error", error_msg)
                        break
                    
                    time.sleep(0.5)
        
        except Exception as e:
            if self.logger:
                self.logger.log_error("yolo_thread_fatal_error", str(e))
        
        print("[INFO] YOLO thread stopped")
    
    def _try_cuda_memory_cleanup(self):
        """Attempt to clean up CUDA memory"""
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                print("[INFO] CUDA memory cache cleared")
        except Exception as e:
            print(f"[WARNING] CUDA memory cleanup failed: {e}")
    
    def _try_reload_model(self):
        """Attempt to reload the YOLO model"""
        try:
            print("[INFO] Attempting to reload YOLO model...")
            
            # Clear CUDA cache first
            self._try_cuda_memory_cleanup()
            
            # Reload model
            self.model = YOLO(self.paths['model_path'])
            self.model.to(self.device)
            
            if self.logger:
                self.logger.log_system_event("model_reloaded", {
                    "model_path": self.paths['model_path'],
                    "device": self.device
                })
            
            return True
        except Exception as e:
            print(f"[ERROR] Model reload failed: {e}")
            if self.logger:
                self.logger.log_error("model_reload_failed", str(e))
            return False
    
    def _yolo_detect(self, frame):
        """YOLO detection with error handling"""
        try:
            results = self.model.predict(
                frame,
                conf=self.yolo_config['confidence_threshold'],
                iou=self.yolo_config['iou_threshold'],
                agnostic_nms=False,
                max_det=self.yolo_config['max_detections'],
                augment=self.yolo_config['use_augmentation'],
                verbose=False,
                half=self.yolo_config['use_half_precision']
            )
            
            if len(results) > 0 and results[0].boxes is not None:
                result = results[0]
                boxes = result.boxes.xyxy.cpu().numpy().astype(np.int32)
                scores = result.boxes.conf.cpu().numpy()
                categories = result.boxes.cls.cpu().numpy().astype(np.int32)
            else:
                boxes = np.array([]).reshape(0, 4)
                scores = np.array([])
                categories = np.array([])
                
        except Exception as e:
            error_msg = f"YOLO detection error: {e}"
            print(error_msg)
            if self.logger:
                self.logger.log_error("yolo_detection_error", e)
            boxes = np.array([]).reshape(0, 4)
            scores = np.array([])
            categories = np.array([])
        
        return boxes, scores, categories
    
    def run(self):
        """Main processing loop with detection state machine"""
        try:
            if not self.initialize():
                return False
            
            self.start_threads()
            self.running = True
            
            # Initialize processing variables
            stop_time = 0
            previous_centroid = [0, 0]
            
            # === NEW: Initialize detection state machine ===
            self.detection_state = 'WAITING'
            self.detection_result = None
            current_status = self._call_light_once('Standby')
            
            # Reset sealant detection state
            self.current_sealant_result = None
            self.sealant_detection_results = {}
            
            # === NEW: Initialize tolerance mechanism ===
            self.absent_frame_count = 0
            self.result_hold_start_time = 0
            
            consecutive_empty_results = 0
            max_empty_results = 300000
            
            # === 24/7 Monitoring Mode: Removed hard timeout limit ===
            # Instead of stopping, we log warnings and continue running
            frame_timeout_count = 0
            
            print("Starting main processing loop with state machine...")
            print(f"Initial state: {self.detection_state}")
            print("*** 24/7 Monitoring Mode Enabled ***")
            print("\n=== Keyboard Shortcuts ===")
            print("  Q: Quit | S: Screenshot | F: Fit screen | G: Fullscreen")
            print("  P: Statistics | R: Reset stats | V: Toggle recording")
            print("===========================\n")
            
            # Create display window with proper flags for fullscreen support
            self._create_display_window(self.display_config['window_name'])
            
            if self.logger:
                self.logger.log_system_event("main_loop_started", {
                    "mode": "24/7_monitoring"
                })
                self.logger.log_state_change(None, self.detection_state, "initialization")
            
            while self.running and not self.stop_event.is_set():
                # === Heartbeat logging for system health monitoring ===
                self._log_heartbeat()
                
                try:
                    frame, boxes, scores, categories = result_queue.get(timeout=2.0)
                    frame_timeout_count = 0
                    self.last_frame_received_time = time.time()
                    self.idle_warning_logged = False
                    
                except queue.Empty:
                    frame_timeout_count += 1
                    
                    # === 24/7 Mode: Log warning instead of stopping ===
                    idle_duration = time.time() - self.last_frame_received_time
                    
                    # Log warning at intervals (every 30 seconds of idle)
                    if frame_timeout_count % 15 == 0:  # Every 30 seconds (15 * 2.0s timeout)
                        print(f"[WARNING] No frames received for {idle_duration:.0f} seconds")
                        if self.logger:
                            self.logger.log_system_event("frame_idle_warning", {
                                "idle_duration_seconds": round(idle_duration, 1),
                                "timeout_count": frame_timeout_count
                            })
                    
                    # Extended idle warning (configurable threshold)
                    if idle_duration >= self.idle_warning_threshold and not self.idle_warning_logged:
                        print(f"[WARNING] Extended idle period: {idle_duration:.0f} seconds without frames")
                        if self.logger:
                            self.logger.log_system_event("extended_idle_warning", {
                                "idle_duration_seconds": round(idle_duration, 1),
                                "threshold": self.idle_warning_threshold
                            })
                        self.idle_warning_logged = True
                    
                    # Continue running - DO NOT BREAK for 24/7 operation
                    continue
                
                if frame is None:
                    # Only break if explicitly signaled to stop
                    print("[INFO] Received stop signal (None frame)")
                    break
                
                self.frame_count += 1
                
                # Check for empty results
                if len(categories) == 0:
                    consecutive_empty_results += 1
                else:
                    consecutive_empty_results = 0
                    self.detection_count += 1
                
                # === Fixed: Check video end only for 'video' mode, not by source_name ===
                video_config = self.config_manager.get_video_source_config()
                # if (video_config['mode'] == 'video' and 
                #     consecutive_empty_results >= max_empty_results):
                #     print("[INFO] Video appears to have ended or no detections for too long.")
                #     if self.logger:
                #         self.logger.log_system_event("video_ended")
                #     break
                
                # Process frame
                raw_frame = frame.copy()
                sealant_boxes = []
                
                self._add_ui_elements(frame, current_status)
                
                # Process detections
                if len(categories) > 0 and self.class_map['carrier'] in categories:
                    correct_carrier_box = self._check_correct_carrier(categories, boxes)
                    if correct_carrier_box is not None:
                        sealant_boxes = self._draw_detections(frame, boxes, categories, correct_carrier_box)
                
                # Record video based on configuration
                if self.video_recorder and self.video_recorder.is_recording:
                    if self.recording_config['record_type'] == 'raw':
                        self.video_recorder.record_frame(raw_frame)
                    else:  # 'inference'
                        self.video_recorder.record_frame(frame)
                
                # === NEW: Enhanced detection logic with state machine ===
                required_classes = set(self.class_map.values())
                detected_classes = set(categories)
                has_required_objects = required_classes.issubset(detected_classes)
                
                if has_required_objects:
                    # Objects are present - handle based on current state
                    # V6.0: Pass scores for confidence tracking
                    current_status, stop_time, previous_centroid = self._handle_objects_present(
                        frame, raw_frame, boxes, categories, sealant_boxes, scores,
                        current_status, stop_time, previous_centroid
                    )
                else:
                    # Objects are absent - handle based on current state
                    current_status, stop_time, previous_centroid = self._handle_objects_absent(
                        current_status, stop_time, previous_centroid
                    )
                
                # V6.0: Update manual review window if open
                if self.manual_review_window and self.manual_review_window.is_open:
                    review_result = self.manual_review_window.update()
                    if review_result in ['confirm', 'cancel']:
                        self._handle_review_complete(review_result)
                
                self._display_frame(frame, self.display_config['window_name'])
                
                # Check for quit command
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    print("Quit command received")
                    if self.logger:
                        self.logger.log_system_event("quit_command_received")
                    break
                elif key == ord('s') or key == ord('S'):
                    # Screenshot function
                    self._take_screenshot(frame)
                elif key == ord('p') or key == ord('P'):
                    # Print statistics (moved from 's')
                    self._print_statistics()
                elif key == ord('r'):
                    self._reset_statistics()
                elif key == ord('f') or key == ord('F'):
                    # Toggle fit screen mode
                    self._toggle_fit_screen()
                elif key == ord('g') or key == ord('G'):
                    # Toggle fullscreen mode
                    self._toggle_fullscreen(self.display_config['window_name'])
                elif key == ord('v') or key == ord('V'):
                    # Toggle video recording (can start even if not enabled in config)
                    self._toggle_recording()
                elif key == ord('m') or key == ord('M'):
                    # V6.0: Open manual review window (only in RESULT_HOLD state)
                    if self.detection_state == 'RESULT_HOLD':
                        self._open_manual_review()
                
                # Performance monitoring
                if self.frame_count % 100 == 0:
                    self._print_performance()
                    
        except Exception as e:
            error_msg = f"Main loop error: {e}"
            print(error_msg)
            if self.logger:
                self.logger.log_error("main_loop_error", e)
            import traceback
            traceback.print_exc()
        finally:
            self.stop()
        
        return True
    
    def _handle_objects_present(self, frame, raw_frame, boxes, categories, sealant_boxes, scores,
                               current_status, stop_time, previous_centroid):
        """Handle the case when carrier and sealant are detected"""
        
        # === NEW: Reset absent frame counter when objects are present ===
        self.absent_frame_count = 0
        
        if self.detection_state == 'RESULT_HOLD':
            # 🔑 KEY LOGIC: Already have detection result, maintain state
            # No further detection, no light changes, just return current status
            return current_status, stop_time, previous_centroid
        
        elif self.detection_state == 'WAITING':
            # Check if we should trigger detection
            # V6.0: Pass scores for confidence tracking
            return self._check_detection_trigger(
                frame, raw_frame, boxes, categories, sealant_boxes, scores,
                current_status, stop_time, previous_centroid
            )
        
        elif self.detection_state == 'DETECTED':
            # Detection already triggered, maintain status
            return current_status, stop_time, previous_centroid
        
        return current_status, stop_time, previous_centroid
    
    def _handle_objects_absent(self, current_status, stop_time, previous_centroid):
        """Handle the case when carrier or sealant are not detected"""
        
        # === NEW: Increment absent frame counter ===
        self.absent_frame_count += 1
        
        # Get tolerance settings from config
        absent_tolerance = self.detection_config.get('absent_tolerance_frames', 5)
        result_hold_min = self.detection_config.get('result_hold_min_seconds', 2.0)
        
        if self.detection_state == 'RESULT_HOLD':
            # === NEW: Check tolerance before resetting ===
            # Check 1: Are we still within the absent tolerance?
            if self.absent_frame_count < absent_tolerance:
                # Still within tolerance, don't reset
                return current_status, stop_time, previous_centroid
            
            # Check 2: Are we still within the protection period?
            time_in_result_hold = time.time() - self.result_hold_start_time
            if time_in_result_hold < result_hold_min:
                # Still within protection period, don't reset
                return current_status, stop_time, previous_centroid
            
            # V6.0: Write pending review log before leaving RESULT_HOLD
            self._write_pending_review_log()
            
            # V6.0: Close manual review window if open
            if self.manual_review_window and self.manual_review_window.is_open:
                self.manual_review_window.close()
            
            # Both checks passed, now reset
            old_state = self.detection_state
            self.detection_state = 'WAITING'
            self.detection_result = None
            
            print(f"Objects removed (absent {self.absent_frame_count} frames), state: {old_state} → {self.detection_state}")
            if self.logger:
                self.logger.log_state_change(old_state, self.detection_state, 
                    f"objects_removed_after_{self.absent_frame_count}_frames")
            
            # Reset sealant detection state
            self.current_sealant_result = None
            self.sealant_detection_results = {}
            
            # V6.0: Reset review state for next carrier
            self.user_reviewed_flag = False
            self.pending_review = None
            self.current_raw_frame = None
            self.current_sealant_boxes = []
            self.current_scores = {}
            self.review_log_written = False
            
            # End detection session
            if self.logger and self.logger.detection_session_active:
                self.logger.log_detection_session_end()
            
            # Reset to standby light (yellow)
            current_status = self._call_light_once('Standby')
            
            # Reset timing
            stop_time = 0
            previous_centroid = [0, 0]
        
        elif self.detection_state in ['DETECTED', 'WAITING']:
            # === NEW: Check tolerance before resetting ===
            if self.absent_frame_count < absent_tolerance:
                # Still within tolerance, maintain stop_time (don't reset)
                return current_status, stop_time, previous_centroid
            
            # Objects disappeared beyond tolerance
            if self.detection_state == 'DETECTED':
                old_state = self.detection_state
                self.detection_state = 'WAITING'
                print(f"Objects disappeared during detection (absent {self.absent_frame_count} frames), state: {old_state} → {self.detection_state}")
                if self.logger:
                    self.logger.log_state_change(old_state, self.detection_state, 
                        f"objects_disappeared_during_detection_after_{self.absent_frame_count}_frames")
            
            # Ensure light is on standby if not already
            if current_status != 'Standby':
                current_status = self._call_light_once('Standby')
            
            # Reset timing when truly absent
            stop_time = 0
            previous_centroid = [0, 0]
        
        return current_status, stop_time, previous_centroid
    
    def _check_detection_trigger(self, frame, raw_frame, boxes, categories, sealant_boxes, scores,
                               current_status, stop_time, previous_centroid):
        """Check if detection should be triggered (only called in WAITING state)"""
        
        h, w = frame.shape[:2]
        
        carrier_idx = list(categories).index(self.class_map['carrier'])
        carrier_pos = boxes[carrier_idx]
        centroid = [
            int((carrier_pos[2] - carrier_pos[0]) / 2) + carrier_pos[0],
            int((carrier_pos[3] - carrier_pos[1]) / 2) + carrier_pos[1]
        ]
        
        cv2.circle(frame, centroid, 5, (0, 255, 0), 2)
        
        distance = np.linalg.norm(np.array(centroid) - np.array(previous_centroid))
        
        if centroid[1] > h/2 and distance < self.detection_config['distance_threshold_pixels']:
            stop_time += 1
            
            if stop_time / self.detection_config['fps'] > self.detection_config['stop_threshold_seconds']:
                print(f'Detection triggered after {round(stop_time/self.detection_config["fps"], 2)} seconds')
                
                # === NEW: Record detection start time for cycle time calculation ===
                self.detection_start_time = time.time()
                
                # === STATE TRANSITION: WAITING → DETECTED ===
                old_state = self.detection_state
                self.detection_state = 'DETECTED'
                if self.logger:
                    self.logger.log_state_change(old_state, self.detection_state, "trigger_conditions_met")
                
                # Start detection session
                if self.logger and not self.logger.detection_session_active:
                    self.logger.log_detection_session_start()
                
                # 🔑 EXECUTE DETECTION ONLY ONCE
                sealant_result = self._cut_sealant_detection(raw_frame, sealant_boxes)
                self.current_sealant_result = sealant_result
                
                # Log detection results
                self._log_detection_results(sealant_result, frame.shape, centroid)
                
                # === STATE TRANSITION: DETECTED → RESULT_HOLD ===
                old_state = self.detection_state
                self.detection_state = 'RESULT_HOLD'
                
                # === NEW: Record time entering RESULT_HOLD for protection period ===
                self.result_hold_start_time = time.time()
                
                if self.logger:
                    self.logger.log_state_change(old_state, self.detection_state, "detection_completed")
                
                # Set result and update lights ONCE
                if not sealant_result:
                    self.detection_result = 'Pass'
                    current_status = self._call_light_once('Pass')
                    print("Detection result: PASS - Light changed to GREEN")
                else:
                    self.detection_result = 'NG'
                    current_status = self._call_light_once('NG')
                    print("Detection result: NG - Light changed to RED")
                
                # Save detection image
                self._save_img_data(raw_frame, sealant_boxes, sealant_result)
                
                # === NEW: Write statistics log (CSV) ===
                if self.statistics_logger:
                    cycle_time_ms = int((time.time() - self.detection_start_time) * 1000)
                    total_count = len(self.sealant_detection_results)
                    ng_count = sum(1 for is_broken in self.sealant_detection_results.values() if is_broken)
                    ok_count = total_count - ng_count
                    self.statistics_logger.log_detection(
                        sealant_result, ok_count, ng_count, total_count, cycle_time_ms
                    )
                
                # === V6.0: Build pending_review data for rejudge log ===
                self.carrier_count += 1
                self.user_reviewed_flag = False  # Reset for new carrier
                self.current_raw_frame = raw_frame.copy()
                self.current_sealant_boxes = sealant_boxes.copy() if sealant_boxes else []
                
                # Build confidence scores mapping
                self.current_scores = {}
                for i, box in enumerate(sealant_boxes):
                    box_key = f"{box[0]}_{box[1]}_{box[2]}_{box[3]}"
                    # Find corresponding score from YOLO detection
                    # scores array corresponds to boxes array, need to find sealant indices
                    for j, (b, cat) in enumerate(zip(boxes, categories)):
                        if cat == self.class_map.get('sealant', 1):
                            if np.array_equal(b, box):
                                self.current_scores[box_key] = float(scores[j]) if j < len(scores) else 0.0
                                break
                
                # Build pending_review structure
                self.pending_review = {
                    'carrier_id': self.carrier_count,
                    'frame_number': self.frame_count,
                    'user_reviewed': False,
                    'sealants': []
                }
                
                for i, box in enumerate(sealant_boxes):
                    box_key = f"{box[0]}_{box[1]}_{box[2]}_{box[3]}"
                    is_broken = self.sealant_detection_results.get(box_key, False)
                    ai_result = 'NG' if is_broken else 'Pass'
                    confidence = self.current_scores.get(box_key, 0.0)
                    
                    self.pending_review['sealants'].append({
                        'index': i,
                        'bbox': [int(box[0]), int(box[1]), int(box[2] - box[0]), int(box[3] - box[1])],
                        'ai_result': ai_result,
                        'confidence': confidence,
                        'user_agrees': True  # Default: agree with AI
                    })
                
                print(f"[V6.0] Carrier #{self.carrier_count} pending review created with {len(self.pending_review['sealants'])} sealants")
                
                stop_time = 0
        else:
            stop_time = 0
            
        return current_status, stop_time, centroid
    
    def _call_light_once(self, status):
        """Call light control only when status actually changes"""
        if self.last_light_status != status:
            print(f"Light status change: {self.last_light_status} → {status}")
            result = self._call_light(status)
            if result == status:  # 成功才更新狀態
                self.last_light_status = status
                return status
            else:
                print(f"⚠️ Light change failed, retry next cycle")
                return self.last_light_status  # 返回舊狀態，下次會重試
        return status
    
    def _log_detection_results(self, sealant_result, frame_shape, centroid):
        """Log detection results"""
        if not self.logger:
            return
            
        # Calculate detection statistics
        total_sealants = len(self.sealant_detection_results)
        ng_count = sum(1 for is_broken in self.sealant_detection_results.values() if is_broken)
        ok_count = total_sealants - ng_count
        
        # Prepare detection details for logging
        detection_details = []
        for i, (box_key, is_broken) in enumerate(self.sealant_detection_results.items(), 1):
            box_coords = box_key.split('_')
            detection_details.append({
                "sealant_id": i,
                "box_coordinates": {
                    "x1": int(box_coords[0]),
                    "y1": int(box_coords[1]),
                    "x2": int(box_coords[2]),
                    "y2": int(box_coords[3])
                },
                "status": "NG" if is_broken else "OK",
                "is_broken": is_broken
            })
        
        h, w = frame_shape[:2]
        frame_info = {
            "frame_width": w,
            "frame_height": h,
            "carrier_centroid": centroid,
            "source": self.source_name,
            "detection_state": self.detection_state
        }
        
        # Log detection results
        self.logger.log_detection_result(
            total_sealants, ok_count, ng_count,
            detection_details, frame_info
        )
    
    def _print_statistics(self):
        """Print processing statistics"""
        elapsed_time = time.time() - self.start_time
        fps = self.frame_count / elapsed_time if elapsed_time > 0 else 0
        detection_rate = (self.detection_count / self.frame_count * 100) if self.frame_count > 0 else 0
        
        stats = {
            "frames_processed": self.frame_count,
            "detections": self.detection_count,
            "detection_rate_percent": round(detection_rate, 1),
            "average_fps": round(fps, 1),
            "running_time_seconds": round(elapsed_time, 1),
            "detection_state": self.detection_state,
            "detection_result": self.detection_result
        }
        
        print(f"\n=== Statistics ===")
        print(f"Frames processed: {stats['frames_processed']}")
        print(f"Detections: {stats['detections']}")
        print(f"Detection rate: {stats['detection_rate_percent']}%")
        print(f"Average FPS: {stats['average_fps']}")
        print(f"Running time: {stats['running_time_seconds']}s")
        print(f"Detection state: {stats['detection_state']}")
        print(f"Detection result: {stats['detection_result']}")
        print("==================\n")
        
        # Log performance statistics
        if self.logger:
            self.logger.log_performance_stats(stats)
    
    def _reset_statistics(self):
        """Reset statistics"""
        self.frame_count = 0
        self.detection_count = 0
        self.start_time = time.time()
        print("Statistics reset")
        if self.logger:
            self.logger.log_system_event("statistics_reset")
    
    def _print_performance(self):
        """Print performance info"""
        elapsed_time = time.time() - self.start_time
        fps = self.frame_count / elapsed_time if elapsed_time > 0 else 0
        print(f"Performance: {self.frame_count} frames, {fps:.1f} FPS, State: {self.detection_state}")
    
    def stop(self):
        """Stop video processor and cleanup resources"""
        print("Stopping video processor...")
        self.running = False
        self.stop_event.set()
        
        # V6.0: Write any pending review before stopping
        if self.pending_review:
            print("[V6.0] Writing pending review before shutdown...")
            self._write_pending_review_log()
        
        # V6.0: Close manual review window if open
        if self.manual_review_window and self.manual_review_window.is_open:
            self.manual_review_window.close()
        
        # Wait for threads to stop
        if self.capture_thread and self.capture_thread.is_alive():
            self.capture_thread.join(timeout=2.0)
        if self.yolo_thread and self.yolo_thread.is_alive():
            self.yolo_thread.join(timeout=2.0)
        
        # Release video capture
        if self.cap:
            self.cap.release()
        
        # Close display windows
        cv2.destroyAllWindows()
        
        # Log final statistics and close logger
        if self.logger:
            elapsed_time = time.time() - self.start_time
            fps = self.frame_count / elapsed_time if elapsed_time > 0 else 0
            final_stats = {
                "session_duration_seconds": round(elapsed_time, 1),
                "total_frames_processed": self.frame_count,
                "total_detections": self.detection_count,
                "final_average_fps": round(fps, 1),
                "final_detection_state": self.detection_state,
                "final_detection_result": self.detection_result,
                "total_carriers": self.carrier_count  # V6.0: Add carrier count
            }
            self.logger.log_performance_stats(final_stats)
            self.logger.close()
        
        # Close statistics logger
        if self.statistics_logger:
            self.statistics_logger.close()
        
        # Stop video recording
        if self.video_recorder:
            self.video_recorder.stop_recording()
        
        print("Video processor stopped")
    
    def _toggle_recording(self):
        """Toggle video recording on/off"""
        # Check if frame_shape is available
        if self.frame_shape is None:
            print("Cannot start recording: frame shape not available")
            return
        
        # Initialize recorder if not exists
        if not self.video_recorder:
            session_id = self.logger.session_id if self.logger else datetime.now().strftime('%Y%m%d_%H%M%S')
            self.video_recorder = VideoRecorder(
                self.recording_config,
                self.frame_shape,
                session_id,
                self.logger
            )
        
        if self.video_recorder:
            if self.video_recorder.is_recording:
                # Stop recording
                self.video_recorder.stop_recording()
                print("Recording stopped by user")
                if self.logger:
                    self.logger.log_system_event("recording_stopped_by_user")
            else:
                # Start new recording - use start_new_recording method
                self.video_recorder.start_new_recording()
                
                if self.video_recorder.is_recording:
                    print("Recording started by user")
                    if self.logger:
                        self.logger.log_system_event("recording_started_by_user", {
                            "filename": self.video_recorder.output_filename
                        })
                else:
                    print("Failed to start recording")
    
    # === V6.0: Manual Review Methods ===
    
    def _open_manual_review(self):
        """Open manual review window (V6.0)"""
        if self.detection_state != 'RESULT_HOLD':
            print("[V6.0] Cannot open review: not in RESULT_HOLD state")
            return
        
        if not self.pending_review or self.current_raw_frame is None:
            print("[V6.0] Cannot open review: no pending data")
            return
        
        # Mark that user pressed M key
        self.user_reviewed_flag = True
        self.pending_review['user_reviewed'] = True
        
        print(f"[V6.0] Opening manual review for Carrier #{self.pending_review['carrier_id']}")
        
        # If fullscreen is active, exit it so the review window can appear on top
        if self.is_fullscreen:
            self._toggle_fullscreen(self.display_config['window_name'])
            self.fullscreen_paused_for_review = True
            print("[V6.0] Fullscreen paused for manual review")
        
        # Create manual review window if needed
        padding = self.rejudge_config.get('review_window_padding', 50)
        thickness = self.rejudge_config.get('review_border_thickness', 3)
        
        if not self.manual_review_window:
            self.manual_review_window = ManualReviewWindow(padding=padding, border_thickness=thickness)
        
        # Prepare sealant boxes as list of arrays
        sealant_boxes_list = []
        for sealant in self.pending_review['sealants']:
            bbox = sealant['bbox']
            # Convert [x, y, w, h] back to [x1, y1, x2, y2]
            x1, y1 = bbox[0], bbox[1]
            x2, y2 = bbox[0] + bbox[2], bbox[1] + bbox[3]
            sealant_boxes_list.append(np.array([x1, y1, x2, y2]))
        
        # Build confidence dict
        confidences = {}
        for sealant in self.pending_review['sealants']:
            bbox = sealant['bbox']
            x1, y1 = bbox[0], bbox[1]
            x2, y2 = bbox[0] + bbox[2], bbox[1] + bbox[3]
            box_key = f"{x1}_{y1}_{x2}_{y2}"
            confidences[box_key] = sealant.get('confidence', 0.0)
        
        # Open review window, passing current pending_review states to restore previous user_agrees
        previous_states = self.pending_review.get('sealants', None)
        success = self.manual_review_window.open(
            self.current_raw_frame,
            sealant_boxes_list,
            self.sealant_detection_results,
            confidences,
            previous_states=previous_states
        )
        
        if success:
            print("[V6.0] Review window opened")
            if self.logger:
                self.logger.log_system_event("manual_review_opened", {
                    "carrier_id": self.pending_review['carrier_id'],
                    "sealant_count": len(self.pending_review['sealants'])
                })
        else:
            print("[V6.0] Failed to open review window")
    
    def _handle_review_complete(self, result):
        """Handle manual review completion (V6.0)"""
        if not self.manual_review_window or not self.pending_review:
            return
        
        if result == 'confirm':
            # Get user modifications from review window
            review_results = self.manual_review_window.get_results()
            
            # Update pending_review using source_index to handle skipped crops correctly
            for state in review_results:
                source_index = state.get('source_index')
                if source_index is not None and source_index < len(self.pending_review['sealants']):
                    self.pending_review['sealants'][source_index]['user_agrees'] = state.get('user_agrees', True)
            
            # Count modifications
            disagree_count = sum(1 for s in self.pending_review['sealants'] if not s.get('user_agrees', True))
            print(f"[V6.0] Review confirmed: {disagree_count} modifications")
            
            if self.logger:
                self.logger.log_system_event("manual_review_confirmed", {
                    "carrier_id": self.pending_review['carrier_id'],
                    "modifications": disagree_count
                })
        
        elif result == 'cancel':
            # Reset all to agree (auto-agree)
            for sealant in self.pending_review['sealants']:
                sealant['user_agrees'] = True
            
            print("[V6.0] Review cancelled - auto-agree applied")
            
            if self.logger:
                self.logger.log_system_event("manual_review_cancelled", {
                    "carrier_id": self.pending_review['carrier_id']
                })
        
        # Close the review window
        self.manual_review_window.close()
        
        # Restore fullscreen if it was paused for review
        if self.fullscreen_paused_for_review:
            self._toggle_fullscreen(self.display_config['window_name'])
            self.fullscreen_paused_for_review = False
            print("[V6.0] Fullscreen restored after manual review")
    
    def _write_pending_review_log(self):
        """Write pending review data to rejudge CSV log (V6.0)"""
        if not self.pending_review:
            return
        
        if self.review_log_written:
            return
        
        if not self.rejudge_logger:
            print("[V6.0] Warning: rejudge_logger not initialized")
            return
        
        # Close review window if still open
        if self.manual_review_window and self.manual_review_window.is_open:
            # Treat as cancel - auto agree
            for sealant in self.pending_review['sealants']:
                sealant['user_agrees'] = True
            self.manual_review_window.close()
        
        # Write to CSV
        self.rejudge_logger.write_records(self.pending_review, self.source_name)
        self.review_log_written = True
        
        if self.logger:
            self.logger.log_system_event("rejudge_log_written", {
                "carrier_id": self.pending_review['carrier_id'],
                "user_reviewed": self.pending_review.get('user_reviewed', False),
                "sealant_count": len(self.pending_review['sealants'])
            })
    
    # Include all the helper methods from the original code
    def _cut_sealant_detection(self, frame, boxes):
        """Detect if sealant is properly cut and store individual results"""
        if len(boxes) == 0:
            return False
            
        frame_hsl = cv2.cvtColor(frame, cv2.COLOR_BGR2HLS)
        result_list = []
        self.sealant_detection_results = {}
        
        for idx, box in enumerate(boxes):
            try:
                crop_img = frame_hsl[box[1]:box[3], box[0]:box[2], :]
                if crop_img.size == 0:
                    continue
                    
                kmeans_img, center = self._img_kmeans(crop_img[:, :, 1])
                seg_sealant = cv2.inRange(kmeans_img, max(center), max(center))
                
                cnt_sealant, _ = cv2.findContours(seg_sealant, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
                
                if cnt_sealant:
                    cnt_area = [cv2.contourArea(cnt) for cnt in cnt_sealant]
                    sealant_idx = np.argmax(cnt_area)
                    closed_loop = self._loop_closed_identify(cnt_sealant[sealant_idx])
                    result_list.append(closed_loop)
                    
                    box_key = f"{box[0]}_{box[1]}_{box[2]}_{box[3]}"
                    self.sealant_detection_results[box_key] = not closed_loop
                    
            except Exception as e:
                error_msg = f"Sealant detection error: {e}"
                print(error_msg)
                if self.logger:
                    self.logger.log_error("sealant_detection_error", e, {
                        "box_index": idx,
                        "box_coordinates": box.tolist() if hasattr(box, 'tolist') else str(box)
                    })
                continue
        
        return any(item == False for item in result_list)
    
    def _loop_closed_identify(self, cnt):
        """Check if contour is a closed loop"""
        if len(cnt) < 3:
            return False
        epsilon = 0.02 * cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, epsilon, True)
        return cv2.isContourConvex(approx)
    
    def _img_kmeans(self, img):
        """K-means clustering for image segmentation"""
        Z = img.reshape(-1).astype(np.float32)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1)
        
        ret, label, center = cv2.kmeans(
            Z, 
            self.detection_config['kmeans_clusters'], 
            None, 
            criteria, 
            self.detection_config['kmeans_rounds'], 
            cv2.KMEANS_RANDOM_CENTERS
        )
        center = center.astype(np.uint8)
        res = center[label.flatten()]
        
        return res.reshape(img.shape), center
    
    def _call_light(self, status):
        """Control stack light status"""
        commands = {
            'Standby': ['2', '0', '1', '0', '0', '0'],
            'Tracking': ['2', '0', '2', '0', '0', '0'],
            'Pass': ['2', '0', '0', '1', '0', '0'],
            'NG': ['2', '1', '0', '0', '0', '0']
        }
        
        try:
            if status in commands:
                result = subprocess.call(['python', self.paths['stack_light_path']] + commands[status])
                if result != 0:
                    print(f"⚠️ Light command failed with code: {result}")
                    return None
                if status == 'NG':
                    subprocess.call(['python', self.paths['stack_light_path']] + ['4', '3', '1', '1', '14'])
        except Exception as e:
            error_msg = f"Light control error: {e}"
            print(error_msg)
            if self.logger:
                self.logger.log_error("light_control_error", e, {"status": status})
            return None
        
        print(f'Status: {status}')
        if self.logger:
            self.logger.log_system_event("status_change", {"new_status": status})
        return status
    
    def _save_img_data(self, frame, boxes, result):
        """Save image data for validation"""
        try:
            # Create validation directory if it doesn't exist
            validation_dir = self.paths['validation_path']
            if not os.path.exists(validation_dir):
                os.makedirs(validation_dir)
            
            # Generate filename with timestamp and result
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')[:-3]  # milliseconds
            result_str = 'NG' if result else 'OK'
            filename = f"{validation_dir}{timestamp}_{result_str}.jpg"
            
            # Save the frame
            cv2.imwrite(filename, frame)
            
            # Log the save operation
            if self.logger:
                self.logger.log_system_event("image_saved", {
                    "filename": filename,
                    "result": result_str,
                    "detection_state": self.detection_state
                })
                
            print(f"Validation image saved: {filename}")
            
        except Exception as e:
            error_msg = f"Failed to save validation image: {e}"
            print(error_msg)
            if self.logger:
                self.logger.log_error("image_save_error", e)
    
    def _check_correct_carrier(self, categories, boxes):
        """Check for correct carrier detection"""
        carrier_indices = [i for i, cat in enumerate(categories) if cat == self.class_map['carrier']]
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
        best_iou = 0.0
        best_match = None
        
        for box_key in stored_box_keys:
            iou = self._calculate_iou(current_box, box_key)
            if iou > best_iou and iou >= iou_threshold:
                best_iou = iou
                best_match = box_key
        
        return best_match
    
    def _draw_detections(self, frame, boxes, categories, correct_carrier_box):
        """Draw detection boxes and return sealant boxes"""
        sealant_boxes = []
        
        for i, (box, category) in enumerate(zip(boxes, categories)):
            color = (0, 255, 0) if category == self.class_map['carrier'] else (255, 0, 0)
            
            # Only draw carrier if it's the correct one
            if category == self.class_map['carrier']:
                if np.array_equal(box, correct_carrier_box):
                    cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), color, 2)
                    #cv2.putText(frame, 'carrier', (box[0], box[1]-10), 
                    #           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
            else:
                # This is a sealant
                sealant_boxes.append(box)
                
                # Determine color based on detection result using IoU matching
                if self.detection_state == 'RESULT_HOLD' and self.sealant_detection_results:
                    # Use IoU matching instead of exact coordinate matching
                    matched_box_key = self._find_matching_box(box, self.sealant_detection_results.keys())
                    if matched_box_key:
                        is_broken = self.sealant_detection_results[matched_box_key]
                        color = (0, 0, 255) if is_broken else (0, 255, 0)  # Red if broken, green if OK
                        status_text = 'NG' if is_broken else 'OK'
                        #cv2.putText(frame, status_text, (box[0], box[1]-25), 
                        #           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
                
                cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), color, 2)
                #cv2.putText(frame, 'sealant', (box[0], box[1]-10), 
                #           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
        
        return sealant_boxes
    
    def _get_screen_size(self):
        """Get actual screen size using tkinter (cached)"""
        if self.cached_screen_size is not None:
            return self.cached_screen_size
        
        try:
            root = tk.Tk()
            root.withdraw()  # Hide the window
            screen_width = root.winfo_screenwidth()
            screen_height = root.winfo_screenheight()
            root.destroy()
            self.cached_screen_size = (screen_width, screen_height)
            return screen_width, screen_height
        except Exception:
            # Fallback to default values
            self.cached_screen_size = (1920, 1080)
            return 1920, 1080
    
    def _log_heartbeat(self):
        """Log heartbeat for system health monitoring (24/7 mode)"""
        current_time = time.time()
        
        if current_time - self.last_heartbeat_time >= self.heartbeat_interval:
            self.last_heartbeat_time = current_time
            
            # Calculate runtime statistics
            elapsed_time = current_time - self.start_time
            fps = self.frame_count / elapsed_time if elapsed_time > 0 else 0
            
            # Get memory usage if available
            memory_info = self._get_memory_usage()
            
            heartbeat_data = {
                "uptime_seconds": round(elapsed_time, 1),
                "uptime_hours": round(elapsed_time / 3600, 2),
                "total_frames": self.frame_count,
                "total_detections": self.detection_count,
                "average_fps": round(fps, 1),
                "detection_state": self.detection_state,
                "camera_reconnect_attempts": self.camera_reconnect_attempts,
                "memory_usage": memory_info
            }
            
            if self.logger:
                self.logger.log_system_event("heartbeat", heartbeat_data)
            
            # Print brief status to console
            print(f"[HEARTBEAT] Uptime: {elapsed_time/3600:.1f}h | Frames: {self.frame_count} | FPS: {fps:.1f} | State: {self.detection_state}")
    
    def _get_memory_usage(self):
        """Get current memory usage statistics"""
        try:
            import psutil
            process = psutil.Process()
            mem_info = process.memory_info()
            return {
                "rss_mb": round(mem_info.rss / (1024 * 1024), 1),
                "vms_mb": round(mem_info.vms / (1024 * 1024), 1)
            }
        except ImportError:
            return {"note": "psutil not available"}
        except Exception:
            return {"note": "error getting memory info"}
    
    def _create_display_window(self, window_name):
        """Create display window with proper flags for fullscreen support"""
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, 800, 600)  # Initial size
    
    def _display_frame(self, frame, window_name):
        """Display frame with auto-fit capability"""
        try:
            display_frame = frame.copy()
            
            if self.fit_screen_mode and self.display_config.get('auto_fit_screen', True):
                # Get actual screen size (cached)
                screen_width, screen_height = self._get_screen_size()
                
                # Get frame dimensions
                frame_height, frame_width = display_frame.shape[:2]
                
                # Calculate scale to fit screen
                scale_x = screen_width * 0.85 / frame_width  # Use 85% of screen width
                scale_y = screen_height * 0.85 / frame_height  # Use 85% of screen height
                scale = min(scale_x, scale_y)
                
                # Apply min/max scale limits
                max_scale = self.display_config.get('max_scale', 2.0)
                min_scale = self.display_config.get('min_scale', 0.2)
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

            # --- Bottom bar: general shortcuts (white, always visible) ---
            shortcut_text = "[Q] Quit  [S] Screenshot  [F] Fit Screen  [G] Fullscreen  [P] Statistics  [R] Reset  [V] Recording"
            text_size = cv2.getTextSize(shortcut_text, font, font_scale, thickness)[0]
            bar_x = (dw - text_size[0]) // 2
            bar_y = dh - 12
            overlay = display_frame.copy()
            cv2.rectangle(overlay, (0, dh - 30), (dw, dh), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.5, display_frame, 0.5, 0, display_frame)
            cv2.putText(display_frame, shortcut_text, (bar_x, bar_y),
                        font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)

            # --- M key hint: center-right, blue, only in RESULT_HOLD ---
            if self.detection_state == 'RESULT_HOLD':
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
            if self.logger:
                self.logger.log_error("display_error", e)
    
    def _toggle_fit_screen(self):
        """Toggle fit screen mode on/off"""
        self.fit_screen_mode = not self.fit_screen_mode
        mode_str = "ON" if self.fit_screen_mode else "OFF"
        print(f"Fit screen mode: {mode_str}")
        if self.logger:
            self.logger.log_system_event("fit_screen_toggled", {"mode": mode_str})
    
    def _take_screenshot(self, frame):
        """Take a screenshot and save to file"""
        try:
            # Create screenshot directory if it doesn't exist
            if not os.path.exists(self.screenshot_path):
                os.makedirs(self.screenshot_path)
            
            # Generate filename with timestamp
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')[:-3]
            filename = f"{self.screenshot_path}screenshot_{timestamp}.jpg"
            
            # Save the frame
            cv2.imwrite(filename, frame)
            
            print(f"Screenshot saved: {filename}")
            if self.logger:
                self.logger.log_system_event("screenshot_saved", {"filename": filename})
                
        except Exception as e:
            error_msg = f"Screenshot error: {e}"
            print(error_msg)
            if self.logger:
                self.logger.log_error("screenshot_error", e)
    
    def _toggle_fullscreen(self, window_name):
        """Toggle fullscreen mode"""
        try:
            self.is_fullscreen = not self.is_fullscreen
            if self.is_fullscreen:
                cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
                print("Fullscreen mode: ON")
            else:
                cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)
                print("Fullscreen mode: OFF")
            
            if self.logger:
                self.logger.log_system_event("fullscreen_toggled", {"fullscreen": self.is_fullscreen})
        except Exception as e:
            print(f"Fullscreen toggle error: {e}")
    
    def _add_ui_elements(self, frame, current_status):
        """Add UI elements to frame"""
        h, w = frame.shape[:2]
        
        # Add crosshair if enabled
        if self.display_config.get('show_crosshair', True):
            cv2.line(frame, (w//2-20, h//2), (w//2+20, h//2), (255, 255, 255), 1)
            cv2.line(frame, (w//2, h//2-20), (w//2, h//2+20), (255, 255, 255), 1)
        
        # Add timestamp if enabled
        if self.display_config.get('show_timestamp', True):
            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            cv2.putText(frame, timestamp, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3, cv2.LINE_AA)
        
        # Add status if enabled
        if self.display_config.get('show_status', True):
            status_text = f"Status: {current_status}"
            cv2.putText(frame, status_text, (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3, cv2.LINE_AA)
        
        # === NEW: Add detection state information ===
        state_text = f"State: {self.detection_state}"
        state_color = {
            'WAITING': (0, 255, 255),      # Yellow
            'DETECTED': (255, 165, 0),     # Orange  
            'RESULT_HOLD': (0, 255, 0) if self.detection_result == 'Pass' else (0, 0, 255)  # Green/Red
        }.get(self.detection_state, (255, 255, 255))
        
        cv2.putText(frame, state_text, (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 1.5, state_color, 3, cv2.LINE_AA)
        
        # Show detection result if available
        if self.detection_result:
            result_text = f"Result: {self.detection_result}"
            result_color = (0, 255, 0) if self.detection_result == 'Pass' else (0, 0, 255)
            cv2.putText(frame, result_text, (10, 190), cv2.FONT_HERSHEY_SIMPLEX, 1.5, result_color, 3, cv2.LINE_AA)
        
        # Add sealant detection results if available
        if self.detection_state == 'RESULT_HOLD' and self.sealant_detection_results:
            broken_count = sum(1 for is_broken in self.sealant_detection_results.values() if is_broken)
            total_count = len(self.sealant_detection_results)
            
            if broken_count > 0:
                status_text = f"Broken: {broken_count}/{total_count}"
                cv2.putText(frame, status_text, (10, 220), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3, cv2.LINE_AA)
            else:
                cv2.putText(frame, "All sealant OK", (10, 220), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3, cv2.LINE_AA)
        



# Enhanced ConfigManager class with logging support
class ConfigManager:
    """Configuration manager for reading config.ini with logging support"""
    
    def __init__(self, config_file='config_v6.0.ini'):
        self.config = configparser.ConfigParser()
        self.config_file = config_file
        self.load_config()
        
    def load_config(self):
        """Load configuration from ini file"""
        if not os.path.exists(self.config_file):
            self.create_default_config()
        
        #self.config.read(self.config_file)
        self.config.read(self.config_file, encoding="utf-8-sig")
    def create_default_config(self):
        """Create default configuration file if it doesn't exist"""
        print(f"Creating default config file: {self.config_file}")
        
        self.config['PATHS'] = {
            'model_path': './models/best.pt',
            'video_path': './videoData/*.avi',
            'validation_path': './validation/',
            'stack_light_path': 'LR6main.py'
        }
        
        self.config['VIDEO_SOURCE'] = {
            'mode': 'camera',
            'camera_index': '0',
            'camera_width': '1080',
            'camera_height': '1920',
            'camera_focus': '0',
            'video_index': '0',
            # TIS DFK camera settings
            'tis_dll_path': './tisgrabber_x64.dll',
            'tis_device_name': 'DFK 33UX178',
            'tis_camera_index': '-1'
        }
        
        self.config['YOLO_SETTINGS'] = {
            'confidence_threshold': '0.55',
            'iou_threshold': '0.45',
            'max_detections': '100',
            'use_half_precision': 'true',
            'use_augmentation': 'false'
        }
        
        self.config['DETECTION_SETTINGS'] = {
            'fps': '15',
            'stop_threshold_seconds': '0.95',
            'distance_threshold_pixels': '10',
            'kmeans_clusters': '3',
            'kmeans_rounds': '10',
            # Tolerance mechanism for detection box flickering
            'absent_tolerance_frames': '5',
            'result_hold_min_seconds': '2.0'
        }
        
        self.config['DISPLAY_SETTINGS'] = {
            'show_crosshair': 'true',
            'show_timestamp': 'true',
            'show_status': 'true',
            'window_name': 'frame',
            'auto_fit_screen': 'true',
            'max_scale': '1.0',
            'min_scale': '0.1'
        }
        
        self.config['DEVICE_SETTINGS'] = {
            'device': 'cpu'
        }
        
        self.config['CLASSES'] = {
            'carrier': '0',
            'sealant': '1'
        }
        
        # Add logging settings
        self.config['LOGGING_SETTINGS'] = {
            'log_level': 'INFO',
            'save_detection_log': 'true',
            'log_file_path': './logs/detection.log'
        }
        
        # Add recording settings
        self.config['RECORDING_SETTINGS'] = {
            'enable_recording': 'false',
            'record_type': 'inference',  # 'raw' or 'inference'
            'output_path': './recordings/',
            'video_format': 'mp4',
            'fps': '15',
            'quality': 'medium'  # 'low', 'medium', 'high'
        }
        
        # Save default config
        with open(self.config_file, 'w') as f:
            self.config.write(f)
    
    def get_paths(self):
        """Get all path configurations"""
        return {
            'model_path': self.config.get('PATHS', 'model_path'),
            'video_path': self.config.get('PATHS', 'video_path'),
            'validation_path': self.config.get('PATHS', 'validation_path'),
            'stack_light_path': self.config.get('PATHS', 'stack_light_path')
        }
    
    def get_video_source_config(self):
        """Get video source configuration"""
        return {
            'mode': self.config.get('VIDEO_SOURCE', 'mode'),
            'camera_index': self.config.getint('VIDEO_SOURCE', 'camera_index'),
            'camera_width': self.config.getint('VIDEO_SOURCE', 'camera_width'),
            'camera_height': self.config.getint('VIDEO_SOURCE', 'camera_height'),
            'camera_focus': self.config.getint('VIDEO_SOURCE', 'camera_focus'),
            'video_index': self.config.getint('VIDEO_SOURCE', 'video_index'),
            # TIS DFK camera settings
            'tis_dll_path': self.config.get('VIDEO_SOURCE', 'tis_dll_path', fallback='./tisgrabber_x64.dll'),
            'tis_device_name': self.config.get('VIDEO_SOURCE', 'tis_device_name', fallback='DFK 33UX178'),
            'tis_camera_index': self.config.getint('VIDEO_SOURCE', 'tis_camera_index', fallback=-1)
        }
    
    def get_yolo_config(self):
        """Get YOLO configuration"""
        return {
            'confidence_threshold': self.config.getfloat('YOLO_SETTINGS', 'confidence_threshold'),
            'iou_threshold': self.config.getfloat('YOLO_SETTINGS', 'iou_threshold'),
            'max_detections': self.config.getint('YOLO_SETTINGS', 'max_detections'),
            'use_half_precision': self.config.getboolean('YOLO_SETTINGS', 'use_half_precision'),
            'use_augmentation': self.config.getboolean('YOLO_SETTINGS', 'use_augmentation')
        }
    
    def get_detection_config(self):
        """Get detection configuration"""
        return {
            'fps': self.config.getint('DETECTION_SETTINGS', 'fps'),
            'stop_threshold_seconds': self.config.getfloat('DETECTION_SETTINGS', 'stop_threshold_seconds'),
            'distance_threshold_pixels': self.config.getint('DETECTION_SETTINGS', 'distance_threshold_pixels'),
            'kmeans_clusters': self.config.getint('DETECTION_SETTINGS', 'kmeans_clusters'),
            'kmeans_rounds': self.config.getint('DETECTION_SETTINGS', 'kmeans_rounds'),
            # Tolerance mechanism for detection box flickering
            'absent_tolerance_frames': self.config.getint('DETECTION_SETTINGS', 'absent_tolerance_frames', fallback=5),
            'result_hold_min_seconds': self.config.getfloat('DETECTION_SETTINGS', 'result_hold_min_seconds', fallback=2.0)
        }
    
    def get_display_config(self):
        """Get display configuration"""
        return {
            'show_crosshair': self.config.getboolean('DISPLAY_SETTINGS', 'show_crosshair'),
            'show_timestamp': self.config.getboolean('DISPLAY_SETTINGS', 'show_timestamp'),
            'show_status': self.config.getboolean('DISPLAY_SETTINGS', 'show_status'),
            'window_name': self.config.get('DISPLAY_SETTINGS', 'window_name'),
            'auto_fit_screen': self.config.getboolean('DISPLAY_SETTINGS', 'auto_fit_screen', fallback=True),
            'max_scale': self.config.getfloat('DISPLAY_SETTINGS', 'max_scale', fallback=1.0),
            'min_scale': self.config.getfloat('DISPLAY_SETTINGS', 'min_scale', fallback=0.1)
        }
    
    def get_device(self):
        """Get device configuration"""
        return self.config.get('DEVICE_SETTINGS', 'device')
    
    def get_classes(self):
        """Get class configuration"""
        classes = {}
        for name, index in self.config['CLASSES'].items():
            classes[name] = int(index)
        return classes
    
    def get_logging_config(self):
        """Get logging configuration"""
        return {
            'log_level': self.config.get('LOGGING_SETTINGS', 'log_level', fallback='INFO'),
            'save_detection_log': self.config.getboolean('LOGGING_SETTINGS', 'save_detection_log', fallback=True),
            'log_file_path': self.config.get('LOGGING_SETTINGS', 'log_file_path', fallback='./logs/detection.log')
        }
    
    def get_recording_config(self):
        """Get recording configuration"""
        return {
            'enable_recording': self.config.getboolean('RECORDING_SETTINGS', 'enable_recording', fallback=False),
            'record_type': self.config.get('RECORDING_SETTINGS', 'record_type', fallback='inference'),
            'output_path': self.config.get('RECORDING_SETTINGS', 'output_path', fallback='./recordings/'),
            'video_format': self.config.get('RECORDING_SETTINGS', 'video_format', fallback='mp4'),
            'fps': self.config.getint('RECORDING_SETTINGS', 'fps', fallback=15),
            'quality': self.config.get('RECORDING_SETTINGS', 'quality', fallback='medium')
        }
    
    def get_rejudge_config(self):
        """Get rejudge/manual review configuration (V6.0)"""
        return {
            'review_window_padding': self.config.getint('REJUDGE_SETTINGS', 'review_window_padding', fallback=50),
            'review_border_thickness': self.config.getint('REJUDGE_SETTINGS', 'review_border_thickness', fallback=3)
        }


def main():
    """Main function with improved error handling"""
    processor = SafeVideoProcessor()
    
    try:
        success = processor.run()
        if success:
            print("Program completed successfully")
        else:
            print("Program failed to initialize")
            return 1
            
    except KeyboardInterrupt:
        print("\nProgram interrupted by user")
    except Exception as e:
        print(f"Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    finally:
        processor.stop()
    
    return 0


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
