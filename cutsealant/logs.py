"""Append-only system events, production statistics and manual-review records."""
import csv
import json
import logging
import os
from datetime import datetime


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
