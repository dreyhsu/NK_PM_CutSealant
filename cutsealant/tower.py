"""Nonblocking, serialized PATLITE CLI controller with bounded recovery."""
import json
import os
import subprocess
import sys
import tempfile
import threading
import time
from datetime import datetime
from uuid import uuid4


class TowerController:
    COMMANDS = {
        'Standby': ['2', '0', '1', '0', '0', '0'],
        'Tracking': ['2', '0', '2', '0', '0', '0'],
        'Pass': ['2', '0', '0', '1', '0', '0'],
        'NG': ['2', '1', '0', '0', '0', '0'],
    }

    def __init__(self, context, monotonic=time.monotonic):
        self.context = context
        self.now = monotonic
        self.last_status = None
        self.desired_status = None
        self.health = 'unavailable'
        self.buzzer_outcome = None
        self.session_id = uuid4().hex
        self._condition = threading.Condition()
        self._dispatch = threading.Lock()
        self._stopped = threading.Event()
        self._thread = None
        self._version = 0
        self._attempt = 0
        self._next_attempt = 0
        self._alarm_carrier = None
        self._pending_alarm = None
        self._request_carrier = None

    def _option(self, key, default):
        return self.context.light_tower_config.get(key, default)

    def _log(self, event, **data):
        # Tower diagnostics survive disabling detection logs.
        row = dict(timestamp=datetime.now().isoformat(), session_id=self.session_id,
                   event_type=event, desired_status=self.desired_status,
                   transmitted_status=self.last_status, health=self.health,
                   physical_readback=False, **data)
        try:
            path = self._option('log_path', './logs/tower.jsonl')
            os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
            with open(path, 'a', encoding='utf-8') as stream:
                stream.write(json.dumps(row, ensure_ascii=False) + '\n')
        except Exception as exc:
            print(f'Tower logging failed: {exc}; {row}')

    def start(self):
        with self._condition:
            if self._stopped.is_set() or self._thread is not None:
                return
            self._thread = threading.Thread(target=self._run, name='TowerScheduler', daemon=True)
            self._thread.start()

    def _call_light_once(self, status):
        if status not in self.COMMANDS:
            raise ValueError(f'Invalid tower status: {status}')
        with self._condition:
            if self._stopped.is_set():
                return self.desired_status
            if status != self.desired_status:
                self.desired_status = status
                self._version += 1
                self._attempt = 0
                self._next_attempt = 0
                if status != 'NG':
                    self._pending_alarm = None
                carrier = self.context.carrier_count
                self._request_carrier = carrier
                if status == 'NG' and carrier != self._alarm_carrier:
                    self._alarm_carrier = carrier
                    self._pending_alarm = carrier
                self._condition.notify_all()
        return status  # Requested state, never physical confirmation.

    def retry_latest(self):
        """Operator retry after an exhausted recovery batch (T key)."""
        with self._condition:
            if not self._stopped.is_set():
                self.last_status = None
                self._version += 1
                self._attempt = 0
                self._next_attempt = 0
                self._condition.notify_all()

    _call_light = _call_light_once

    def _run(self):
        while not self._stopped.is_set():
            self.service_once()
            with self._condition:
                self._condition.wait(timeout=0.05)

    def _execute(self, args):
        timeout = float(self._option('command_timeout_seconds', 3.0))
        runner = self.context.command_runner
        if runner is not None:
            # Injected runners must honor the same timeout contract.
            result = runner(args, timeout=timeout)
            return getattr(result, 'returncode', result), str(getattr(result, 'stdout', ''))[-4096:], str(getattr(result, 'stderr', ''))[-4096:]
        # Files avoid pipe deadlock/unbounded memory if the child is noisy.
        with tempfile.TemporaryFile() as out, tempfile.TemporaryFile() as err:
            child = subprocess.Popen(args, stdout=out, stderr=err)
            deadline = self.now() + timeout
            while child.poll() is None:
                if self._stopped.wait(0.05) or self.now() >= deadline:
                    try:
                        child.kill()
                        child.wait(timeout=0.5)
                    except (OSError, subprocess.TimeoutExpired):
                        self._stopped.set()  # Never launch an overlapping USB owner.
                        self.health = 'unavailable'
                        self._log('tower_recovery_exhausted', reason='child termination failed')
                    raise subprocess.TimeoutExpired(args, timeout,
                                                    output=self._tail(out), stderr=self._tail(err))
            return child.returncode, self._tail(out), self._tail(err)

    @staticmethod
    def _tail(stream):
        stream.seek(0, 2)
        stream.seek(max(0, stream.tell() - 4096))
        return stream.read().decode('utf-8', errors='replace')

    def _send(self, operation, args, version, attempt, carrier):
        fields = dict(command_id=uuid4().hex, version=version, operation=operation,
                      attempt=attempt, carrier_id=carrier, command_args=args,
                      reset_after_command=self._option('reset_after_command', True))
        self._log('tower_command_requested', **fields)
        started = self.now()
        try:
            if self._option('mode', 'usb') == 'mock':
                code, stdout, stderr = 0, 'mock; no physical device', ''
            else:
                command = [sys.executable, self.context.paths['stack_light_path'], *args]
                if not self._option('reset_after_command', True):
                    command.append('--no-reset')
                code, stdout, stderr = self._execute(command)
            ok = code == 0
            self._log('tower_command_completed' if ok else 'tower_command_failed',
                      **fields, exit_code=code, stdout=stdout, stderr=stderr,
                      duration_seconds=self.now() - started, transport_success=ok)
            return ok
        except Exception as exc:
            self._log('tower_command_timeout' if isinstance(exc, subprocess.TimeoutExpired)
                      else 'tower_command_failed', **fields, error_type=type(exc).__name__,
                      error=str(exc), stdout=str(getattr(exc, 'output', '') or '')[-4096:],
                      stderr=str(getattr(exc, 'stderr', '') or '')[-4096:],
                      duration_seconds=self.now() - started)
            return False

    def service_once(self):
        """One serialized dispatch; also used by deterministic fake-clock tests."""
        with self._dispatch:
            with self._condition:
                if (self._stopped.is_set() or self.desired_status is None
                        or self._next_attempt is None or self.now() < self._next_attempt):
                    return
                status, version = self.desired_status, self._version
                alarm = self._pending_alarm
                carrier = self._request_carrier
                self._attempt += 1
                attempt = self._attempt
            if attempt > 1:
                self._log('tower_reconnect_attempt', attempt=attempt, version=version)
            ok = self._send('led', self.COMMANDS[status], version, attempt,
                            carrier)
            events = []
            with self._condition:
                if self._stopped.is_set():
                    return
                if version != self._version:
                    self.last_status = None
                    return
                if ok:
                    self.last_status = status
                    self.health = 'connected' if self._option('mode', 'usb') != 'mock' else 'mock'
                    if attempt > 1:
                        events.append(('tower_reconnected', {'version': version}))
                    self._attempt = 0
                    self._next_attempt = self.now() + float(self._option('refresh_seconds', 30))
                else:
                    self.last_status = None
                    self.health = 'retrying'
                    events.append(('tower_disconnected', {'version': version}))
                    if attempt >= int(self._option('max_attempts', 5)):
                        self.health = 'unavailable'
                        self._next_attempt = None
                        events.append(('tower_recovery_exhausted', {'attempt': attempt}))
                    else:
                        self._next_attempt = self.now() + min(
                            float(self._option('retry_max_seconds', 30)),
                            float(self._option('retry_initial_seconds', 1)) * 2 ** (attempt - 1))
                if ok and alarm is not None and self._pending_alarm == alarm:
                    self._pending_alarm = None
                else:
                    alarm = None
            # File I/O must never hold the lock used by UI requests/shutdown.
            for event, data in events:
                self._log(event, **data)
            if not ok:
                return
            if alarm is not None and not self._stopped.is_set():
                with self._condition:
                    if version != self._version:
                        return
                success = self._send('buzzer', ['4', '3', '1', '1', '14'], version, 1, alarm)
                self.buzzer_outcome = 'transport_succeeded' if success else 'failed_or_unknown'

    def stop(self):
        self._stopped.set()
        with self._condition:
            self._condition.notify_all()
        if self._thread and self._thread is not threading.current_thread():
            self._thread.join(timeout=1.5)
