"""Lights-only PATLITE endurance test; run with the inspection application closed.

Default: eight hours of yellow -> green -> yellow -> red, five seconds per
color. No buzzer commands or final all-off command are sent. USB write success
is not physical lamp readback: observe the tower, especially after five hours.

Windows: double-click run_light_tower_stability_test.bat, or run:
    run_light_tower_stability_test.bat --hours 8 --hold-seconds 5
Logs are written beside this script under logs/light_tower_stability/.
"""

import argparse
from datetime import datetime
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import signal
import subprocess
import sys
import tempfile
import threading
import time
from uuid import uuid4


ROOT = Path(__file__).resolve().parent
SEQUENCE = (
    ("yellow", ("2", "0", "1", "0", "0", "0")),
    ("green", ("2", "0", "0", "1", "0", "0")),
    ("yellow", ("2", "0", "1", "0", "0", "0")),
    ("red", ("2", "1", "0", "0", "0", "0")),
)
OUTPUT_LIMIT = 4096
CLEANUP_TIMEOUT = 1.0


def positive_number(value):
    number = float(value)
    if not math.isfinite(number) or number <= 0:
        raise argparse.ArgumentTypeError("must be a finite number greater than zero")
    return number


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--hours", type=positive_number, default=8.0)
    parser.add_argument("--hold-seconds", type=positive_number, default=5.0,
                        help="wait after each command, including failures (default: 5)")
    parser.add_argument("--timeout-seconds", type=positive_number, default=3.0,
                        help="overall child-process timeout (default: 3)")
    args = parser.parse_args(argv)
    if not math.isfinite(args.hours * 3600):
        parser.error("--hours is too large")
    return args


def stop_child(child):
    """Bound cleanup and confirm exit before permitting another USB owner."""
    # A second Ctrl+C must not interrupt the kill/wait pair.
    main_thread = threading.current_thread() is threading.main_thread()
    previous = signal.signal(signal.SIGINT, signal.SIG_IGN) if main_thread else None
    try:
        if child.poll() is None:
            try:
                child.kill()
            except OSError:
                # The process may have exited between poll() and kill(). A
                # successful wait is still required before another command.
                pass
        child.wait(timeout=CLEANUP_TIMEOUT)
        return True, None
    except (OSError, subprocess.TimeoutExpired) as exc:
        return False, f"{type(exc).__name__}: {exc}"
    finally:
        if main_thread:
            signal.signal(signal.SIGINT, previous)


def output_tail(stream):
    stream.seek(0, os.SEEK_END)
    length = stream.tell()
    stream.seek(max(0, length - OUTPUT_LIMIT))
    return stream.read().decode("utf-8", errors="replace"), length > OUTPUT_LIMIT


def execute_command(command, timeout, cwd):
    """Run one CLI command with bounded output in memory and bounded cleanup."""
    started = time.monotonic()
    result = dict(outcome="failed", exit_code=None, timed_out=False,
                  interrupted=False, termination_confirmed=True, pid=None)
    child = None
    # Files avoid pipe deadlock and keep noisy children out of parent memory.
    with tempfile.TemporaryFile() as stdout, tempfile.TemporaryFile() as stderr:
        try:
            if os.name == "nt":
                options = {"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP}
            else:
                options = {"start_new_session": True}
            child = subprocess.Popen(command, cwd=str(cwd), stdout=stdout,
                                     stderr=stderr, **options)
            result["pid"] = child.pid
            remaining = max(0, timeout - (time.monotonic() - started))
            result["exit_code"] = child.wait(timeout=remaining)
            result["outcome"] = "success" if child.returncode == 0 else "failed"
        except subprocess.TimeoutExpired as exc:
            result.update(outcome="timeout", timed_out=True,
                          error_type=type(exc).__name__, error=str(exc))
        except KeyboardInterrupt:
            result.update(outcome="interrupted", interrupted=True)
        except (OSError, subprocess.SubprocessError) as exc:
            result.update(error_type=type(exc).__name__, error=str(exc))
        finally:
            if child is not None:
                confirmed, error = stop_child(child)
                result.update(termination_confirmed=confirmed, exit_code=child.returncode)
                if not confirmed:
                    result.update(outcome="cleanup_failed", cleanup_error=error)
            result["stdout"], result["stdout_truncated"] = output_tail(stdout)
            result["stderr"], result["stderr_truncated"] = output_tail(stderr)
    result["duration_seconds"] = time.monotonic() - started
    return result


class RunLog:
    def __init__(self, stream, session_id, started):
        self.stream = stream
        self.session_id = session_id
        self.started = started

    def write(self, event, **fields):
        row = dict(timestamp=datetime.now().astimezone().isoformat(),
                   elapsed_seconds=time.monotonic() - self.started,
                   session_id=self.session_id, event=event,
                   physical_readback=False, **fields)
        self.stream.write(json.dumps(row, ensure_ascii=True) + "\n")
        self.stream.flush()


def run_test(args, controller, log):
    deadline = log.started + args.hours * 3600
    counts = dict(attempted=0, succeeded=0, failed=0, timeouts=0,
                  interrupted_commands=0)
    status, exit_code = "completed", 0
    try:
        while time.monotonic() < deadline:
            color, values = SEQUENCE[counts["attempted"] % len(SEQUENCE)]
            command = [sys.executable, str(controller), *values]
            counts["attempted"] += 1
            fields = dict(sequence_number=counts["attempted"], color=color,
                          command_args=command)
            # A start record survives a hang or an external process termination.
            log.write("command_start", **fields)
            print(f"[{time.monotonic() - log.started:9.1f}s] "
                  f'#{counts["attempted"]} requesting {color}', flush=True)
            result = execute_command(command, args.timeout_seconds, controller.parent)
            if result["outcome"] == "success":
                counts["succeeded"] += 1
            elif result["outcome"] == "interrupted":
                counts["interrupted_commands"] += 1
            else:
                counts["failed"] += 1
            counts["timeouts"] += int(result["timed_out"])
            log.write("command_result", **fields, **result, counts=counts.copy())
            print(f'  {result["outcome"].upper()} ({result["duration_seconds"]:.3f}s); '
                  f'OK={counts["succeeded"]} FAIL={counts["failed"]} '
                  f'TIMEOUT={counts["timeouts"]}', flush=True)
            if result["outcome"] not in ("success", "interrupted"):
                print(f'  Error/output: {result.get("error", "")} '
                      f'{result["stderr"] or result["stdout"]}', flush=True)
            if not result["termination_confirmed"]:
                status, exit_code = "aborted", 2
                print("STOPPED: child exit could not be confirmed. See cleanup_error in log.", flush=True)
                break
            if result["interrupted"]:
                status, exit_code = "interrupted", 130
                break
            remaining = deadline - time.monotonic()
            if remaining > 0:
                time.sleep(min(args.hold_seconds, remaining))
    except KeyboardInterrupt:
        status, exit_code = "interrupted", 130
    except Exception as exc:
        status, exit_code = "aborted", 2
        log.write("run_error", error_type=type(exc).__name__, error=str(exc))
        print(f"STOPPED: {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
    if status == "completed" and counts["failed"]:
        exit_code = 1
    log.write("run_summary", status=status, exit_code=exit_code, **counts)
    print(f"Run {status}; exit code {exit_code}. "
          f'Commands={counts["attempted"]}, OK={counts["succeeded"]}, '
          f'FAIL={counts["failed"]}, TIMEOUT={counts["timeouts"]}.', flush=True)
    return exit_code


def main(argv=None):
    args = parse_args(argv)
    controller = ROOT / "LR6main.py"
    if not controller.is_file():
        print(f"ERROR: controller not found: {controller}", file=sys.stderr)
        return 2
    session_id = uuid4().hex
    log_dir = ROOT / "logs" / "light_tower_stability"
    name = datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + session_id[:8] + ".jsonl"
    try:
        log_dir.mkdir(parents=True, exist_ok=True)
        with (log_dir / name).open("x", encoding="utf-8") as stream:
            log = RunLog(stream, session_id, time.monotonic())
            try:
                pyusb_version = importlib.metadata.version("pyusb")
            except importlib.metadata.PackageNotFoundError:
                pyusb_version = "not installed"
            log.write("run_start", hours=args.hours, hold_seconds=args.hold_seconds,
                      timeout_seconds=args.timeout_seconds,
                      cleanup_timeout_seconds=CLEANUP_TIMEOUT,
                      sequence=[color for color, _ in SEQUENCE],
                      reset_after_command=True, buzzer_commands=False,
                      python_executable=sys.executable, python_version=sys.version,
                      platform=platform.platform(), pyusb_version=pyusb_version,
                      controller=str(controller), working_directory=str(controller.parent))
            print(f"Lights-only test: {args.hours:g} hours; {args.hold_seconds:g}s per color.\n"
                  "Sequence: yellow -> green -> yellow -> red. No buzzer commands.\n"
                  "Close the inspection application before running this test.\n"
                  "Observe physical colors, especially after five hours; USB success is not lamp readback.\n"
                  "Ctrl+C stops the test and preserves the last light state.\n"
                  f"Log: {log_dir / name}", flush=True)
            return run_test(args, controller, log)
    except OSError as exc:
        print(f"ERROR: cannot write endurance log: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
