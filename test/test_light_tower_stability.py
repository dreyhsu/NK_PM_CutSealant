"""Endurance-runner checks with fake time and hardware-free CLI children."""
import io
import json
import subprocess
import sys
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

import light_tower_stability_test as endurance


def read_rows(stream):
    return [json.loads(line) for line in stream.getvalue().splitlines()]


def test_defaults_and_invalid_numbers():
    args = endurance.parse_args([])
    assert (args.hours, args.hold_seconds, args.timeout_seconds) == (8, 5, 3)
    for flag in ("--hours", "--hold-seconds", "--timeout-seconds"):
        for value in ("0", "-1", "nan", "inf"):
            with pytest.raises(SystemExit) as error:
                endurance.parse_args([flag, value])
            assert error.value.code == 2


def test_eight_hours_continue_after_five_hour_failure(monkeypatch, tmp_path):
    now = [0.0]
    clock = SimpleNamespace(monotonic=lambda: now[0],
                            sleep=lambda duration: now.__setitem__(0, now[0] + duration))
    monkeypatch.setattr(endurance, "time", clock)
    monkeypatch.setattr(endurance, "print", lambda *a, **kw: None, raising=False)
    class FlushedStream(io.StringIO):
        def write(self, value):
            self.last_written = value
            return super().write(value)

        def flush(self):
            self.last_flushed = self.last_written
            super().flush()

    stream = FlushedStream()
    log = endurance.RunLog(stream, "simulated-eight-hours", 0)
    commands = []

    def command_runner(command, timeout, cwd):
        # The start record must already have been flushed before transport starts.
        assert json.loads(stream.last_flushed)["event"] == "command_start"
        commands.append(command)
        failed = now[0] == 5 * 3600
        return dict(outcome="timeout" if failed else "success", timed_out=failed,
                    interrupted=False, termination_confirmed=True,
                    duration_seconds=0, stdout="raw write length", stderr="",
                    exit_code=-1 if failed else 0)

    monkeypatch.setattr(endurance, "execute_command", command_runner)
    controller = tmp_path / "repo with spaces" / "LR6main.py"
    assert endurance.run_test(endurance.parse_args([]), controller, log) == 1
    assert now[0] == 8 * 3600
    assert len(commands) == 5760
    expected_leds = [["0", "1", "0", "0", "0"],
                     ["0", "0", "1", "0", "0"],
                     ["0", "1", "0", "0", "0"],
                     ["1", "0", "0", "0", "0"]]
    for index, command in enumerate(commands):
        assert command == [sys.executable, str(controller), "2", *expected_leds[index % 4]]
    rows = read_rows(stream)
    failed = [row for row in rows if row.get("outcome") == "timeout"]
    assert len(failed) == 1 and failed[0]["elapsed_seconds"] == 18000
    assert rows[-2]["outcome"] == "success"  # Commands continued and recovered.
    assert rows[-1]["status"] == "completed"
    assert rows[-1]["failed"] == rows[-1]["timeouts"] == 1
    assert rows[-1]["succeeded"] == 5759
    assert all(row["physical_readback"] is False for row in rows)


def test_real_children_failure_timeout_cleanup_and_recovery(tmp_path):
    folder = tmp_path / "repo with spaces"
    folder.mkdir()
    script = folder / "fake controller.py"
    script.write_text("import sys, time\n"
                      "print('usb diagnostic', flush=True)\n"
                      "if sys.argv[1] == 'fail':\n"
                      "    print('simulated USB error', file=sys.stderr, flush=True)\n"
                      "    sys.exit(4)\n"
                      "if sys.argv[1] == 'hang': time.sleep(60)\n", encoding="utf-8")
    failure = endurance.execute_command([sys.executable, str(script), "fail"], 3, folder)
    assert failure["outcome"] == "failed" and failure["exit_code"] == 4
    assert "simulated USB error" in failure["stderr"]
    timeout = endurance.execute_command([sys.executable, str(script), "hang"], .2, folder)
    assert timeout["outcome"] == "timeout"
    assert timeout["timed_out"] and timeout["termination_confirmed"]
    assert timeout["exit_code"] is not None
    assert timeout["duration_seconds"] < 3
    recovered = endurance.execute_command([sys.executable, str(script), "ok"], 3, folder)
    assert recovered["outcome"] == "success"
    assert "usb diagnostic" in recovered["stdout"]


@pytest.mark.parametrize("during", ["command", "hold"])
def test_interruption_cleans_child_and_stops_sequence(monkeypatch, tmp_path, during):
    child = Mock(pid=123, returncode=None)

    def wait(timeout):
        if timeout == endurance.CLEANUP_TIMEOUT:
            child.returncode = -9
            return -9
        if during == "command":
            raise KeyboardInterrupt
        child.returncode = 0
        return 0

    child.wait.side_effect = wait
    child.poll.side_effect = lambda: child.returncode
    launch = Mock(return_value=child)
    monkeypatch.setattr(subprocess, "Popen", launch)
    if during == "hold":
        def interrupt_sleep(duration):
            raise KeyboardInterrupt
        monkeypatch.setattr(endurance, "time", SimpleNamespace(
            monotonic=lambda: 0, sleep=interrupt_sleep))
    stream = io.StringIO()
    log = endurance.RunLog(stream, "interrupt", endurance.time.monotonic())
    assert endurance.run_test(endurance.parse_args([]), tmp_path / "LR6main.py", log) == 130
    assert launch.call_count == 1
    if during == "command":
        child.kill.assert_called_once()
    else:
        child.kill.assert_not_called()
    assert child.wait.call_args.kwargs["timeout"] == endurance.CLEANUP_TIMEOUT
    summary = read_rows(stream)[-1]
    assert summary["status"] == "interrupted"
    assert summary["interrupted_commands"] == (1 if during == "command" else 0)
    assert summary["failed"] == 0


def test_unconfirmed_child_exit_aborts_without_overlap(monkeypatch, tmp_path):
    child = Mock(pid=123, returncode=None)
    child.poll.return_value = None
    child.wait.side_effect = subprocess.TimeoutExpired("fake USB child", 1)
    launch = Mock(return_value=child)
    monkeypatch.setattr(subprocess, "Popen", launch)
    stream = io.StringIO()
    log = endurance.RunLog(stream, "cleanup-failed", endurance.time.monotonic())
    assert endurance.run_test(endurance.parse_args([]), tmp_path / "LR6main.py", log) == 2
    assert launch.call_count == 1
    child.kill.assert_called_once()
    rows = read_rows(stream)
    assert rows[-2]["outcome"] == "cleanup_failed"
    assert rows[-2]["termination_confirmed"] is False
    assert "TimeoutExpired" in rows[-2]["cleanup_error"]
    assert rows[-1]["status"] == "aborted"


def test_main_paths_metadata_and_success_exit(monkeypatch, tmp_path):
    root = tmp_path / "repo with spaces"
    root.mkdir()
    # This fake controller checks that only the expected LED-only arguments arrive.
    (root / "LR6main.py").write_text(
        "import sys\n"
        "assert sys.argv[1:] == ['2', '0', '1', '0', '0', '0']\n"
        "print('fake USB write complete')\n", encoding="utf-8")
    monkeypatch.setattr(endurance, "ROOT", root)
    now = [0.0]
    monkeypatch.setattr(endurance, "time", SimpleNamespace(
        monotonic=lambda: now[0],
        sleep=lambda duration: now.__setitem__(0, now[0] + duration)))
    assert endurance.main(["--hours", ".00002", "--hold-seconds", "1"]) == 0
    log_file, = (root / "logs" / "light_tower_stability").glob("*.jsonl")
    rows = [json.loads(line) for line in log_file.read_text().splitlines()]
    assert rows[0]["event"] == "run_start"
    assert rows[0]["python_executable"] == sys.executable
    assert rows[0]["reset_after_command"] is True
    assert rows[0]["buzzer_commands"] is False
    assert rows[1]["event"] == "command_start"
    assert rows[2]["stdout"].strip() == "fake USB write complete"
    assert rows[-1]["succeeded"] == 1 and rows[-1]["exit_code"] == 0


def test_output_is_bounded_and_missing_controller_is_clear(tmp_path, monkeypatch, capsys):
    stream = io.BytesIO(b"x" * 10000)
    tail, truncated = endurance.output_tail(stream)
    assert truncated and len(tail) == endurance.OUTPUT_LIMIT
    monkeypatch.setattr(endurance, "ROOT", tmp_path)
    assert endurance.main([]) == 2
    assert "controller not found" in capsys.readouterr().err


def test_spawn_failure_is_logged_as_failed(tmp_path):
    result = endurance.execute_command([str(tmp_path / "missing-python")], 3, tmp_path)
    assert result["outcome"] == "failed"
    assert result["termination_confirmed"] is True
    assert result["error_type"] == "FileNotFoundError"
