import subprocess
from unittest.mock import Mock

import pytest


@pytest.mark.parametrize("status,args", [
    ("Standby", ["2", "0", "1", "0", "0", "0"]),
    ("Tracking", ["2", "0", "2", "0", "0", "0"]),
    ("Pass", ["2", "0", "0", "1", "0", "0"]),
    ("NG", ["2", "1", "0", "0", "0", "0"]),
])
def test_tower_command_and_cache(processor, monkeypatch, status, args):
    send = Mock(return_value=0)
    monkeypatch.setattr(subprocess, "call", send)
    assert processor._call_light_once(status) == status
    assert send.call_args_list[0].args[0] == ["python", "LR6main.py", *args]
    count = 2 if status == "NG" else 1
    assert send.call_count == count
    if status == "NG":
        assert send.call_args.args[0] == ["python", "LR6main.py", "4", "3", "1", "1", "14"]
    processor.logger.log_system_event.assert_called_once_with("status_change", {"new_status": status})
    processor._call_light_once(status)
    assert send.call_count == count


def test_launch_exception_logged_and_cache_not_updated(processor, monkeypatch):
    monkeypatch.setattr(subprocess, "call", Mock(side_effect=OSError("cannot launch")))
    processor.last_light_status = "Standby"
    assert processor._call_light_once("Pass") == "Standby"
    assert processor.last_light_status == "Standby"
    assert processor.logger.log_error.call_args.args[0] == "light_control_error"


@pytest.mark.parametrize("status", ["Standby", "Tracking", "Pass", "NG"])
def test_mock_tower_prints_status_without_usb_command(processor, capsys, status):
    processor.light_tower_config = {"mode": "mock"}
    processor.command_runner = Mock(side_effect=AssertionError("USB command must not run"))

    assert processor._call_light_once(status) == status
    assert processor.last_light_status == status
    assert f"[Mock light tower] Status: {status}" in capsys.readouterr().out
    processor.command_runner.assert_not_called()
    processor.logger.log_system_event.assert_called_once_with(
        "status_change", {"new_status": status}
    )


@pytest.mark.known_issue
def test_nonzero_exit_only_printed(processor, monkeypatch, capsys):
    monkeypatch.setattr(subprocess, "call", Mock(return_value=1))
    assert processor._call_light("Pass") is None
    processor.logger.log_error.assert_not_called()
    processor.logger.log_system_event.assert_not_called()
    assert "failed with code" in capsys.readouterr().out


@pytest.mark.known_issue
def test_buzzer_failure_still_logs_ng(processor, monkeypatch):
    monkeypatch.setattr(subprocess, "call", Mock(side_effect=[0, 1]))
    assert processor._call_light_once("NG") == "NG"
    processor.logger.log_system_event.assert_called_once_with("status_change", {"new_status": "NG"})


@pytest.mark.known_issue
def test_usb_child_preserves_reset_and_ignored_false(monkeypatch):
    import LR6main as child
    dev = Mock()
    dev.write.return_value = 0
    assert child.send_command(dev, b"12345678") is False
    dev.reset.assert_called_once()
    monkeypatch.setattr(child, "usb_open", Mock(return_value=dev))
    monkeypatch.setattr(child, "set_tower", Mock(return_value=False))
    monkeypatch.setattr(child.sys, "argv", ["LR6main.py", "2", "0", "0", "1", "0", "0"])
    assert child.main() is None  # ignored False: no failing process exit status
