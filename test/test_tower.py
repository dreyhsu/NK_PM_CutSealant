import json
import subprocess
import sys
import threading
import time
from unittest.mock import Mock

import pytest

from cutsealant.tower import TowerController


@pytest.mark.parametrize('status', ['Standby', 'Tracking', 'Pass', 'NG'])
def test_commands_cache_and_buzzer(processor, status):
    processor.command_runner = Mock(return_value=0)
    tower = processor.tower
    tower._call_light_once(status)
    assert tower.last_status is None
    tower.service_once()
    assert tower.last_status == status
    assert processor.command_runner.call_args_list[0].args[0] == [
        sys.executable, 'LR6main.py', *tower.COMMANDS[status]]
    tower._call_light_once(status)
    tower.service_once()
    assert processor.command_runner.call_count == (2 if status == 'NG' else 1)


def test_retry_during_hold_without_duplicate_inspection(processor):
    now = [0.0]
    tower = TowerController(processor, monotonic=lambda: now[0])
    processor.command_runner = Mock(side_effect=[1, 0])
    processor.detection_state = 'RESULT_HOLD'
    processor.carrier_count = 7
    tower._call_light_once('Pass')
    tower.service_once()
    assert tower.last_status is None and tower.health == 'retrying'
    tower.service_once()
    assert processor.command_runner.call_count == 1
    now[0] = 1
    tower.service_once()
    assert tower.last_status == 'Pass' and tower.health == 'connected'
    assert processor.carrier_count == 7 and processor.detection_state == 'RESULT_HOLD'
    rows = [json.loads(line) for line in open('logs/tower.jsonl')]
    events = [row['event_type'] for row in rows]
    assert 'tower_command_failed' in events and 'tower_reconnected' in events
    assert len({row['command_id'] for row in rows if 'command_id' in row}) == 2


def test_newest_state_supersedes_inflight_and_no_stale_buzzer(processor):
    tower = processor.tower
    def run(args, **kwargs):
        tower._call_light_once('Pass')
        return 0
    processor.command_runner = Mock(side_effect=run)
    tower._call_light_once('NG')
    tower.service_once()
    assert tower.last_status is None
    tower.service_once()
    assert tower.last_status == 'Pass'
    assert processor.command_runner.call_count == 2


def test_buzzer_failure_independent_and_never_replayed(processor):
    tower = processor.tower
    processor.command_runner = Mock(side_effect=[0, 1, 0, 0])
    tower._call_light_once('NG')
    tower.service_once()
    assert tower.last_status == 'NG'
    assert tower.buzzer_outcome == 'failed_or_unknown'
    tower.retry_latest()
    tower.service_once()
    tower._call_light_once('Pass')
    tower._call_light_once('NG')
    tower.service_once()
    assert processor.command_runner.call_count == 4
    assert sum(call.args[0][2] == '4' for call in processor.command_runner.call_args_list) == 1


def test_backoff_exhaustion_and_operator_retry(processor):
    now = [0.0]
    tower = TowerController(processor, lambda: now[0])
    processor.light_tower_config.update(max_attempts=2)
    processor.command_runner = Mock(return_value=1)
    tower._call_light_once('Standby')
    tower.service_once()
    now[0] = 1
    tower.service_once()
    assert tower.health == 'unavailable'
    now[0] = 100
    tower.service_once()
    assert processor.command_runner.call_count == 2
    tower.retry_latest()
    processor.command_runner.return_value = 0
    tower.service_once()
    assert tower.last_status == 'Standby'


def test_periodic_refresh_restores_same_color(processor):
    now = [0.0]
    tower = TowerController(processor, lambda: now[0])
    processor.command_runner = Mock(side_effect=[0, 1, 0])
    tower._call_light_once('Pass')
    tower.service_once()
    now[0] = 30
    tower.service_once()
    assert tower.last_status is None
    now[0] = 31
    tower.service_once()
    assert tower.last_status == 'Pass'


def test_mock_never_launches_usb_and_logs_without_detection_logger(processor):
    processor.logger = None
    processor.light_tower_config = {'mode': 'mock'}
    processor.command_runner = Mock(side_effect=AssertionError('no USB'))
    processor.tower._call_light_once('NG')
    processor.tower.service_once()
    assert processor.tower.health == 'mock'
    processor.command_runner.assert_not_called()
    assert 'tower_command_completed' in open('logs/tower.jsonl').read()


def test_real_child_timeout_and_bounded_stop(processor, tmp_path):
    # A harmless Python child simulates a hung controller, without USB access.
    script = tmp_path / 'hang.py'
    script.write_text('import time\ntime.sleep(60)\n')
    processor.paths['stack_light_path'] = str(script)
    processor.light_tower_config['command_timeout_seconds'] = .1
    tower = processor.tower
    tower._call_light_once('Pass')
    started = time.monotonic()
    tower.service_once()
    assert time.monotonic() - started < 2
    assert tower.health == 'retrying'
    assert 'tower_command_timeout' in open('logs/tower.jsonl').read()
    tower.retry_latest()
    tower.start()
    tower.stop()
    tower.stop()
    assert not tower._thread.is_alive()


def test_usb_cli_false_exit_cleanup_and_optional_reset(monkeypatch):
    import LR6main as child
    dev = Mock()
    dev.write.return_value = 0
    monkeypatch.setattr(child, 'usb_open', Mock(return_value=dev))
    dispose = Mock()
    monkeypatch.setattr(child.usb.util, 'dispose_resources', dispose)
    monkeypatch.setattr(sys, 'argv', ['LR6main.py', '2', '0', '0', '1', '0', '0', '--no-reset'])
    assert child.main() == 3
    dev.reset.assert_not_called()
    dispose.assert_called_once_with(dev)


@pytest.mark.parametrize('platform,raw', [('linux', 8), ('win32', 9)])
def test_vendor_packet_and_windows_length(monkeypatch, platform, raw):
    import LR6main as child
    dev = Mock()
    dev.write.return_value = raw
    monkeypatch.setattr(sys, 'platform', platform)
    assert child.set_tower(dev, 0, 0, 1, 0, 0)
    assert dev.write.call_args.args[1] == bytes([0, 0, 15, 0, 0, 16, 0, 0])
    dev.reset.assert_called_once()


def test_open_failure_releases_resources(monkeypatch):
    import LR6main as child
    dev = Mock()
    dev.set_configuration.side_effect = RuntimeError('configuration failed')
    monkeypatch.setattr(child.usb.core, 'find', Mock(return_value=dev))
    dispose = Mock()
    monkeypatch.setattr(child.usb.util, 'dispose_resources', dispose)
    with pytest.raises(RuntimeError):
        child.usb_open()
    dispose.assert_called_once_with(dev)


def test_request_responsive_while_worker_busy_and_shutdown_cancels(processor, monkeypatch):
    active = threading.Event()
    child = Mock()
    child.poll.return_value = None
    def launch(*args, **kwargs):
        active.set()
        return child
    monkeypatch.setattr(subprocess, 'Popen', launch)
    tower = processor.tower
    tower._call_light_once('NG')
    tower.start()
    assert active.wait(timeout=1)
    started = time.monotonic()
    tower._call_light_once('Pass')
    assert time.monotonic() - started < .1
    tower.stop()
    assert not tower._thread.is_alive()
    child.kill.assert_called_once()
    child.wait.assert_called_once_with(timeout=.5)


def test_invalid_cli_and_usb_exception_do_not_report_success(monkeypatch):
    import LR6main as child
    opened = Mock()
    monkeypatch.setattr(child, 'usb_open', opened)
    monkeypatch.setattr(sys, 'argv', ['LR6main.py', '2', '16'])
    with pytest.raises(SystemExit) as exc:
        child.main()
    assert exc.value.code == 2
    opened.assert_not_called()
    dev = Mock()
    opened.return_value = dev
    dev.write.side_effect = RuntimeError('USB stalled')
    dispose = Mock()
    monkeypatch.setattr(child.usb.util, 'dispose_resources', dispose)
    monkeypatch.setattr(sys, 'argv', ['LR6main.py', '2', '0', '0', '1', '0', '0'])
    assert child.main() == 4
    dispose.assert_called_once_with(dev)
