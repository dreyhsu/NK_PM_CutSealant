import csv
import json
from pathlib import Path

import numpy as np
import pytest


def test_config_defaults_and_supplied_ini(legacy, tmp_path):
    config = legacy.ConfigManager(str(tmp_path / "new.ini"))
    assert config.get_device() == "cpu"
    assert config.get_classes() == {"carrier": 0, "sealant": 1}
    root = Path(__file__).resolve().parents[1]
    supplied = legacy.ConfigManager(str(root / "config" / "config.ini"))
    assert supplied.get_device() == "cuda:0"
    assert supplied.get_detection_config()["absent_tolerance_frames"] == 5
    assert supplied.get_rejudge_config()["review_window_padding"] == 50
    assert supplied.get_light_tower_config()['mode'] == 'mock'


def test_light_tower_config_defaults_to_usb_for_older_ini(tmp_path):
    from cutsealant.config import ConfigManager

    path = tmp_path / "older.ini"
    path.write_text("[DEVICE_SETTINGS]\ndevice = cpu\n")
    assert ConfigManager(path).get_light_tower_config()['mode'] == 'usb'


def test_invalid_numeric_config_raises(legacy, tmp_path):
    config = legacy.ConfigManager(str(tmp_path / "bad.ini"))
    config.config["DETECTION_SETTINGS"]["fps"] = "invalid"
    with pytest.raises(ValueError):
        config.get_detection_config()


def test_default_config_uses_versionless_config_folder(tmp_path):
    from cutsealant.config import ConfigManager

    config = ConfigManager()
    path = tmp_path / "config" / "config.ini"
    assert config.config_file == "config/config.ini"
    assert path.is_file()
    assert not (tmp_path / "config_v6.0.ini").exists()
    config.config["DEVICE_SETTINGS"]["device"] = "test_device"
    with path.open("w") as f:
        config.config.write(f)
    assert ConfigManager().get_device() == "test_device"


def test_custom_nested_config_creates_parent_directories(tmp_path):
    from cutsealant.config import ConfigManager

    path = tmp_path / "custom" / "nested" / "settings.ini"
    config = ConfigManager(path)
    assert path.is_file()
    assert config.get_classes() == {"carrier": 0, "sealant": 1}


def test_json_and_error_log_contents(legacy, tmp_path):
    path = tmp_path / "logs" / "detection.log"
    logger = legacy.JSONLogger(str(path))
    try:
        logger.log_state_change("WAITING", "DETECTED", "test")
        logger.log_detection_result(4, 3, 1, [{"sealant_id": 1}], {"source": "fake"})
        logger.log_error("test_failure", "simulated", {"status": "Pass"})
        logger.close()
        rows = [json.loads(s) for s in path.read_text().splitlines()]
        assert rows[0]["event_type"] == "session_start"
        assert rows[-1]["event_type"] == "session_end"
        assert all(r["timestamp"] and r["session_id"] for r in rows)
        result = next(r for r in rows if r["log_type"] == "detection_result")
        assert result["summary"] == {"total_sealants": 4, "ok_count": 3, "ng_count": 1, "overall_result": "NG"}
        assert "simulated" in path.with_name("detection_errors.log").read_text()
    finally:
        for handler in logger.logger.handlers[:]:
            handler.close()
            logger.logger.removeHandler(handler)


def test_json_logger_serializes_numpy_values(legacy, tmp_path):
    path = tmp_path / "logs" / "detection.log"
    logger = legacy.JSONLogger(str(path))
    try:
        logger.log_detection_result(
            np.int32(4),
            np.int64(3),
            np.int32(1),
            [{"is_broken": np.bool_(True), "score": np.float32(0.75)}],
            {
                "carrier_centroid": [np.int32(120), np.int32(340)],
                "box": np.array([1, 2, 3, 4], dtype=np.int32),
            },
        )
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        result = next(row for row in rows if row["log_type"] == "detection_result")
        assert result["summary"] == {
            "total_sealants": 4,
            "ok_count": 3,
            "ng_count": 1,
            "overall_result": "NG",
        }
        assert result["detection_details"] == [
            {"is_broken": True, "score": 0.75}
        ]
        assert result["frame_info"] == {
            "carrier_centroid": [120, 340],
            "box": [1, 2, 3, 4],
        }
    finally:
        for handler in logger.logger.handlers[:]:
            handler.close()
            logger.logger.removeHandler(handler)


def test_statistics_counts_and_flush(legacy, tmp_path):
    logger = legacy.StatisticsLogger(str(tmp_path), "session")
    logger.log_detection(False, 4, 0, 4, 23)
    logger.log_detection(True, 3, 1, 4, 25)
    rows = list(csv.DictReader(Path(logger.csv_file_path).open()))
    assert [r["result"] for r in rows] == ["OK", "NG"]
    assert rows[0]["cycle_time_ms"] == "23"
    assert logger.get_summary()["yield_rate_percent"] == 87.5
    logger.close()
    assert "# Summary:" in Path(logger.csv_file_path).read_text()


@pytest.mark.parametrize("ai,agrees,expected", [("Pass", True, "Pass"), ("NG", True, "NG"), ("Pass", False, "REFAIL"), ("NG", False, "REPASS")])
def test_rejudge_result_mapping(legacy, tmp_path, ai, agrees, expected):
    logger = legacy.RejudgeLogger(str(tmp_path), "session")
    assert logger._calculate_final_result(ai, agrees) == expected


def test_rejudge_append_and_fields(legacy, tmp_path):
    logger = legacy.RejudgeLogger(str(tmp_path), "session")
    pending = {"carrier_id": 7, "frame_number": 42, "user_reviewed": True,
               "sealants": [{"index": 0, "bbox": [1, 2, 3, 4], "ai_result": "NG", "confidence": .923456, "user_agrees": False}]}
    logger.write_records(pending, "fake_camera")
    logger.write_records({**pending, "carrier_id": 8}, "fake_camera")
    with Path(logger._get_csv_path()).open() as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 2
    assert [r["carrier_id"] for r in rows] == ["7", "8"]
    assert rows[0]["final_result"] == "REPASS"
    assert rows[0]["confidence"] == "0.9235"
    assert rows[0]["bbox_w"] == "3"
    assert rows[0]["frame_number"] == "42"


@pytest.mark.parametrize('setting', ['case_pass_consecutive_frames', 'case_fail_consecutive_frames'])
@pytest.mark.parametrize('value', ['0', '-1', '1.5', 'invalid'])
def test_case_streak_settings_require_positive_integers(tmp_path, setting, value):
    from cutsealant.config import ConfigManager
    config = ConfigManager(tmp_path / 'config.ini')
    config.config['DETECTION_SETTINGS'][setting] = value
    with pytest.raises(ValueError):
        config.get_detection_config()


def test_case_streak_settings_default_for_old_ini(tmp_path):
    from cutsealant.config import ConfigManager
    config = ConfigManager(tmp_path / 'config.ini')
    for key, expected in [('case_pass_consecutive_frames', 10), ('case_fail_consecutive_frames', 5)]:
        del config.config['DETECTION_SETTINGS'][key]
        assert config.get_detection_config()[key] == expected
