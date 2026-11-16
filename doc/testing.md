# Test plan and results

Run the hardware-free suite from the repository root:

```bash
python -m pip install -r requirements-test.txt
python -m pytest test/ -q
```

Tests use real NumPy/OpenCV image operations and fake model, camera, USB and GUI interfaces. Automatic fixtures reject unmocked device access and subprocess calls, close GUI operations, and run each test in a temporary working directory. No model download or physical equipment is required.

## Ranked conditions (highest importance first)

These IDs identify test groups; parameterized functions cover multiple cases. Passing characterization tests document existing behavior, including defects. They do not certify that behavior as a safe production rule.

| Priority / ID | Purpose and expected behavior | Automated tests |
| --- | --- | --- |
| P0-01 | All four evaluated regions pass; one failing region makes the entire carrier NG. | `test_inspection.py::test_region_aggregation` |
| P0-02 | Inspect once, only after stationary counter exceeds its threshold; do not trigger for motion or the upper image half. | `test_workflow.py::test_trigger_boundary_and_one_inspection`, `test_motion_and_upper_half_do_not_trigger` |
| P0-03 | Hold results through flicker; reset only when both absence and hold thresholds are met; clear absence on reappearance. | `test_workflow.py::test_absence_and_hold_boundaries`, `test_reappearance_clears_absence_counter` |
| P0-04 | Verify exact yellow/green/red/buzzer commands, duplicate suppression and launch-exception reporting. | `test_tower.py::test_tower_command_and_cache`, `test_launch_exception_logged_and_cache_not_updated` |
| P0-05 | Persist correct inspection counts, JSON errors, review reversals and appended CSV rows. | `test_logs_config.py::test_json_and_error_log_contents`, `test_statistics_counts_and_flush`, `test_rejudge_result_mapping`, `test_rejudge_append_and_fields` |
| P0-06 | Write pending review once; stop workers, release capture, flush statistics and stop recording. | `test_workflow.py::test_pending_review_written_only_once`, `test_stop_flushes_pending_and_releases_resources` |
| P0-K01 | **Known limitation:** only three passing detections can still Pass. | `test_inspection.py::test_missing_fourth_region_can_pass` |
| P0-K02 | **Known limitation:** empty, missing-contour or exceptional crops are skipped; one large crop considers only its largest contour. | `test_inspection.py::test_unanalysed_region_is_skipped`, `test_large_box_analyses_only_largest_contour` |
| P0-K03 | **Known limitation:** nonzero tower exit is console-only; buzzer exit ignored; USB child resets after writes and ignores command False. | `test_tower.py::test_nonzero_exit_only_printed`, `test_buzzer_failure_still_logs_ng`, `test_usb_child_preserves_reset_and_ignored_false` |
| P1-01 | Configuration defaults, actual supplied settings and invalid numeric values. | `test_logs_config.py::test_config_defaults_and_supplied_ini`, `test_invalid_numeric_config_raises` |
| P1-02 | Convexity classification and deterministic K-means clustering. This does not measure real-world accuracy. | `test_inspection.py::test_contour_decision`, `test_kmeans_uses_brightest_cluster_values` |

## Manual acceptance: not verified by automated tests

| Priority / ID | Procedure | Required result / evidence |
| --- | --- | --- |
| P0-M01 | On Windows with the real model, replay the same labeled video before and after extraction; retain logs and seed OpenCV RNG consistently for comparison. | Same evaluated-region decisions, aggregate outcomes, state/tower command sequence and record counts; ignore timestamps/session IDs. Investigate timing-dependent differences separately. |
| P0-M02 | With a real PATLITE, request yellow, green, red and buzzer. Disconnect/reconnect the device during operation. | Record actual physical responses, subprocess errors and recovery behavior. No automatic recovery is promised by this refactor. |
| P0-M03 | Run production-like tower changes for at least 24 hours and longer than the observed failure interval. | Retain command/physical observations, failure timestamps, memory, handle counts and any recovery/reboot requirements. Do not infer physical state from status_change logs. |
| P1-M01 | Use each deployed camera mode; verify read failure, reconnect and stop. | Correct device, usable frames, documented recovery and resource release. |
| P1-M02 | Operate screenshot, fit, fullscreen, review and recording on the actual desktop. | Correct visible frames, review choices, readable screenshots and playable recordings. |

The repository has no trained `.pt` model. Real detection accuracy, actual camera behavior and Windows USB stability remain unverified until these procedures are completed. Existing bugs remain unchanged by the refactor.
