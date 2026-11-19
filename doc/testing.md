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
| P0-04 | Verify exact yellow/green/red/buzzer commands, duplicate suppression, bounded recovery and independent buzzer failures. | `test_tower.py::test_commands_cache_and_buzzer`, `test_retry_during_hold_without_duplicate_inspection`, `test_buzzer_failure_independent_and_never_replayed` |
| P0-05 | Persist correct inspection counts, JSON errors, review reversals and appended CSV rows. | `test_logs_config.py::test_json_and_error_log_contents`, `test_statistics_counts_and_flush`, `test_rejudge_result_mapping`, `test_rejudge_append_and_fields` |
| P0-06 | Write pending review once; stop workers, release capture, flush statistics and stop recording. | `test_workflow.py::test_pending_review_written_only_once`, `test_stop_flushes_pending_and_releases_resources` |
| P0-K01 | **Known limitation:** only three passing detections can still Pass. | `test_inspection.py::test_missing_fourth_region_can_pass` |
| P0-K02 | **Known limitation:** empty, missing-contour or exceptional crops are skipped; one large crop considers only its largest contour. | `test_inspection.py::test_unanalysed_region_is_skipped`, `test_large_box_analyses_only_largest_contour` |
| P0-K03 | Stage 1 USB lifecycle: truthful exit codes, explicit resource release, retained vendor reset/Windows length rule, timeout, supersession and bounded shutdown. | `test_tower.py::test_usb_cli_false_exit_cleanup_and_optional_reset`, `test_vendor_packet_and_windows_length`, `test_real_child_timeout_and_bounded_stop`, `test_newest_state_supersedes_inflight_and_no_stale_buzzer`, `test_request_responsive_while_worker_busy_and_shutdown_cancels` |
| P1-01 | Configuration defaults, supplied settings, invalid numeric values, versionless default location and automatic parent directory creation. | `test_logs_config.py::test_config_defaults_and_supplied_ini`, `test_invalid_numeric_config_raises`, `test_default_config_uses_versionless_config_folder`, `test_custom_nested_config_creates_parent_directories` |
| P1-02 | Convexity classification and deterministic K-means clustering. This does not measure real-world accuracy. | `test_inspection.py::test_contour_decision`, `test_kmeans_uses_brightest_cluster_values` |
| P1-03 | Initialize with injected model/camera; reject missing weights before device access; keep log writers disabled when configured. | `test_camera_detector.py::test_initialize_with_injected_model_camera_and_no_logs`, `test_missing_model_fails_before_loading_or_opening_camera` |
| P1-04 | Verify USB/UVC failure, file selection and simulated TIS selection; reconnect releases the old capture and honors attempt limits. | `test_camera_detector.py::test_camera_setup_failure_and_invalid_mode`, `test_video_source_uses_configured_file`, `test_tis_source_uses_wrapper_without_loading_dll`, `test_reconnect_releases_old_capture_and_resets_on_success`, `test_reconnect_stops_after_limit` |
| P1-05 | Deliver frames/results, drop when queues are full, stop capture after read-failure threshold; isolate queues/state between instances. | `test_camera_detector.py::test_capture_worker_delivers_or_drops_frame`, `test_capture_failure_stops_after_threshold_without_waiting`, `test_yolo_worker_forwards_or_drops_result`, `test_instances_do_not_share_queues_or_carrier_state` |
| P1-06 | Preserve YOLO coordinates/scores/classes and prediction settings; handle empty/failed inference; test injected reload, repeated CUDA errors and general worker failure limit. | `test_camera_detector.py::test_yolo_outputs_and_prediction_settings`, `test_empty_or_failed_yolo_returns_empty_arrays`, `test_model_reload_uses_injected_factory`, `test_worker_cuda_recovery_threshold`, `test_worker_general_failure_limit` |
| P1-07 | Toggle, confirm, cancel and reopen review; accept keyboard actions; preserve source indices; retain tower state; discard unconfirmed choices on departure. | `test_review_display_recording.py::test_review_toggle_reopen_and_cancel`, `test_review_keyboard`, `test_review_empty_and_invalid_crops`, `test_review_confirm_updates_correct_source_without_changing_light`, `test_open_review_on_departure_resets_agreement_before_write`, `test_review_not_opened_outside_hold` |
| P1-08 | Match boxes and draw NG color; write a readable screenshot; fit displayed frames; use even recording dimensions; release active writers and report writer failure. | `test_review_display_recording.py::test_box_matching_and_frame_colors`, `test_screenshot_and_fit_display`, `test_recording_dimensions`, `test_recording_writer_lifecycle` |
| P1-09 | Run a main-loop iteration and quit; process successive Pass/NG carriers with exact command sequence; stop after entrypoint exceptions; return an independent inspection snapshot. | `test_application.py::test_main_loop_standby_quit_and_cleanup`, `test_two_carrier_sequence_pass_then_ng`, `test_entrypoint_exception_still_stops_processor`, `test_inspection_snapshot_does_not_alias_mutable_state` |
| P2-01 | Emit heartbeat at the configured boundary with required health fields. | `test_review_display_recording.py::test_heartbeat_boundary_and_fields` |
| P2-02 | Both launchers invoke the same entry point; clean-process imports create no files, access no devices and do not import Ultralytics/Torch. | `test_application.py::test_new_and_legacy_launchers_use_same_entrypoint`, `test_import_has_no_external_side_effects` |

## Reproduce the baseline comparison

The original source is preserved in commit `1b341eb`. The first 38 cases were run against the monolith before extraction. They can also be rerun directly from Git history, without replacing working files:

```bash
# macOS / Linux
CUTSEALANT_BASELINE_REF=1b341eb python -m pytest test/test_inspection.py test/test_workflow.py test/test_tower.py test/test_logs_config.py -q
```

```powershell
# Windows PowerShell
$env:CUTSEALANT_BASELINE_REF = "1b341eb"
python -m pytest test/test_inspection.py test/test_workflow.py test/test_tower.py test/test_logs_config.py -q
Remove-Item Env:CUTSEALANT_BASELINE_REF
```

This optional comparison requires Git history; ordinary tests do not. Do not set the baseline variable for module-specific tests. Reference outcomes live in the test assertions (including exact command sequences, contour results, record fields and boundary transitions); only runtime timestamps and session IDs are variable.

## Executed verification

On 2026-11-16, using Python 3.12.14, pytest 9.2.0, NumPy 2.5.5, OpenCV 5.0.0 and PyUSB 1.3.1 on macOS:

- Original baseline characterization suite: **38 passed**.
- The same characterization suite after extraction: **38 passed**.
- Expanded module/integration suite: **81 passed**, including known-issue characterizations; no skipped or expected-failure cases.
- Structural comparison of all 36 extracted processor service methods: no statement changes after normalizing context names, injected dependencies and instance queues. Original independent class bodies are preserved.

The listed versions describe the executed environment, not a claim of validation on every version allowed by the dependency manifests. Full Ultralytics/PyTorch runtime installation was not needed for this hardware-free run.

After moving the active configuration to `config/config.ini`, the expanded suite has **83 passing cases**. The two additional cases verify the new default location, reloading saved settings, and creation of missing configuration directories. The historical baseline counts above describe the initial refactor verification.

## Manual acceptance: not verified by automated tests

| Priority / ID | Procedure | Required result / evidence |
| --- | --- | --- |
| P0-M01 | On Windows with the real model, replay the same labeled video before and after extraction; retain logs and seed OpenCV RNG consistently for comparison. | Same evaluated-region decisions, aggregate outcomes, state/tower command sequence and record counts; ignore timestamps/session IDs. Investigate timing-dependent differences separately. |
| P0-M02 | With a real PATLITE, request yellow, green, red and buzzer. Disconnect/reconnect the device during operation. | Record actual physical responses, subprocess errors and recovery behavior. No automatic recovery is promised by this refactor. |
| P0-M03 | Run production-like tower changes for at least 24 hours and longer than the observed failure interval. | Retain command/physical observations, failure timestamps, memory, handle counts and any recovery/reboot requirements. Do not infer physical state from status_change logs. |
| P1-M01 | Use each deployed camera mode; verify read failure, reconnect and stop. | Correct device, usable frames, documented recovery and resource release. |
| P1-M02 | Operate screenshot, fit, fullscreen, review and recording on the actual desktop. | Correct visible frames, review choices, readable screenshots and playable recordings. |

The repository has no trained `.pt` model. Real detection accuracy, actual camera behavior and Windows USB stability remain unverified until these procedures are completed. Existing bugs remain unchanged by the refactor.
