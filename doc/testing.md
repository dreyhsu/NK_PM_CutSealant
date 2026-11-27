# Tests and production acceptance

Run the active suite from the repository root:

```bash
python -m pip install -r requirements-test.txt
python -m pytest test/ -q
```

Tests use real NumPy/OpenCV image calculations with simulated model, capture, USB and GUI interfaces. Fixtures reject unmocked device access and subprocess calls and isolate generated files in temporary directories. No trained model or hardware is required.

## Position-cycle acceptance

| Behavior | Automated coverage |
| --- | --- |
| Crop-only model input at reference/scaled resolutions; original-coordinate box restoration/clipping; unchanged raw frame; thick yellow ROI overlay; invalid/missing ROI prevents inference | `test_inference_roi.py`, `test_camera_detector.py` |
| One to four remembered quadrants; arbitrary discovery order, relative motion/jitter, center boundaries, confidence/area duplicate ranking, outside boxes | `test_workflow.py` |
| Nine Pass observations remain pending; tenth passes; missing/Unavailable/NG/movement reset streaks; five NG observations request review; custom thresholds | `test_workflow.py`, `test_logs_config.py` |
| Passed cases survive covering and skip further judgment; new positions withdraw readiness; zero cases cannot pass | `test_workflow.py` |
| Manual Pass changes readiness; confirmed NG requires M; cancel/close remains unresolved; NG never opens review automatically; later failures keep independent evidence until M; X wins over unconfirmed edits | `test_review_display_recording.py` |
| One keyboard pump handles both windows and inference-idle periods; X discards queued results; camera reset still requires fresh stability | `test_application.py`, `test_camera_health.py` |
| Ready departure requires absence tolerance and hold time; early departure holds; same carrier can restart after termination; shutdown is Interrupted | `test_workflow.py` |
| Exactly one terminal audit and review write; production statistics only for completed cycles; logging errors do not block termination | `test_workflow.py`, `test_application.py`, `test_logs_config.py` |
| Four position statuses and workflow/tower overlays; actual original-resolution crop observations; invalid images are Unavailable | `test_workflow_panel.py`, `test_inspection.py` |

Other modules retain coverage for camera setup/reconnection, injected model loading, YOLO prediction arguments and worker failures, recording/screenshots, resource cleanup, heartbeat logging, tower recovery, bounded USB shutdown, and low-level PATLITE protocol handling. Tests of the legacy aggregate inspection API retain its old count requirement; that API is not the active position-cycle decision path.

The original extraction was checked against 38 historical characterization cases. Those historical results do not establish equivalence with the new workflow. Current assertions intentionally reflect position memory and manual decisions rather than immediate carrier-wide Pass/NG.

## Manual acceptance with the production camera and model

1. Verify that the thick yellow outline matches the configured inference crop at the deployed resolution; place the carrier inside it. Load carriers with one through four cases in different quadrants. Verify IDs 01 top-left, 02 top-right, 03 bottom-left and 04 bottom-right, including a partially loaded carrier with only 02/04.
2. Observe progress to 10 passing processed results. Briefly block an unfinished case: its counter resets, while already passed cases remain passed. Move the carrier and verify a fresh stability interval before unfinished checks resume.
3. Produce five consecutive evaluated failures. Verify the case is marked for manual check without opening a window. Press M and verify that the popup shows the correct position and its failure image. Confirm NG, correct the case, and verify automatic results cannot clear it; press M and manually approve it. Cancel and close the popup separately, confirming neither action passes the case or repeatedly reopens it.
4. Fail another case while the first popup remains open. Confirm or cancel the first review and verify that no new popup opens. Press M again and verify it contains the later case's own evidence. Check keyboard and mouse confirmation and fullscreen restoration.
5. Cover passing cases with metal one by one until none are detected. Verify green remains if every discovered case passed and the carrier is still visible. Reveal an additional case: readiness must be withdrawn until it passes.
6. Remove a ready carrier: after absence/hold thresholds, verify one Completed audit, final per-position review records and one production statistics row. Remove an unfinished carrier: verify its cycle remains held until restored or terminated.
7. Press X from the main window and from review, including with no new inference frames. Verify immediate Terminated outcome, cleared review/evidence, yellow standby, and a new stability period before the same carrier starts again. X must not clear CAMERA_ERROR. Quit with an active cycle and verify one Interrupted audit.
8. Verify that popup text, position labels, counters and keyboard hints are readable at the deployed screen scale. Replay representative videos with real occlusion and retain cycle/review logs for threshold tuning.
9. With the actual PATLITE, verify yellow blink/steady, green, red and the once-per-carrier buzzer, USB disconnect/reconnect, retry, shutdown, and a 24-hour production-like stability run. Check physical lamp behavior independently of software transport success.

The model is external to this repository. Real detection accuracy, actual camera/GUI behavior and Windows USB reliability are not established by simulated tests. Cases hidden for an entire cycle cannot be discovered. Hiding the carrier can resemble departure after readiness. The contour algorithm remains a convexity heuristic and does not directly measure sealant gaps.
