# CutSealant inspection

Python application for inspecting sealant on cases carried on a carrier. It reads a camera or video, locates the carrier and cases with YOLO, evaluates the detected case crops with OpenCV, and sends commands to a PATLITE USB signal tower. The retrained model's classes are `case` (0), `cover` (1), `base` (2), and `carrier` (3). Each `case` supplies the crop for sealant inspection; `cover` and `base` do not contribute inspection results or position counts. Configure IDs in `[CLASSES]`; older configurations using `sealant` instead of `case` remain supported.

## YOLO inference area

YOLO receives only the crop configured in [`config/inference_roi.json`](config/inference_roi.json):

```json
{
  "version": 1,
  "reference_size": [1344, 756],
  "roi_xyxy": [454, 312, 850, 670]
}
```

`reference_size` is `[width, height]`; `roi_xyxy` is `[left, top, right, bottom]` with exclusive right/bottom bounds. At the reference resolution this is a **396 × 358** crop. Coordinates scale independently with the actual frame width and height: left/top round down, right/bottom round up. The model's normal internal resizing still applies to this crop; the whole frame is never passed to YOLO.

A thick yellow rectangle shows this same area on the full-frame display and annotated recordings. It guides carrier placement. Crop-relative detections are clipped to the crop and translated back to full-frame coordinates before case assignment, sealant judgment, review and drawing. Sealant quality still uses original-resolution pixels; raw recording remains unannotated. The existing camera-alignment ROI and lower-half stop check remain separate checks in full-frame coordinates.

The JSON is loaded beside the active INI at startup; restart after editing it. Missing or invalid crop configuration stops initialization instead of falling back to whole-frame inference. Creating a new default INI also creates the default ROI JSON if it is absent; it never overwrites an existing ROI file.

## How detection works

Each carrier cycle remembers up to four case positions relative to the selected carrier bounding box:

```text
01 top-left      02 top-right
03 bottom-left   04 bottom-right
```

1. Start in yellow Standby. When a carrier arrives, request blinking yellow Tracking.
2. Wait for the carrier to stop in the lower image half and pass the configured camera-alignment ROI check.
3. Discover occupied quadrants throughout the cycle. A detected case center assigns its position; detections outside the carrier are ignored. If several detections occupy the same quadrant, use the highest confidence, then the smaller area.
4. For each unfinished automatic case, require 10 consecutive passing observations. Five consecutive NG observations mark the case as requiring manual check; press **M** to open review. Both thresholds are configurable in `[DETECTION_SETTINGS]`:

   ```ini
   case_pass_consecutive_frames = 10
   case_fail_consecutive_frames = 5
   ```

5. Missing detections, unusable crops, or carrier movement/loss reset unfinished streaks. They do not create a new NG decision. Passed cases remain passed when covered by hands or metal, and are no longer judged.
6. Manual review shows **Position 01–04**, using the saved image that caused the failure. Agreeing with NG confirms NG; disagreeing and confirming immediately produces Manual Pass. Confirmed NG requires another manual check with **M**; automatic passing frames cannot clear it. Cancel, Escape, or closing review leaves the case unresolved. Other positions continue checking while review is open; later failures keep their saved evidence for the next time the operator presses **M**. Review never opens automatically.
7. When at least one case has been discovered and every discovered case passes, request green and show **Ready - remove carrier**. Discovery continues: a newly revealed case withdraws readiness until it passes.
8. Close successfully only when the carrier disappears for `absent_tolerance_frames` processed results (default 5) and readiness has lasted `result_hold_min_seconds` (default 2). Zero visible cases while the carrier remains does not close the cycle.
9. If the carrier disappears before readiness, hold its results and show **Cycle incomplete - restore carrier or press X**. Press **X** to terminate without confirmation, save partial results, clear review and case memory, and return to Standby. The same carrier can start a new cycle after a fresh stability period. Quit/shutdown records an active cycle as Interrupted.

The 10/5 streaks count distinct processed inference results, not display refreshes or a fixed duration. The existing stop estimate uses processed results divided by configured FPS. After movement/loss, a fresh stability period is required before unfinished cases resume inspection. Review and application controls share one keyboard handler, including when inference is idle.

YOLO supplies carrier and sealant locations; it does not classify sealant quality. Each original-resolution crop is converted to HLS, its lightness values are grouped into three clusters, and the largest contour of the brightest cluster is tested with `cv2.isContourConvex()`. A usable crop without a contour is NG; invalid crops and calculation errors are Unavailable and reset the streak. This remains a shape heuristic, not a direct gap-width or continuity measurement. YOLO inference sizing is unchanged.

Before inspection starts, the carrier center must be inside the `[CAMERA_HEALTH]` ROI. By default this is the central 50% of the frame (`x=0.25`, `y=0.25`, `width=0.50`, `height=0.50`). An outside center latches `CAMERA_ERROR`, requests steady yellow, and shows a blinking red warning. Correct alignment and press **C**; a fresh stability period and ROI check are required. **X does not clear this error.** This check infers possible misalignment; it does not measure camera angle.

### Loading assumptions and limits

There is one case per quadrant, the carrier orientation stays consistent, and cases are not replaced during a cycle. Position IDs are quadrant IDs, so a carrier with cases only at 02 and 04 keeps those numbers. Centers exactly on a dividing line belong to the right/bottom quadrant. `expected_case_count` and the older `expected_module_count` are accepted for the legacy aggregate inspection API; active cycles discover occupied positions and do not require four cases.

A case hidden for the entire cycle cannot be known to the system. Carrier absence is camera-based: hiding the carrier itself can resemble departure after readiness. Independent physical departure verification is outside this implementation. One oversized detection still analyzes only its largest contour and represents only its center's quadrant.

## Install and run

Python **3.12** is the development/test baseline. The active configuration is [`config/config.ini`](config/config.ini), with the inference crop in [`config/inference_roi.json`](config/inference_roi.json). Run commands from the repository root: configuration and relative paths are resolved against the current working directory. Paths inside the INI remain relative to the repository root, not the `config/` folder. If the configuration is missing, the application creates the folder and a default INI file.

### Windows PowerShell

```powershell
git clone https://github.com/dreyhsu/NK_PM_CutSealant.git
cd NK_PM_CutSealant
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
# Edit config/config.ini and supply the model/video or camera before starting.
.\.venv\Scripts\python.exe main.py
```

### macOS / Linux (video mode)

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
# Edit config/config.ini and supply the model/video before starting.
.venv/bin/python main.py
```

With the intended Python environment activated, the common launch command is:

```bash
python main.py
```

The existing launcher remains supported:

```bash
python yoloVideoMain_V6.0.7.py
```

### Required local setup

1. Obtain the trained `.pt` model from the original developer. This repository does not include it. Set `[PATHS] model_path` to that file.
2. Choose `[VIDEO_SOURCE] mode`: `video`, `camera`, or `dfk_camera`. Supply `video_path` for video mode or the correct camera settings for a live source.
3. Replace the original developer's absolute model/video paths. For example, use `./models/yolo11s_20251223.pt` and your own video path.
4. Select `[DEVICE_SETTINGS] device` for the host. The checked-in configuration uses `cuda:0`; use `mps` on a supported Mac. For a CPU configuration, use `cpu` and set `[YOLO_SETTINGS] use_half_precision = false`. CUDA requires an appropriate local PyTorch/CUDA installation.
5. TIS industrial-camera mode requires the matching vendor camera software and x64 DLLs, including the configured `tisgrabber_x64.dll`, installed locally. These binaries are excluded from Git. General camera mode currently uses OpenCV DirectShow, so its existing implementation targets Windows.
6. Set `[LIGHT_TOWER] mode = mock` to simulate tower transport without accessing USB. Set `mode = usb` to control the physical PATLITE; this requires PyUSB and the appropriate working USB backend/driver setup for the Windows installation. Preserve a known-working driver setup; installing Python packages alone does not establish it. USB commands run in serialized background child processes using the application's Python interpreter. Activate the environment when using the physical tower:

```powershell
.\.venv\Scripts\Activate.ps1
python main.py
```

Tkinter is used for screen sizing. Use a Python installation with Tk support; on Linux it may require the OS package for Tk. Optional memory metrics can be enabled with `python -m pip install psutil`.

The only `.exe` in the original delivery is a camera installer; it is not a compiled inspection application containing the model. No EXE, DLL, model weights, input videos or runtime output are committed. Existing small requirement PDFs and the UI mockup are retained.

## What each script does

| Script / module | Responsibility |
| --- | --- |
| `main.py` | Recommended application launcher. |
| `yoloVideoMain_V6.0.7.py` | Compatibility launcher; re-exports the original public classes. |
| `cutsealant/app.py` | Creates the components, owns bounded queues, runs the main/UI loop and performs shutdown. |
| `cutsealant/config.py` | Reads the existing INI file and default values. |
| `cutsealant/camera.py` | Opens the source, captures frames and attempts camera reconnection. |
| `cutsealant/detector.py` | Lazily loads YOLO, predicts object boxes/classes/confidence and runs the inference worker. |
| `cutsealant/inspection.py` | Evaluates crops using lightness clustering and contour convexity. |
| `cutsealant/workflow.py` | Owns carrier state and controls inspection triggers and result hold. |
| `cutsealant/tower.py` | Schedules serialized PATLITE processes, timeouts, LED recovery and independent buzzer outcomes. |
| `cutsealant/display.py` | Draws overlays, matches boxes, displays frames and handles screenshots/window sizing. |
| `cutsealant/review.py` | Displays sealant crops, handles agreement choices and writes pending review records. |
| `cutsealant/recording.py` | Records raw or annotated frames. |
| `cutsealant/logs.py` | Writes system JSON, error text, inspection statistics and review CSV. |
| `cutsealant/monitoring.py` | Reports heartbeat/statistics and saves validation images. |
| `LR6main.py` | Existing low-level PATLITE USB command-line controller. |
| `tis_camera_wrapper.py` | Adapts a TIS camera to an OpenCV-style interface. |
| `TIS_API.py`, `tisgrabber.py` | Existing vendor camera SDK bindings. |

There is one application, not one process per module. Background capture and inference communicate through queues of capacity two; full queues drop new items after the existing write timeout. All GUI and carrier-state decisions stay on the main thread.

`SafeVideoProcessor` retains its compatibility methods. It composes services rather than inheriting mixins. Carrier fields delegate to `workflow.state`; capture/model handles belong to the camera/detector services. Services receive an explicit application context. The constructor accepts `model_factory`, `capture_factory`, `command_runner`, and `clock` for isolated tests; importing modules does not open devices or load a model.

## Controls and outputs

| Key | Action |
| --- | --- |
| `Q` / `q` | Quit; record an active cycle as Interrupted. |
| Close main window | Quit and stop tower retries. |
| `T` / `t` | Retry the latest desired tower state after recovery exhaustion. |
| `S` / `s` | Save screenshot. |
| `F` / `f` | Toggle fit-to-screen. |
| `G` / `g` | Toggle fullscreen. |
| `P` / `p` | Print and log statistics. |
| `r` | Reset processing counters. |
| `V` / `v` | Toggle video recording. |
| `M` / `m` | Reopen manual check for unresolved failed positions. |
| `X` / `x` | Terminate the cycle; restart after a fresh carrier stability period. |
| `C` / `c` | Reset a latched camera alignment error after correcting alignment. |
| Enter / ESC | Confirm / cancel in the review window. |

Review borders: green = agree with Pass, red = agree with NG, yellow = disagree. Confirmation stores choices for the rejudge CSV. Cancellation restores agreement with the original result. Confirmed manual Pass changes the case result and can make the carrier ready. Confirmed NG remains unresolved until manual recheck. Pending edits do not change a saved decision.

With the supplied paths, outputs are:

| Location | Contents |
| --- | --- |
| `logs/detection.log` | JSON lines: configuration, state changes, inspection results, health and review events. |
| `logs/detection_errors.log` | Errors explicitly reported by the application logger. |
| `logs/statistics_<session>.csv` | Completed cycles only: final case counts, cycle time and closing summary. |
| `logs/cycles_<session>.jsonl` | One terminal Completed, Terminated or Interrupted record per cycle, with per-position state and decision source. |
| `logs/rejudge/rejudge_YYYYMMDD.csv` | Per-sealant AI result, agreement, confidence and final `Pass`, `NG`, `REPASS` or `REFAIL`. |
| `validation/` | Legacy validation-image helper output; active cycles keep evidence crops in memory for review. |
| `screenshots/`, `recordings/` | User screenshots and optional recordings. |

Setting `save_detection_log = false` also disables creation of statistics and rejudge writers. Cycle audit records remain enabled in the configured log directory. Review CSV `sealant_index` is the zero-based quadrant index (0–3); display labels are 01–04. Partial cases are recorded as Pending or Manual check required, never as Pass based on a short streak. Heartbeats are emitted every 60 seconds. Missing frames produce warnings; video end does not automatically exit the main loop in the current monitoring implementation.

## Tests

Full development environment:

```bash
python -m pip install -r requirements-dev.txt
python -m pytest test/ -q
```

For model-free tests without installing Ultralytics/PyTorch:

```bash
python -m pip install -r requirements-test.txt
python -m pytest test/ -q
```

The suite uses real OpenCV calculations with simulated camera, YOLO, GUI and USB interfaces. It does not download a model or operate devices. See [ranked test conditions and manual acceptance procedures](doc/testing.md). Tests marked `known_issue` deliberately characterize existing limitations; they are not production acceptance criteria.

The original extraction was checked against 38 historical characterization cases. The active workflow now intentionally differs from that baseline; the current suite covers position memory, streaks, manual decisions, lifecycle records, module boundaries and simulated integration flows. Real-model accuracy and Windows hardware stability require the manual procedures.

## Tower recovery and remaining hardware checks

Tower commands run outside the UI thread, with a 3-second command timeout and
5 total attempts per recovery batch by default. A newer requested color replaces
queued work. Successful LED states refresh every 30 seconds, including during
result hold; recovery never triggers another inspection/statistics row. After
recovery exhaustion, `USB: UNAVAILABLE` remains visible. Press **T** or request
a different state to begin another batch.

The overlay distinguishes `REQUEST` from USB transport health. Successful writes
do not physically confirm the lamp. Buzzer delivery is tracked separately and
attempted at most once per carrier; ambiguous/failed delivery shows `BUZZER: ERROR`
without automatically sounding it again. Tower events always append to
`./logs/tower.jsonl` (configurable), even with detection logging disabled.

Stage 1 keeps the official per-command `set_configuration()` and USB reset.
Resources are explicitly released on completion/failure. `reset_after_command = false`
enables controlled no-reset hardware testing; a persistent USB session remains
future work. Closing OpenCV or pressing Q cancels recovery and releases the
controller. Shutdown preserves the existing policy of sending no all-off command.

Windows driver/hardware recovery and 24-hour stability require real-device
validation. See [implementation status and remaining tower work](doc/light_tower_pending_changes.md).
