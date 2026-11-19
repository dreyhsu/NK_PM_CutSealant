# CutSealant inspection

Python application for inspecting sealant on EPS power-module carriers. It reads a camera or video, locates the carrier and sealant with YOLO, evaluates the detected sealant crops with OpenCV, and sends commands to a PATLITE USB signal tower.

## How detection works

```text
Start: YELLOW standby
         |
YOLO finds a carrier and sealant boxes
         |
Carrier center is in the lower half of the image
and its position stays stable for about one second
         |
Continuously analyze all four detected sealant crops while the board is present
         |
Any evaluated crop fails? ---- Yes ---> NG: RED + buzzer
         |
         No
         |
    Pass: GREEN
         |
Keep the result live; red can change to green when all four modules pass
Optionally press M for manual review
         |
Carrier or sealant class disappears for at least 5 processed frames
AND result has been held for at least 2 seconds
         |
Save review records; return to YELLOW for the next carrier
```

YOLO locates objects; it does **not** directly classify sealant quality in this program. For each crop, OpenCV groups lightness values into three clusters, selects the brightest cluster, finds its largest external contour, and tests the approximated contour for convexity. A failed evaluated contour makes the entire carrier NG.

The one-second stability estimate uses processed frames divided by configured FPS. YOLO and quality analysis continue while the carrier remains present, so a temporary red result caused by an obstruction can recover to green when all four modules are detected and pass together. Thresholds are configurable.

### Four inspection locations: current limitations

The application now requires exactly four detected regions for Pass, but it does not enforce fixed physical positions:

| YOLO output | Current behavior |
| --- | --- |
| Four separate sealant boxes | Analyze each box; any evaluated failure makes the carrier NG. |
| Three passing boxes | NG; all four configured module regions are required for Pass. |
| One large box containing four sealants | Analyze only the largest contour within that crop, not four separate locations. |
| Empty crop, missing contour or caught crop error | NG; an unanalysed module cannot contribute to Pass. |
| Carrier with no sealant class | Do not trigger a new inspection; after absence tolerance, return to/remain in standby. |

The function historically named “closed loop” actually uses `cv2.isContourConvex()`. This is a shape heuristic, not a direct gap-width or continuity measurement. Sealant boxes are not filtered for containment inside the selected carrier. Review labels follow detector order, not fixed physical positions.

This refactor preserves these rules. Production rules requiring four valid locations must be implemented as a separate change.

## Install and run

Python **3.12** is the development/test baseline. The active configuration is [`config/config.ini`](config/config.ini). Run commands from the repository root: configuration and relative paths are resolved against the current working directory. Paths inside the INI remain relative to the repository root, not the `config/` folder. If the configuration is missing, the application creates the folder and a default INI file.

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
4. Select `[DEVICE_SETTINGS] device` for the host. The checked-in configuration uses `mps` for the original Mac. For a CPU configuration, use `cpu` and set `[YOLO_SETTINGS] use_half_precision = false`. CUDA requires an appropriate local PyTorch/CUDA installation.
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
| `q` | Quit (the current implementation uses lowercase). |
| Close main window | Quit and stop tower retries. |
| `T` / `t` | Retry the latest desired tower state after recovery exhaustion. |
| `S` / `s` | Save screenshot. |
| `F` / `f` | Toggle fit-to-screen. |
| `G` / `g` | Toggle fullscreen. |
| `P` / `p` | Print and log statistics. |
| `r` | Reset processing counters. |
| `V` / `v` | Toggle video recording. |
| `M` / `m` | Open manual review during `RESULT_HOLD`. |
| Enter / ESC | Confirm / cancel in the review window. |

Review borders: green = agree with Pass, red = agree with NG, yellow = disagree. Confirmation stores choices for the rejudge CSV. Cancellation restores agreement with the original result. A changed review result does **not** change the tower light or original inspection statistics.

With the supplied paths, outputs are:

| Location | Contents |
| --- | --- |
| `logs/detection.log` | JSON lines: configuration, state changes, inspection results, health and review events. |
| `logs/detection_errors.log` | Errors explicitly reported by the application logger. |
| `logs/statistics_<session>.csv` | Per-inspection results, counts, cycle time and closing summary. |
| `logs/rejudge/rejudge_YYYYMMDD.csv` | Per-sealant AI result, agreement, confidence and final `Pass`, `NG`, `REPASS` or `REFAIL`. |
| `validation/` | Raw inspection images labeled OK/NG in the filename. |
| `screenshots/`, `recordings/` | User screenshots and optional recordings. |

Setting `save_detection_log = false` also disables creation of statistics and rejudge writers. Heartbeats are emitted every 60 seconds. Missing frames produce warnings; video end does not automatically exit the main loop in the current monitoring implementation.

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

The original 38 characterization cases pass against both the committed monolith and the extracted implementation. The expanded suite covers module boundaries and simulated lifecycle flows. Real-model accuracy and Windows hardware stability require the manual procedures.

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
