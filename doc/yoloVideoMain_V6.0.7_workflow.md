# CutSealant V6.0.7 workflow

Source: [yoloVideoMain_V6.0.7.py](../yoloVideoMain_V6.0.7.py). These Mermaid charts describe the current implementation. View this file in a Markdown viewer that supports Mermaid.

## 1. Application and processing threads

```mermaid
flowchart TD
    START([Start: main]) --> CONFIG["Read config_v6.0.ini<br/>Create defaults if missing"]
    CONFIG --> LOGS["Initialize JSON, error, statistics and rejudge logs<br/>when save_detection_log is enabled"]
    LOGS --> MODEL["Check external model file exists<br/>Load YOLO onto configured device"]
    MODEL --> SOURCE["Open video file, USB/UVC camera<br/>or TIS industrial camera"]
    SOURCE --> READY{"Initialization successful?"}
    READY -- No --> STOP
    READY -- Yes --> RECORD["Read initial frame<br/>Initialize optional video recorder"]
    RECORD --> THREADS[Start capture and YOLO threads]
    THREADS --> INIT["Set state = WAITING<br/>Request Standby: yellow light"]
    THREADS --> CAPTURE

    subgraph WORKERS[Background processing]
        CAPTURE[Capture thread: read frames] --> FQ["frame_queue<br/>capacity: 2"]
        FQ --> YOLO["YOLO thread: locate carrier and sealants<br/>Output boxes, confidence scores and classes"]
        YOLO --> RQ["result_queue<br/>capacity: 2"]
        CAPTURE -. Camera failures .-> RECONNECT[Attempt camera reconnection]
        RECONNECT -. Retry .-> CAPTURE
        YOLO -. CUDA / memory errors .-> RECOVER["Attempt memory cleanup<br/>or model reload after repeated errors"]
        RECOVER -. Retry if recovered .-> YOLO
    end

    INIT --> NEXT["Main loop<br/>Heartbeat every 60 seconds<br/>Wait up to 2 seconds for a result"]
    RQ --> NEXT
    NEXT --> AVAILABLE{"Result received?"}
    AVAILABLE -- No --> IDLE[Log idle warnings at intervals]
    IDLE --> NEXT
    AVAILABLE -- Yes --> SENTINEL{"Frame is None?"}
    SENTINEL -- Yes --> STOP
    SENTINEL -- No --> FRAME["Keep raw frame copy; update counters<br/>Draw overlays and detection boxes<br/>Record frame if recording is active"]
    FRAME --> DETECTION["Process carrier detection state machine<br/>See chart 2"]
    DETECTION --> REVIEW[Update manual-review window if open]
    REVIEW --> DISPLAY[Display frame and process keyboard input]
    DISPLAY --> QUIT{"Quit requested?"}
    QUIT -- No --> NEXT
    QUIT -- Yes --> STOP
    DISPLAY -. M during RESULT_HOLD .-> MANUAL["Open manual review<br/>See chart 3"]
    MANUAL -. Subsequent loop iterations .-> REVIEW

    STOP["Stop / cleanup<br/>Write pending review; close review window<br/>Signal and join workers; release camera<br/>Close display windows; write final statistics<br/>Close logs and stop recording"] --> END([Exit])

    classDef standby fill:#fff3cd,stroke:#ac7900,color:#332600;
    classDef io fill:#e8f1ff,stroke:#3973ad,color:#102f50;
    class INIT standby;
    class FQ,RQ,LOGS io;
```

Queue writes have a timeout; a frame or result is dropped when its queue remains full. The main loop keeps waiting when no results arrive, including after video-file exhaustion or a worker stopping; automatic exit on video end is commented out. Unhandled main-loop exceptions and keyboard interruption also lead to cleanup.

## 2. One quality inspection per carrier

```mermaid
flowchart TD
    FRAME([Process current YOLO result]) --> PRESENT{"Carrier AND sealant<br/>classes present?"}
    PRESENT -- Yes --> RESETABS[Reset consecutive absence counter]
    RESETABS --> STATE{"Current state?"}
    STATE -- RESULT_HOLD --> KEEP["Keep previous inspection result<br/>Do not repeat quality analysis<br/>Do not change tower light"]
    STATE -- DETECTED --> KEEP
    STATE -- WAITING --> POSITION{"Carrier centroid below image midpoint<br/>and movement less than 10 pixels?"}
    POSITION -- No --> RESETTIME[Reset stationary counter]
    POSITION -- Yes --> COUNT[Increment stationary counter]
    COUNT --> TIME{"Counter / configured FPS<br/>greater than 1 second?"}
    TIME -- No --> DONE
    TIME -- Yes --> DETECTED["WAITING → DETECTED<br/>Start inspection timing and log session"]
    DETECTED --> HLS[Convert raw frame from BGR to HLS]
    HLS --> CROPS[For each detected sealant: crop lightness channel]
    CROPS --> KMEANS["K-means segmentation: 3 clusters<br/>Select brightest cluster"]
    KMEANS --> CONTOUR["Find largest external contour<br/>Approximate polygon; test convexity"]
    CONTOUR --> AGGREGATE{"Any evaluated contour<br/>fails the convexity test?"}
    AGGREGATE -- No --> PASS["Result: Pass<br/>Request green light"]
    AGGREGATE -- Yes --> NG["Result: NG<br/>Request red light and buzzer"]
    PASS --> HOLD
    NG --> HOLD
    HOLD["DETECTED → RESULT_HOLD<br/>Start minimum hold timer"] --> OUTPUT["Save detection JSON and raw validation JPG<br/>Write inspection statistics CSV<br/>Store frame, boxes and pending review data"]
    OUTPUT --> DONE

    PRESENT -- No --> ABSENT[Increment consecutive absence counter]
    ABSENT --> ISHOLD{"Current state = RESULT_HOLD?"}
    ISHOLD -- Yes --> RELEASE{"At least 5 consecutive absent results<br/>AND hold duration at least 2 seconds?"}
    RELEASE -- No --> KEEP
    RELEASE -- Yes --> WRITE["Write pending rejudge CSV<br/>Close any open review window"]
    WRITE --> CLEAR[Clear inspection and review data]
    CLEAR --> WAIT["State → WAITING<br/>Request Standby: yellow light<br/>Reset stationary tracking"]
    ISHOLD -- No --> TOLERANCE{"At least 5 consecutive<br/>absent results?"}
    TOLERANCE -- No --> KEEP
    TOLERANCE -- Yes --> WAIT
    WAIT --> DONE
    KEEP --> DONE
    RESETTIME --> DONE
    DONE([Return to main loop])

    classDef pass fill:#dcfce7,stroke:#248344,color:#14532d;
    classDef fail fill:#fee2e2,stroke:#b94040,color:#7f1d1d;
    classDef standby fill:#fff3cd,stroke:#ac7900,color:#332600;
    class PASS pass;
    class NG fail;
    class WAIT standby;
```

Thresholds shown are from the supplied INI: `fps=15`, `stop_threshold_seconds=1`, `distance_threshold_pixels=10`, `absent_tolerance_frames=5`, `result_hold_min_seconds=2`, and `kmeans_clusters=3`. Stationary time is estimated from a processed-result counter divided by configured FPS, rather than measured elapsed time.

YOLO supplies object locations; the separate OpenCV contour test supplies the quality verdict. Despite the helper name `_loop_closed_identify`, its actual final test is `cv2.isContourConvex`. Empty crops, missing contours and caught per-crop errors do not add a failing result, so “Pass” here means no evaluated result failed; it does not guarantee every crop was successfully evaluated.

The chart groups result selection, light requests and entry into `RESULT_HOLD` for readability. In the code, the detection JSON is written first, the state changes to `RESULT_HOLD`, and then the light command and remaining outputs run. The brief `DETECTED` state normally completes within the same main-loop iteration.

## 3. On-demand manual review

```mermaid
flowchart TD
    KEY([M key pressed]) --> ALLOWED{"State = RESULT_HOLD<br/>and pending review available?"}
    ALLOWED -- No --> IGNORE[Ignore / return]
    ALLOWED -- Yes --> OPEN["Display stored sealant crops<br/>Preserve relative spatial positions<br/>Restore previous agreement choices"]
    OPEN --> CLICK["Click a crop to toggle agreement<br/>Green = agree with Pass<br/>Red = agree with NG<br/>Yellow = disagree"]
    CLICK --> ACTION{"Review action?"}
    ACTION -- "Confirm / Enter" --> CONFIRM[Store agreement choices in pending review]
    ACTION -- "Cancel / ESC" --> CANCEL[Reset every crop to agree with original result]
    CONFIRM --> CLOSE[Close review window; log review event]
    CANCEL --> CLOSE
    CLOSE --> PENDING["Remain in RESULT_HOLD<br/>Can press M again<br/>Tower light and original statistics stay unchanged"]
    PENDING --> LEAVE[Carrier removal conditions met, or shutdown]
    LEAVE --> WRITE["Append one CSV row per sealant<br/>to logs/rejudge/rejudge_YYYYMMDD.csv"]
    WRITE --> FINAL["Agree: keep Pass or NG<br/>Disagree with Pass: REFAIL<br/>Disagree with NG: REPASS"]
```

Without manual review, all crops default to agreement. If the review window is still open when pending data is written, the program treats it as cancellation and resets all agreement flags to `True`. The CSV also records whether the operator opened review. JSON, statistics CSV and rejudge CSV writers are initialized only when `save_detection_log` is enabled.

## 4. PATLITE tower command and logging path

```mermaid
flowchart TD
    REQUEST([Request Standby, Pass or NG]) --> CACHE{"Requested status differs<br/>from cached last_light_status?"}
    CACHE -- No --> SKIP[Skip USB command; return cached status]
    CACHE -- Yes --> CHILD["Run python LR6main.py with light arguments<br/>Subprocess call blocks the main loop<br/>No subprocess-level timeout"]
    CHILD --> USB["Child: find USB device<br/>set_configuration; write command<br/>finally: dev.reset"]
    USB --> RETURN{"Light subprocess outcome?"}
    RETURN -- "Nonzero exit code" --> PRINT["Print failure to console only<br/>Return failure; keep old cached status"]
    RETURN -- "Exit code 0" --> ISNG{"Requested status = NG?"}
    ISNG -- Yes --> BUZZ["Run second child process for buzzer<br/>Ignore its return code"]
    ISNG -- No --> LOG
    BUZZ --> LOG["Write status_change JSON event<br/>with requested new_status"]
    LOG --> UPDATE[Update cached last_light_status]
    CHILD -. Parent-side exception .-> ERROR["Log light_control_error<br/>Return failure; keep old cached status"]
```

This path records software commands, with no physical light readback. Child-process USB exceptions/output are not captured into the application log. Command functions returning `False` are ignored by the child entry point, so some write failures can still yield exit code `0`. A cached successful status also prevents resending the same requested state after a physical disconnect. A failed Pass/NG command is not retried continuously in `RESULT_HOLD`; another request must reach `_call_light_once`.

`stop()` does not explicitly send an all-off command to the tower.

## Keyboard controls

| Key | Action |
| --- | --- |
| `q` | Quit |
| `S` / `s` | Save screenshot |
| `P` / `p` | Print and log statistics |
| `r` | Reset processing counters |
| `F` / `f` | Toggle fit-to-screen |
| `G` / `g` | Toggle fullscreen |
| `V` / `v` | Toggle recording |
| `M` / `m` | Open review during `RESULT_HOLD` |
| Enter / ESC | Confirm / cancel in review window |

## Code navigation

| Area | Main methods / classes |
| --- | --- |
| Startup and settings | `main`, `SafeVideoProcessor.initialize`, `ConfigManager` |
| Capture and inference | `_capture_frames`, `_attempt_camera_reconnect`, `_yolo_worker`, `_yolo_detect` |
| Main loop and inspection state | `run`, `_handle_objects_present`, `_handle_objects_absent`, `_check_detection_trigger` |
| Quality analysis | `_cut_sealant_detection`, `_img_kmeans`, `_loop_closed_identify` |
| Human review | `ManualReviewWindow`, `_open_manual_review`, `_handle_review_complete`, `_write_pending_review_log` |
| Tower control | `_call_light_once`, `_call_light`, external `LR6main.py` |
| Logs and reporting | `JSONLogger`, `StatisticsLogger`, `RejudgeLogger`, `_log_heartbeat` |
| Cleanup | `SafeVideoProcessor.stop` |
