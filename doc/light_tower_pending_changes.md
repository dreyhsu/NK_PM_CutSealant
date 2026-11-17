# Light tower control: pending modifications

Status: **not implemented**. Updated: 2026-11-17.

## Simple explanation: what needs to change

The goal is to keep the PATLITE controllable during long production runs and make failures visible without freezing inspection.

1. **Stop resetting USB after every command.** Send the light command normally; release the connection properly when it closes.
2. **Report success and failure correctly.** A failed light or buzzer command must be recorded as a failure, not accepted as a successful change.
3. **Keep the inspection screen responsive.** A stuck USB command needs a time limit and must not block the main loop indefinitely.
4. **Recover the connection and restore the latest required light.** Retry failed light changes even while an inspection result is being held, without inspecting or counting the same carrier again.
5. **Save useful troubleshooting records.** Record the requested color, command result, USB error, retry and reconnect events so the next freeze can be investigated.

The existing color rules stay the same: **yellow = standby, green = Pass, red + buzzer = NG**. Yellow itself is a normal command issued by this application. The fault to investigate is the tower becoming unresponsive afterward.

Repeated USB resets are a suspect in the current code, but the root cause of the reported Windows failure has not been confirmed. Completing these changes does not by itself prove that a driver or hardware fault can always be recovered without restarting Windows.

## Pending work, highest priority first

All items below are open. They are a backlog for a separate reliability change; the completed module refactor intentionally preserved existing USB behavior.

| Priority / ID | Current problem | Required modification | Acceptance condition |
| --- | --- | --- | --- |
| P0-01 | `send_command()` always calls `dev.reset()` in `finally`, including after successful writes. | Remove routine hardware resets. Do not replace them with an unconditional reset-based retry loop. Keep any future explicit reset/recovery operation separate and logged. | Successful writes and write failures do not automatically call `dev.reset()`. |
| P0-02 | `usb_close()` does nothing; every child invocation calls `set_configuration()`. | Use `try/finally` to release the device handle/interfaces with `usb.util.dispose_resources(dev)`. Check the active configuration before setting it; do not reconfigure an already correctly configured device on each command. Release acquired resources on error paths too. | Fake-device tests verify cleanup on success/failure and configuration is changed only when needed. |
| P0-03 | Child `main()` ignores command functions returning `False`; the parent ignores the buzzer exit code. | Propagate command failures through meaningful exit codes while the CLI is retained. Track light and buzzer outcomes separately. Check actual write lengths and log raw values; verify the existing Windows `write_length -= 1` workaround against the deployed backend before changing it. | Short writes, USB exceptions, invalid arguments and buzzer failures cannot be silently reported as full success. A successful LED write remains distinguishable from a failed buzzer write. |
| P0-04 | `subprocess.call()` blocks the main/UI loop with no overall process timeout. The USB write timeout does not cover opening, configuration or reset. | Execute tower operations outside the main/UI thread, serialize commands, and enforce a bounded overall command wait. During CLI migration, use the running interpreter (`sys.executable`), capture output, and bound child-process execution and cleanup. | A deliberately hung controller does not prevent frame display, keyboard handling or bounded shutdown; its timeout is recorded. |
| P0-05 | Cache suppression can prevent restoring the light after disconnect; failed Pass/NG commands are not retried continuously in `RESULT_HOLD`. | Separate the latest desired state from the last successfully transmitted state. Invalidate transmission state on disconnect/timeout/reconnect. Schedule bounded retries with backoff independently of inspection triggers, including during result hold. Supersede stale queued colors with the latest desired state. | A failed light change can recover during result hold without a second inspection/statistics row. Reconnecting resends the latest desired color. An old delayed command cannot overwrite a newer desired state. |
| P0-06 | Some failures are console-only; child output and USB lifecycle events are absent from saved logs. | Persist command requests/outcomes, nonzero exits, exceptions, timeouts, reconnect attempts and recovery results. Capture child stdout/stderr while subprocesses remain. Use truthful transport-success wording instead of implying physical light confirmation. | Tests can reconstruct a request, failure, retry and recovery from saved records; failures are not lost in console output alone. |
| P1-01 | Each color change launches another Python process and opens the device again; NG opens another process for its buzzer. | After the P0 correctness checks, move to a long-lived, serialized USB controller with one owned device session. Prefer a dedicated worker process if backend calls cannot reliably be interrupted, so the parent can bound its wait and restart the worker. Keep LED and buzzer outcomes separate even when sent in one protocol command. | Repeated state changes reuse a connection until a fault occurs; commands cannot write concurrently; worker restart does not start overlapping owners. |
| P1-02 | Retries of NG could repeatedly sound the buzzer; LED commands currently preserve the existing buzzer state. | Define buzzer activation per carrier/NG event separately from restoring a steady LED color. Prevent routine light retries/reconnections from replaying a completed alarm. Specify handling of a partially failed alarm and transitions back to Pass/standby. | Repeated LED recovery does not continually retrigger the same alarm; buzzer failures remain visible and independently controlled. |
| P1-03 | Shutdown releases no explicit tower session and does not send an all-off command. Existing application cleanup can be called more than once. | Make controller cleanup idempotent, stop retries, bound worker termination and release resources. Agree on the production shutdown/alarm policy before changing the physical light state; do not silently make all-off the default. | Repeated stop calls do not leak handles or enqueue new commands. The agreed final-state policy is documented and tested. |
| P1-04 | There is no actual tower health/readback monitoring. | Maintain transport health (`connected`, `retrying`, `unavailable`) independently of inspection Pass/NG. Check the exact model/backend for supported device queries before adding readback. If physical confirmation is unavailable, expose that limitation explicitly. | UI/logs distinguish an inspection result from communication health and never present a cached request as confirmed physical state. |

An isolated worker is a proposed implementation direction, not a guarantee that restarting a process will recover a stalled Windows USB driver. Recovery failure must remain visible rather than being hidden by endless retries.

## Logging additions

Proposed events: `tower_command_requested`, `tower_command_completed`, `tower_command_failed`, `tower_command_timeout`, `tower_disconnected`, `tower_reconnect_attempt`, `tower_reconnected`, and `tower_recovery_exhausted`.

Include the following where available:

- Timestamp, application session ID, command ID, and carrier ID when associated with an inspection.
- Requested light/buzzer state, operation, attempt number, duration and transport outcome.
- USB exception type/message, backend/error code, and expected/actual write lengths.
- Child exit code and bounded stdout/stderr while the CLI is retained.
- Desired state versus last successfully transmitted state; whether physical readback is available.

Do not label a successful write as confirmation that the lamp physically changed. Decide explicitly how tower errors will be retained when the existing `save_detection_log` flag is disabled; the current application only creates its logger when that flag is enabled.

## Verification before deployment

### Automated checks

- [ ] No routine device reset after successful or failed writes.
- [ ] Handles/interfaces released after write, open/configuration failure and shutdown, when acquired.
- [ ] Correct command bytes for standby, Pass, NG and buzzer; correct short-write/error propagation.
- [ ] No update of transmitted-state cache on a failed LED command; partial buzzer failure is represented separately.
- [ ] Hung subprocess/worker is bounded without blocking the UI loop; no concurrent USB writers.
- [ ] Retry/reconnect during `RESULT_HOLD` restores the latest desired light without repeating inspection, statistics or a completed buzzer event.
- [ ] Newer desired states supersede stale queued commands; retry backoff and retry limits use a fake clock.
- [ ] Failure, retry and recovery information persists in logs, including the chosen behavior with detection logging disabled.
- [ ] Shutdown is bounded and idempotent, including when a worker is stuck.

Current [tower tests](../test/test_tower.py) characterize legacy behavior. Replace affected `known_issue` expectations when their corresponding fix is implemented, rather than preserving a defect to keep those tests passing. Retain the existing color-mapping checks and run the full application regression suite. See the [ranked test guide](testing.md).

### Windows and physical-device checks

- [ ] Record the actual tower model, Windows build, Python/PyUSB versions, USB backend/driver, USB port/hub/cable and observed time-to-failure.
- [ ] Verify yellow, green, red and buzzer on the real device; compare physical behavior with the command log.
- [ ] Disconnect/reconnect during standby and result hold. Verify restoration or a clearly reported unrecoverable fault.
- [ ] Compare direct USB-port operation with the deployed hub/cable setup. Investigate Windows power-management and supply conditions if failures persist; do not change drivers or system settings blindly.
- [ ] Run production-like transitions for at least 24 hours and beyond the previously observed failure interval. Record command counts, timeouts, reconnects, memory, handle counts, physical observations and any reboot requirement.

Physical hardware checks have **not** been completed in this repository's automated test environment.

## Implementation locations

- [USB command implementation](../LR6main.py): session setup/release, packet writes, error propagation and CLI exit codes.
- [Tower controller](../cutsealant/tower.py): desired/transmitted state, serialized dispatch, timeouts and recovery.
- [Application lifecycle](../cutsealant/app.py): nonblocking integration and shutdown.
- [Inspection workflow](../cutsealant/workflow.py): keep inspection results independent of tower retry/recovery.
- [Configuration](../config/config.ini): add timeout/backoff/recovery options only when their behavior and defaults are implemented and documented.

Scope exclusions: no change to sealant analysis, four-region completeness rules, manual-review verdicts, or the existing meaning of yellow/green/red as part of these tower-reliability items.
