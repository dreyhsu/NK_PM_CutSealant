# Lights-only endurance test

Close the inspection application, connect the tower using the normal Windows
USB setup, then double-click `run_light_tower_stability_test.bat` in the
repository folder. The launcher activates `env_ultralytics` from
`C:\Users\Admin\mambaforge`. Edit the launcher variables if those locations change.

The default run lasts **8 hours** and repeats **yellow -> green -> yellow -> red**,
waiting **5 seconds after each command**. Only LED command `2` is used; the test
never activates the buzzer. Existing `LR6main.py` USB configuration, packet
format, Windows write-length adjustment, and post-write USB reset are preserved.

For a one-minute setup check, run this from Windows Command Prompt in the repo:

```bat
run_light_tower_stability_test.bat --hours 0.016667 --hold-seconds 5
```

Then run the full test:

```bat
run_light_tower_stability_test.bat --hours 8 --hold-seconds 5 --timeout-seconds 3
```

Each command has a three-second timeout by default. Failures and timeouts are
logged, and the sequence continues after the normal five-second wait. A timed-out
child is killed and given one second to exit. If its exit cannot be confirmed,
the test stops instead of starting another USB owner. The duration includes
command execution and waits; no new command starts after the run deadline.
Final child cleanup may extend slightly past the deadline.

The console shows the requested color and cumulative counts. Each run creates
a timestamped JSONL file in `logs/light_tower_stability/`. It contains startup
settings and environment information, command-start records, outcomes, durations,
exit codes, USB stdout/stderr, and a final summary. Output is limited to the last
4096 bytes per stream in each record, with truncation flags. A start record
without a result can identify a forcibly interrupted run. Check the latest
`run_summary` for completion; a missing summary does not indicate success.

Press **Ctrl+C** to stop. No all-off or buzzer command is sent on shutdown; the
last light state is preserved. Exit codes are **0** for completion without command
errors, **1** for completion with errors, **2** for setup failure or aborted cleanup,
and **130** for interruption. The launcher leaves the console open afterward.

Observe the actual lamp periodically, especially after five hours. Passing means
eight hours without transport errors or timeouts plus observed physical color
changes beyond the old failure point. USB write success is not physical lamp
readback. If yellow stays on despite successful writes, note the wall-clock time
and keep the log. This standalone test isolates the tower path; it does not
exercise camera/YOLO processing or prove that the complete inspection application
is stable. Windows batch execution and physical endurance require the actual
Windows machine and tower.
