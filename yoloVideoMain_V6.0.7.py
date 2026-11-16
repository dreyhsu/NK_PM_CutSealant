# -*- coding: utf-8 -*-
"""Compatibility entry point. Prefer `python main.py`; legacy class imports still work."""
from cutsealant.app import SafeVideoProcessor, main
from cutsealant.config import ConfigManager
from cutsealant.logs import JSONLogger, StatisticsLogger, RejudgeLogger
from cutsealant.review import ManualReviewWindow
from cutsealant.recording import VideoRecorder

__all__ = ["SafeVideoProcessor", "ConfigManager", "JSONLogger", "StatisticsLogger",
           "RejudgeLogger", "ManualReviewWindow", "VideoRecorder", "main"]

if __name__ == "__main__":
    raise SystemExit(main())
