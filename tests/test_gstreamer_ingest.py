import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest

from io_.video_handler import VideoHandler


def test_rtsp_fails_fast_when_pyav_missing(monkeypatch):
    import builtins

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "av":
            raise ImportError("missing av")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    with pytest.raises(RuntimeError, match="PyAV"):
        VideoHandler("rtsp://example.test/live")


def test_rtsp_rejects_unknown_backend(monkeypatch):
    import config as _config

    monkeypatch.setattr(_config, "RTSP_INGEST_BACKEND", "vlc")

    with pytest.raises(RuntimeError, match="Unsupported RTSP_INGEST_BACKEND"):
        VideoHandler("rtsp://example.test/live")


class _FakeCap:
    def get(self, *a, **k):
        return 0

    def read(self):
        return True, None


def test_transient_open_retries_then_succeeds(monkeypatch):
    # First two opens fail with a transient connect error, third succeeds — the worker
    # must ride it out in-process (no crash-loop) rather than raising.
    import config as _config

    monkeypatch.setattr(_config, "RTSP_INITIAL_OPEN_MAX_ATTEMPTS", 5)
    calls = {"n": 0}

    def fake_open(self):
        calls["n"] += 1
        if calls["n"] < 3:
            raise RuntimeError("PyAV failed to open RTSP stream: [Errno 110] timed out")
        return _FakeCap()

    monkeypatch.setattr(VideoHandler, "_open", fake_open)
    monkeypatch.setattr("io_.video_handler.time.sleep", lambda *_: None)
    monkeypatch.setattr(VideoHandler, "_update", lambda self: None)

    vh = VideoHandler("rtsp://example.test/live")
    assert calls["n"] == 3
    assert vh.cap is not None


def test_transient_open_gives_up_after_budget(monkeypatch):
    import config as _config

    monkeypatch.setattr(_config, "RTSP_INITIAL_OPEN_MAX_ATTEMPTS", 3)

    def always_fail(self):
        raise RuntimeError("PyAV failed to open RTSP stream: [Errno 111] refused")

    monkeypatch.setattr(VideoHandler, "_open", always_fail)
    monkeypatch.setattr("io_.video_handler.time.sleep", lambda *_: None)

    with pytest.raises(RuntimeError, match="after 3 attempts"):
        VideoHandler("rtsp://example.test/live")


def test_permanent_open_error_fails_fast(monkeypatch):
    # Missing-library error must not be retried (would waste the whole budget sleeping).
    monkeypatch.setattr("io_.video_handler.time.sleep", lambda *_: None)

    def missing_lib(self):
        raise RuntimeError("RTSP ingest requires PyAV. Install with: pip install av")

    monkeypatch.setattr(VideoHandler, "_open", missing_lib)
    with pytest.raises(RuntimeError, match="requires PyAV"):
        VideoHandler("rtsp://example.test/live")
