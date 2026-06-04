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
