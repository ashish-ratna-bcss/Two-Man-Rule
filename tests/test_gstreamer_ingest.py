import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest

from io_.video_handler import VideoHandler, build_gstreamer_rtsp_pipeline


def test_gstreamer_rtsp_pipeline_uses_audit_defaults():
    pipeline = build_gstreamer_rtsp_pipeline(
        "rtsp://user:pass@example.test/Streaming/Channels/101",
        transport="tcp",
        latency_ms=1000,
        drop_on_latency=False,
    )

    assert "rtspsrc" in pipeline
    assert "protocols=tcp" in pipeline
    assert "latency=1000" in pipeline
    assert "drop-on-latency=false" in pipeline
    assert "decodebin" in pipeline
    assert "appsink name=appsink" in pipeline


def test_gstreamer_pipeline_rejects_unknown_transport():
    with pytest.raises(ValueError, match="Unsupported RTSP transport"):
        build_gstreamer_rtsp_pipeline("rtsp://example.test/live", transport="sctp")


def test_rtsp_fails_fast_when_gstreamer_missing(monkeypatch):
    import builtins

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "gi":
            raise ImportError("missing gi")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    with pytest.raises(RuntimeError, match="PyGObject/GStreamer"):
        VideoHandler("rtsp://example.test/live")
