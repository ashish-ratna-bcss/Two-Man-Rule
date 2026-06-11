#!/usr/bin/env python3

import sys
import time
import threading
from pathlib import Path

import av

# --------------------------------------------------
# Load config.py from project root
# --------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config import STREAMS_CONFIG


REPORT_INTERVAL = 30


class CameraStats:
    def __init__(self):
        self.current = 0.0
        self.minimum = float("inf")
        self.maximum = 0.0
        self.total = 0.0
        self.samples = 0
        self.lock = threading.Lock()

    def update(self, bitrate):
        with self.lock:
            self.current = bitrate
            self.minimum = min(self.minimum, bitrate)
            self.maximum = max(self.maximum, bitrate)
            self.total += bitrate
            self.samples += 1

    def snapshot(self):
        with self.lock:
            return {
                "current": self.current,
                "min": 0.0 if self.minimum == float("inf") else self.minimum,
                "max": self.maximum,
                "avg": (
                    self.total / self.samples
                    if self.samples > 0
                    else 0.0
                ),
            }


camera_stats = {}


def monitor_stream(camera_id, rtsp_url):

    while True:

        try:

            print(f"[{camera_id}] Connecting...")

            container = av.open(
                rtsp_url,
                options={
                    "rtsp_transport": "tcp"
                }
            )

            bytes_received = 0
            start_time = time.time()

            for packet in container.demux():

                bytes_received += packet.size

                elapsed = time.time() - start_time

                if elapsed >= 1.0:

                    bitrate_mbps = (
                        bytes_received * 8
                    ) / elapsed / 1_000_000

                    camera_stats[camera_id].update(
                        bitrate_mbps
                    )

                    bytes_received = 0
                    start_time = time.time()

        except Exception as e:

            print(
                f"[{camera_id}] ERROR: {e}"
            )

            time.sleep(5)


def print_report():

    while True:

        time.sleep(REPORT_INTERVAL)

        print("\n")
        print("=" * 140)
        print(
            f"RTSP BITRATE REPORT - {time.strftime('%Y-%m-%d %H:%M:%S')}"
        )
        print("=" * 140)

        print(
            f"{'CAMERA':<25}"
            f"{'CURRENT(Mbps)':>15}"
            f"{'MIN(Mbps)':>15}"
            f"{'MAX(Mbps)':>15}"
            f"{'AVG(Mbps)':>15}"
        )

        print("-" * 140)

        total_current = 0.0
        total_avg = 0.0

        for camera_id in sorted(camera_stats.keys()):

            s = camera_stats[camera_id].snapshot()

            total_current += s["current"]
            total_avg += s["avg"]

            print(
                f"{camera_id:<25}"
                f"{s['current']:>15.2f}"
                f"{s['min']:>15.2f}"
                f"{s['max']:>15.2f}"
                f"{s['avg']:>15.2f}"
            )

        print("-" * 140)
        print(
            f"{'TOTAL CURRENT':<25}{total_current:.2f} Mbps"
        )
        print(
            f"{'TOTAL AVERAGE':<25}{total_avg:.2f} Mbps"
        )
        print("=" * 140)


def main():

    print(
        f"Loaded {len(STREAMS_CONFIG)} streams"
    )

    for stream in STREAMS_CONFIG:

        camera_id = stream["camera_id"]
        rtsp_url = stream["rtsp_url"]

        camera_stats[camera_id] = CameraStats()

        t = threading.Thread(
            target=monitor_stream,
            args=(camera_id, rtsp_url),
            daemon=True,
        )

        t.start()

        time.sleep(0.5)

    print_report()


if __name__ == "__main__":
    main()