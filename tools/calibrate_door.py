#!/usr/bin/env python3
"""Offline per-stream door-threshold calibration from existing logs.

Parses the `[DOOR] SSIM: ... | Diff: ... | Intensity: ... | Stable: STATE`
telemetry the door verifier already prints, builds per-camera closed-state and
open-state SSIM bands (bucketed by intensity so dawn/day/night are visible), and
suggests a safe `ssim_threshold` that sits below the closed-band floor yet above
the open-band ceiling. Cameras whose bands overlap are flagged — those need
reference/ROI work, not just a threshold.

Read-only: prints a report + a suggested per-stream config block. Writes nothing.

Usage:
    python tools/calibrate_door.py [--logs logs] [--margin 0.05] [--cam GF-25-CAM-30]

See docs/superpowers/specs/2026-06-07-door-accuracy-dawn-falseopen-design.md
"""
from __future__ import annotations

import argparse
import glob
import os
import re
import sys

# Allow `python tools/calibrate_door.py` to import the repo's config.py.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

# [DOOR] SSIM: 0.762 | Diff: 24.8 | Intensity: 44.0 (Δ9.7) | Stable: CLOSED
_DOOR_RE = re.compile(
    r"\[DOOR\] SSIM:\s*([0-9.]+)\s*\|\s*Diff:\s*([0-9.]+)\s*\|\s*"
    r"Intensity:\s*([0-9.]+)\s*\(Δ([0-9.]+)\)\s*\|\s*Stable:\s*(OPEN|CLOSED)"
)
# Camera id sits in the path: logs/<store>/<CAM-ID>/<date>/run_*.log
_CAM_RE = re.compile(r"logs/[^/]+/([^/]+)/\d{2}-\d{2}-\d{4}/")


@dataclass
class Sample:
    ssim: float
    diff: float
    intensity: float
    is_open: bool


@dataclass
class CamStats:
    samples: List[Sample] = field(default_factory=list)

    @property
    def closed(self) -> List[float]:
        return [s.ssim for s in self.samples if not s.is_open]

    @property
    def open(self) -> List[float]:
        return [s.ssim for s in self.samples if s.is_open]


def parse_door_lines(text: str) -> List[Sample]:
    out: List[Sample] = []
    for m in _DOOR_RE.finditer(text):
        out.append(
            Sample(
                ssim=float(m.group(1)),
                diff=float(m.group(2)),
                intensity=float(m.group(3)),
                is_open=(m.group(5) == "OPEN"),
            )
        )
    return out


def _percentile(values: List[float], pct: float) -> Optional[float]:
    """Nearest-rank percentile (pct in 0..100). None for empty input."""
    if not values:
        return None
    s = sorted(values)
    if pct <= 0:
        return s[0]
    if pct >= 100:
        return s[-1]
    k = max(0, min(len(s) - 1, int(round((pct / 100.0) * (len(s) - 1)))))
    return s[k]


def suggest_threshold(
    closed: List[float],
    open_: List[float],
    current: Optional[float] = None,
    margin: float = 0.05,
) -> Dict:
    """Pick a per-stream ssim_threshold from the two SSIM bands.

    Threshold should sit BELOW the closed-band floor (so the closed door never
    reads OPEN even at its worst lighting) yet ABOVE the open-band ceiling (so a
    real open still trips). Returns the suggestion plus diagnostics.
    """
    closed_floor = _percentile(closed, 1)      # worst (lowest) closed SSIM
    open_ceiling = _percentile(open_, 99)       # best (highest) open SSIM
    overlap = (
        closed_floor is not None
        and open_ceiling is not None
        and open_ceiling >= closed_floor
    )

    suggestion: Optional[float] = None
    if closed_floor is not None and open_ceiling is not None and not overlap:
        # Midpoint of the gap keeps maximum margin on both sides.
        suggestion = round((closed_floor + open_ceiling) / 2.0, 3)
    elif closed_floor is not None:
        # No usable open samples: sit a margin below the closed floor, clamped.
        suggestion = round(max(0.30, min(0.95, closed_floor - margin)), 3)

    return {
        "closed_n": len(closed),
        "open_n": len(open_),
        "closed_floor": closed_floor,
        "closed_p50": _percentile(closed, 50),
        "open_ceiling": open_ceiling,
        "open_p50": _percentile(open_, 50),
        "overlap": overlap,
        "current": current,
        "suggested": suggestion,
    }


def _relax_factor(curr_mean: float, lo: float = 0.85,
                  dark: float = 20.0, bright: float = 60.0) -> float:
    """Mirror of DoorVerifier._lighting_relax_factor (smooth, no cliff)."""
    if curr_mean >= bright:
        return 1.0
    if curr_mean <= dark:
        return lo
    return lo + (1.0 - lo) * (curr_mean - dark) / (bright - dark)


def _old_active_threshold(threshold: float, curr_mean: float) -> float:
    """The previous hard-band twilight relax (×0.85 only within [20,45))."""
    if 20.0 <= curr_mean < 45.0:
        return threshold * 0.85
    return threshold


def simulate_transitions(samples: List["Sample"], threshold: float,
                         new_logic: bool, hysteresis: float = 0.05,
                         debounce: int = 5) -> int:
    """Replay a logged (ssim, intensity) sequence through the door decision and
    count CLOSED->OPEN transitions. new_logic=True uses smooth relax + hysteresis;
    False uses the old hard-band logic. Lower count on confirmed-closed data = fewer
    false opens."""
    stable_open = False
    cand = False
    cand_run = 0
    transitions = 0
    for s in samples:
        if new_logic:
            active = threshold * _relax_factor(s.intensity)
            if s.intensity < 20.0:
                raw = False
            elif stable_open:
                raw = s.ssim < active
            else:
                raw = s.ssim < (active - hysteresis)
        else:
            active = _old_active_threshold(threshold, s.intensity)
            raw = False if s.intensity < 20.0 else (s.ssim < active)

        if raw == cand:
            cand_run += 1
        else:
            cand, cand_run = raw, 1
        if cand_run >= debounce and stable_open != cand:
            if cand and not stable_open:
                transitions += 1
            stable_open = cand
    return transitions


def collect(logs_dir: str) -> Dict[str, CamStats]:
    cams: Dict[str, CamStats] = defaultdict(CamStats)
    for path in glob.glob(os.path.join(logs_dir, "**", "*.log"), recursive=True):
        norm = path.replace(os.sep, "/")
        cm = _CAM_RE.search(norm)
        if not cm:
            continue
        cam = cm.group(1)
        try:
            with open(path, "r", errors="ignore") as fh:
                text = fh.read()
        except OSError:
            continue
        cams[cam].samples.extend(parse_door_lines(text))
    return cams


def _current_thresholds() -> Dict[str, float]:
    """Map camera_id -> configured ssim_threshold (best-effort import)."""
    try:
        import config  # noqa: WPS433 (runtime import; tool may run outside repo)
    except Exception:
        return {}
    return {
        s.get("camera_id"): s.get("ssim_threshold")
        for s in getattr(config, "STREAMS_CONFIG", [])
        if s.get("camera_id")
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--logs", default="logs", help="logs root dir")
    ap.add_argument("--margin", type=float, default=0.05)
    ap.add_argument("--cam", default=None, help="restrict to one camera_id")
    args = ap.parse_args()

    cams = collect(args.logs)
    current = _current_thresholds()

    if args.cam:
        cams = {k: v for k, v in cams.items() if k == args.cam}

    print(f"{'CAM':16}{'closeN':>8}{'openN':>7}{'cFloor':>8}{'cP50':>7}"
          f"{'oCeil':>7}{'cur':>6}{'NEW':>7}  FLAG")
    suggestions: List[Tuple[str, float]] = []
    for cam in sorted(cams):
        st = cams[cam]
        r = suggest_threshold(st.closed, st.open, current.get(cam), args.margin)
        flag = "OVERLAP!" if r["overlap"] else ("low-data" if r["closed_n"] < 200 else "ok")
        cf = f"{r['closed_floor']:.3f}" if r["closed_floor"] is not None else "-"
        cp = f"{r['closed_p50']:.3f}" if r["closed_p50"] is not None else "-"
        oc = f"{r['open_ceiling']:.3f}" if r["open_ceiling"] is not None else "-"
        cur = f"{r['current']:.2f}" if r["current"] is not None else "-"
        new = f"{r['suggested']:.3f}" if r["suggested"] is not None else "-"
        print(f"{cam:16}{r['closed_n']:>8}{r['open_n']:>7}{cf:>8}{cp:>7}"
              f"{oc:>7}{cur:>6}{new:>7}  {flag}")
        if r["suggested"] is not None:
            suggestions.append((cam, r["suggested"]))

    print("\n# Suggested per-stream ssim_threshold (review before applying):")
    print("# NOTE: 'open' samples are verifier-labelled and may include false opens.")
    print("#       OVERLAP! = closed/open SSIM bands intersect -> a threshold cannot")
    print("#       separate this cam; it needs a fresh reference or ROI relocation,")
    print("#       NOT just a threshold. Apply suggestions only for 'ok' cams.")
    for cam, thr in suggestions:
        print(f'#   "{cam}": ssim_threshold = {thr}')


if __name__ == "__main__":
    main()
