"""Event payloads pushed over the WebSocket.

Wire format is plain JSON, no schema versioning beyond the top-level `type`
discriminator.  Keep these dumb dicts so the frontend can consume them
without a build step.
"""
from __future__ import annotations
from dataclasses import dataclass, field, asdict
from typing import Any


def _round_list(xs, n=3):
    return [round(float(v), n) for v in xs]


@dataclass
class FrameEvent:
    """Per-window detection update — emitted ~2× per second."""
    type: str = "frame"
    t_s: float = 0.0                        # seconds since pipeline start
    p_drone: float = 0.0                    # smoothed probability for this window
    p_drone_raw: float = 0.0                # raw (unsmoothed) probability
    threat: str = "clear"                   # clear | unknown | detected | decoy
    longest_run: int = 0                    # consecutive windows >= threshold
    drone_position_m: list | None = None    # [x, y, z] — present iff localised
    drone_velocity_m_s: list | None = None  # [vx, vy, vz]
    bearing_deg: float | None = None        # 0-360, 0 = +y axis
    range_m: float | None = None
    altitude_m: float | None = None
    open_set_unknown: bool = False
    decoy_label: str | None = None          # decoy | real_drone | unreliable | insufficient
    residual: float | None = None
    # Drone classification (specific type) — populated when the runtime
    # acoustic fingerprint matches a known reference library.
    # Today we ship a Shahed-136 prototype; "unknown" otherwise.
    drone_class: str = "unknown"
    class_confidence: float = 0.0           # rolling-mean cosine similarity vs prototype
    # Per-mic input health — judges need to SEE their cabling work.
    rms_per_channel: list = field(default_factory=list)        # 0..1 per mic
    peak_per_channel: list = field(default_factory=list)        # 0..1 per mic
    silent_channels: list = field(default_factory=list)         # indices that look dead
    source_state: str = "none"              # none | live | simulate
    source_label: str = ""                  # human-readable source description
    ts_iso: str = ""


@dataclass
class LogEvent:
    """Discrete event for the operator log (push when state transitions)."""
    type: str = "log"
    t_s: float = 0.0
    key: str = ""                           # i18n key — see dashboard/i18n.js
    tone: str = "info"                      # info | ok | warn | alert
    confidence: float | None = None
    ts_iso: str = ""


@dataclass
class CueEvent:
    """Cueing handoff JSON — mirrors scripts/triangulate.py shape."""
    type: str = "cue"
    t_s: float = 0.0
    cueing: dict = field(default_factory=dict)
    ts_iso: str = ""


@dataclass
class HelloEvent:
    """First message after connect — tells the client about the rig."""
    type: str = "hello"
    sample_rate: int = 16_000
    n_channels: int = 4
    threshold: float = 0.25
    mics: list = field(default_factory=list)
    site_id: str = "RUH-14"
    model_id: str = ""
    mode: str = "live"                      # live | simulate | none
    source_label: str = ""
    has_default_simulate: bool = False      # backend was started with a default WAV
    default_simulate_label: str = ""        # filename of the default WAV (no full path)
    ts_iso: str = ""


@dataclass
class SourceEvent:
    """Audio-source state change — emitted on (re)open, error, or close.

    The dashboard surfaces this as a banner so the operator knows
    immediately whether the rig is hearing anything.
    """
    type: str = "source"
    state: str = "none"                     # none | opening | live | simulate | error
    label: str = ""                         # e.g. "Behringer UMC404HD (4 ch @ 48 kHz)"
    detail: str = ""                        # error message or extra info
    sample_rate: int | None = None          # native rate of the source
    n_channels: int | None = None
    ts_iso: str = ""


def to_dict(ev) -> dict[str, Any]:
    return asdict(ev)
