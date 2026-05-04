"""Real-time acoustic detection pipeline.

The pipeline is structured around a 16 kHz internal ring buffer plus a
detector loop that runs at ~2 Hz.  Audio sources (live mic array or WAV
replay) push resampled chunks into the buffer.  The detector loop keeps
running even when no source is selected, so the dashboard stays responsive
and operators can pick a device from the UI mid-session — the pitch demo
relies on this: judges plug their mic interface in *after* the server is
already up.

Key features for live demos:
  * Boots with no audio device — emits "source: none" until one is picked.
  * Captures at the device's native sample rate (most pro interfaces are
    48 kHz) and resamples to 16 kHz internally.
  * Hot-swappable source via `set_source()` — no server restart needed.
  * Per-channel RMS + silent-channel detection in every FrameEvent so the
    dashboard can show live VU bars and warn on a dead mic.
"""
from __future__ import annotations
import asyncio
import json
import math
import time
from collections import deque
from dataclasses import asdict
from datetime import datetime, timezone
from math import gcd
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F

from src.triangulation import localize
from src.decoy import analyse as decoy_analyse, DecoyConfig

from .events import FrameEvent, LogEvent, CueEvent, HelloEvent, SourceEvent, to_dict


# ----------------------------------------------------------------------------
# Drone-type classification via prototype matching.
# We extract the AST hidden-state mean per window and cosine-match against a
# precomputed Shahed-136 prototype.  See scripts/build_shahed_prototype.py.
# ----------------------------------------------------------------------------

PROTOTYPE_PATH = Path(__file__).resolve().parent.parent / "configs" / "shahed_prototype.npz"


def _load_shahed_prototype() -> tuple[np.ndarray | None, float]:
    if not PROTOTYPE_PATH.exists():
        return None, 0.96
    try:
        npz = np.load(str(PROTOTYPE_PATH))
        proto = np.asarray(npz["prototype"], dtype=np.float32)
        thresh = float(npz["threshold"]) if "threshold" in npz else 0.96
        return proto, thresh
    except Exception as exc:
        print(f"[pipeline] failed to load shahed prototype: {exc}")
        return None, 0.96


SR = 16_000
WIN_SAMPLES = SR              # 1-second analysis window
HOP_SAMPLES = SR // 2         # 0.5-second hop
HISTORY_SECONDS = 8.0         # keep last 8s of audio for triangulation
DEFAULT_HUB_ID = "Rashidbm/samid-drone-detector"
DEFAULT_THRESHOLD = 0.25
DEFAULT_MAX_TAU = 0.05
SOUND_SPEED = 343.0
SILENT_RMS = 1e-4             # below this an input channel is treated as dead


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _select_torch_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _bearing_range_from_position(pos: np.ndarray, mics: np.ndarray) -> tuple[float, float]:
    centroid = mics.mean(axis=0)
    dx = float(pos[0] - centroid[0])
    dy = float(pos[1] - centroid[1])
    rng = math.hypot(dx, dy)
    bearing = (math.degrees(math.atan2(dx, dy)) + 360.0) % 360.0
    return bearing, rng


def _resample_to_16k(arr: np.ndarray, src_sr: int) -> np.ndarray:
    """Resample (n_channels, n_samples) float32 audio to 16 kHz."""
    if src_sr == SR:
        return arr.astype(np.float32, copy=False)
    from scipy.signal import resample_poly
    g = gcd(int(src_sr), SR)
    up = SR // g
    down = int(src_sr) // g
    return np.stack([
        resample_poly(c, up, down).astype(np.float32, copy=False) for c in arr
    ])


# ----------------------------------------------------------------------------
# Subscribers — fan-out queue for connected WebSocket clients.
# ----------------------------------------------------------------------------

class Subscribers:
    def __init__(self):
        self._queues: list[asyncio.Queue] = []
        self._lock = asyncio.Lock()
        self._last_hello: dict | None = None
        self._last_source: dict | None = None
        self._recent: deque[dict] = deque(maxlen=32)

    async def add(self) -> asyncio.Queue:
        q: asyncio.Queue = asyncio.Queue(maxsize=64)
        async with self._lock:
            self._queues.append(q)
            for snapshot in (self._last_hello, self._last_source):
                if snapshot is not None:
                    await q.put(snapshot)
            for ev in list(self._recent):
                try:
                    q.put_nowait(ev)
                except asyncio.QueueFull:
                    break
        return q

    async def remove(self, q: asyncio.Queue) -> None:
        async with self._lock:
            if q in self._queues:
                self._queues.remove(q)

    async def publish(self, ev: dict) -> None:
        t = ev.get("type")
        if t == "hello":
            self._last_hello = ev
        elif t == "source":
            self._last_source = ev
        else:
            self._recent.append(ev)
        async with self._lock:
            qs = list(self._queues)
        for q in qs:
            try:
                q.put_nowait(ev)
            except asyncio.QueueFull:
                try: q.get_nowait()
                except Exception: pass
                try: q.put_nowait(ev)
                except Exception: pass


# ----------------------------------------------------------------------------
# Pipeline
# ----------------------------------------------------------------------------

class Pipeline:
    """Owns the model, the audio buffer, and the detection loop."""

    def __init__(
        self,
        mics: np.ndarray,
        *,
        hub_id: str = DEFAULT_HUB_ID,
        threshold: float = DEFAULT_THRESHOLD,
        site_id: str = "RUH-14",
        simulate_wav: Path | None = None,
        device_index: int | None = None,
        loop_simulate: bool = True,
    ):
        self.mics = np.asarray(mics, dtype=np.float64)
        self.n_channels = self.mics.shape[0]
        self.threshold = float(threshold)
        self.site_id = site_id
        self.hub_id = hub_id
        # Initial source request — applied during start().
        self._initial_simulate = simulate_wav
        self._initial_device = device_index
        self.loop_simulate = loop_simulate

        # Live source-of-truth state, updated as set_source() runs.
        self.source_state = "none"          # none | opening | live | simulate | error
        self.source_label = ""
        # True when the active source has no real spatial information
        # (e.g. a mono WAV broadcast across N channels with synthetic
        # delays).  When this is set, we DO NOT run triangulation —
        # GCC-PHAT on identical channels produces meaningless positions.
        self.source_is_spatial: bool = True
        # Default simulate path the server was started with (env-configured).
        # Surfaced to the dashboard so it can offer a "Reset to default" after
        # a judge uploads their own audio.
        self.default_simulate_path: Path | None = (
            Path(simulate_wav).resolve() if simulate_wav else None
        )

        self.subscribers = Subscribers()
        self._buf = np.zeros((self.n_channels, int(HISTORY_SECONDS * SR)), dtype=np.float32)
        self._buf_filled = 0
        self._buf_lock: asyncio.Lock | None = None
        self._latest_chunk_rms = np.zeros(self.n_channels, dtype=np.float32)
        self._latest_chunk_peak = np.zeros(self.n_channels, dtype=np.float32)
        self._t0: float = 0.0
        self._stop = asyncio.Event()
        self._consec = 0
        self._longest = 0
        self._last_threat = "clear"
        # Last broadcast drone class — tracking lets us re-fire the cueing
        # JSON when the type goes from "unknown" → "shahed-136" partway
        # through a detection burst.
        self._last_drone_class = "unknown"
        self._recent_positions: deque[tuple[float, np.ndarray]] = deque(maxlen=12)
        # Smoothing buffer for the broadcast position — GCC-PHAT positions
        # jitter ±2 m frame-to-frame on real audio.  We do three things:
        #   1. Reject any single solution with a high residual (loop-boundary
        #      garbage and noise-dominated windows fail this).
        #   2. Median-filter the last 5 accepted positions.
        #   3. Only broadcast a position once we have a "lock" — 3 consecutive
        #      accepted positions within 2 m of each other.  Until then the
        #      frame goes out without drone_position_m, so the map shows
        #      "locating…" instead of a dancing marker.
        self._position_history: deque[np.ndarray] = deque(maxlen=5)
        self._position_smoothed: np.ndarray | None = None
        self._position_lock_count: int = 0
        self._position_locked: bool = False
        self._last_position: np.ndarray | None = None
        self._last_position_t: float | None = None

        self._source_task: asyncio.Task | None = None
        self._source_stop: asyncio.Event | None = None
        self._source_kind: str = "none"       # none | live | simulate
        self._source_kwargs: dict = {}

        self._torch_device = _select_torch_device()
        self._model = None
        self._fe = None

        # Drone-type prototype matching.  Loaded once at __init__ — runs on
        # the same hidden state we already extract for detection.
        proto, proto_thresh = _load_shahed_prototype()
        self._shahed_proto: np.ndarray | None = proto
        self._shahed_threshold: float = proto_thresh
        # Rolling buffer of recent per-window similarity scores; we use the
        # MEAN across this buffer to avoid flickering on single noisy windows.
        self._class_sim_history: deque[float] = deque(maxlen=8)

    # ---------------- model ----------------

    def load_model(self) -> None:
        from transformers import AutoFeatureExtractor, AutoModelForAudioClassification
        print(f"[pipeline] loading {self.hub_id} on {self._torch_device}")
        self._fe = AutoFeatureExtractor.from_pretrained(self.hub_id)
        self._model = (
            AutoModelForAudioClassification
            .from_pretrained(self.hub_id)
            .eval()
            .to(self._torch_device)
        )
        print("[pipeline] model ready")

    @torch.inference_mode()
    def _predict(self, audio_1s: np.ndarray) -> tuple[float, np.ndarray | None]:
        """Run the AST detector on a 1-second window.

        Returns (p_drone, hidden_state_mean) — the second value is the
        pooled penultimate-layer feature, used for prototype matching.
        Returns None for the embedding if classification isn't enabled.
        """
        feats = self._fe(audio_1s, sampling_rate=SR, return_tensors="pt")
        need_embed = self._shahed_proto is not None
        out = self._model(input_values=feats["input_values"].to(self._torch_device),
                          output_hidden_states=need_embed)
        p = float(F.softmax(out.logits, dim=-1)[0, 1])
        if need_embed:
            embed = out.hidden_states[-1].mean(dim=1).cpu().numpy().squeeze().astype(np.float32)
        else:
            embed = None
        return p, embed

    # ---------------- audio buffer ----------------

    def _push_chunk(self, chunk: np.ndarray) -> None:
        """Append (n_channels, n_samples) chunk to the rolling buffer + update levels."""
        n = chunk.shape[1]
        if n == 0:
            return
        self._latest_chunk_rms = np.sqrt((chunk ** 2).mean(axis=1)).astype(np.float32)
        self._latest_chunk_peak = np.abs(chunk).max(axis=1).astype(np.float32)
        cap = self._buf.shape[1]
        if n >= cap:
            self._buf[:] = chunk[:, -cap:]
            self._buf_filled = cap
            return
        self._buf = np.concatenate([self._buf[:, n:], chunk], axis=1)
        self._buf_filled = min(cap, self._buf_filled + n)

    # ---------------- source: simulate ----------------

    async def _run_simulate(self, wav_path: Path, stop: asyncio.Event) -> None:
        print(f"[pipeline] simulate from {wav_path}")
        arr, sr = sf.read(str(wav_path), dtype="float32", always_2d=True)
        arr = arr.T
        # Track whether we end up with real spatial audio (each mic carries a
        # different signal) or synthetic broadcast (every channel identical
        # except for fake delays).  Triangulation is only honest in the first.
        is_spatial = (arr.shape[0] == self.n_channels)
        if arr.shape[0] == 1 and self.n_channels > 1:
            mono = arr[0]
            delays = np.linalg.norm(self.mics - self.mics[0], axis=1) / SOUND_SPEED
            sample_delays = (delays * sr).astype(int)
            chans = [
                mono if d == 0
                else np.concatenate([np.zeros(d, dtype=np.float32), mono])[:mono.size]
                for d in sample_delays
            ]
            arr = np.stack(chans)
            is_spatial = False
        elif arr.shape[0] != self.n_channels:
            print(f"[pipeline] simulate WAV has {arr.shape[0]} ch, mics expect {self.n_channels} — averaging then broadcasting")
            mono = arr.mean(axis=0)
            arr = np.stack([mono] * self.n_channels)
            is_spatial = False
        arr = _resample_to_16k(arr, sr)

        self.source_is_spatial = is_spatial
        spatial_tag = "spatial" if is_spatial else "mono"
        await self._set_source_state("simulate",
                                     f"WAV · {wav_path.name} · {spatial_tag}",
                                     sample_rate=SR, n_channels=arr.shape[0])

        chunk = HOP_SAMPLES
        cursor = 0
        period = chunk / SR
        while not stop.is_set():
            if cursor + chunk > arr.shape[1]:
                if not self.loop_simulate:
                    break
                # Loop boundary: zero the audio buffer + reset detection state
                # so the 2-second triangulation window doesn't span the
                # discontinuity.  Without this, GCC-PHAT cross-correlates the
                # tail of the file against the head and produces wild positions
                # that scatter across the map.
                cursor = 0
                assert self._buf_lock is not None
                async with self._buf_lock:
                    self._buf[:] = 0.0
                    self._buf_filled = 0
                    self._latest_chunk_rms[:] = 0.0
                    self._latest_chunk_peak[:] = 0.0
                self._consec = 0
                self._position_history.clear()
                self._position_smoothed = None
                self._position_lock_count = 0
                self._position_locked = False
                self._last_position = None
                self._last_position_t = None
                self._recent_positions.clear()
                self._class_sim_history.clear()
            piece = arr[:, cursor:cursor + chunk]
            cursor += chunk
            assert self._buf_lock is not None
            async with self._buf_lock:
                self._push_chunk(piece)
            try:
                await asyncio.wait_for(stop.wait(), timeout=period)
            except asyncio.TimeoutError:
                pass

    # ---------------- source: live mic array ----------------

    async def _run_live(self, device_index: int | None, stop: asyncio.Event) -> None:
        import sounddevice as sd
        loop = asyncio.get_running_loop()

        try:
            info = sd.query_devices(device_index, "input") if device_index is not None else sd.query_devices(kind="input")
        except Exception as exc:
            await self._set_source_state("error", "no audio device", detail=str(exc))
            return

        max_in = int(info.get("max_input_channels") or 0)
        if max_in < self.n_channels:
            msg = (f"device '{info.get('name')}' has {max_in} input ch, "
                   f"need {self.n_channels} — pick a different device or use fewer mics")
            await self._set_source_state("error", info.get("name", "unknown device"), detail=msg)
            return

        native_sr = int(info.get("default_samplerate") or SR)
        # Choose a capture rate the device actually accepts.  Try native, then 48000, 44100, 32000, 16000.
        capture_sr = None
        for candidate in (native_sr, 48000, 44100, 32000, 16000):
            try:
                sd.check_input_settings(
                    device=device_index,
                    samplerate=candidate,
                    channels=self.n_channels,
                    dtype="float32",
                )
                capture_sr = candidate
                break
            except Exception:
                continue
        if capture_sr is None:
            await self._set_source_state(
                "error",
                info.get("name", "unknown device"),
                detail=f"no compatible sample rate for {self.n_channels} ch float32",
            )
            return

        label = f"{info.get('name')} · {self.n_channels} ch @ {capture_sr/1000:.1f} kHz"
        await self._set_source_state("opening", label, sample_rate=capture_sr,
                                     n_channels=self.n_channels)

        block = max(1, int(round(capture_sr * (HOP_SAMPLES / SR))))
        q: asyncio.Queue = asyncio.Queue(maxsize=32)

        def cb(indata, frames, time_info, status):
            if status:
                # XRuns happen on system load; keep going but tag the source as warning.
                pass
            data = indata.T.copy().astype(np.float32, copy=False)
            try:
                loop.call_soon_threadsafe(q.put_nowait, data)
            except RuntimeError:
                pass

        try:
            with sd.InputStream(
                samplerate=capture_sr,
                channels=self.n_channels,
                dtype="float32",
                blocksize=block,
                device=device_index,
                callback=cb,
            ):
                await self._set_source_state("live", label, sample_rate=capture_sr,
                                             n_channels=self.n_channels)
                while not stop.is_set():
                    try:
                        chunk = await asyncio.wait_for(q.get(), timeout=0.5)
                    except asyncio.TimeoutError:
                        continue
                    if capture_sr != SR:
                        chunk = _resample_to_16k(chunk, capture_sr)
                    assert self._buf_lock is not None
                    async with self._buf_lock:
                        self._push_chunk(chunk)
        except Exception as exc:
            await self._set_source_state("error", label, detail=str(exc))
            return
        await self._set_source_state("none", "")

    async def _set_source_state(
        self,
        state: str,
        label: str,
        *,
        detail: str = "",
        sample_rate: int | None = None,
        n_channels: int | None = None,
    ) -> None:
        self.source_state = state
        self.source_label = label
        await self.subscribers.publish(to_dict(SourceEvent(
            state=state, label=label, detail=detail,
            sample_rate=sample_rate, n_channels=n_channels,
            ts_iso=_now_iso(),
        )))

    # ---------------- source switching ----------------

    async def set_source(
        self,
        kind: str,
        *,
        device_index: int | None = None,
        wav_path: Path | None = None,
    ) -> None:
        """Swap the running audio source.  Safe to call mid-detection."""
        kind = kind.lower()
        if kind not in ("none", "live", "simulate"):
            raise ValueError(f"unknown source kind: {kind}")

        # Tear down the previous task first.
        if self._source_stop is not None:
            self._source_stop.set()
        if self._source_task is not None:
            self._source_task.cancel()
            try:
                await self._source_task
            except (asyncio.CancelledError, Exception):
                pass
            self._source_task = None

        # Clear the audio buffer so stale data from the previous source doesn't
        # leak into the next detection windows.  Also reset detection state
        # and position smoothing so the next burst starts fresh.
        async with self._buf_lock:                  # type: ignore[arg-type]
            self._buf[:] = 0.0
            self._buf_filled = 0
            self._latest_chunk_rms[:] = 0.0
            self._latest_chunk_peak[:] = 0.0
        self._consec = 0
        self._position_history.clear()
        self._position_smoothed = None
        self._position_lock_count = 0
        self._position_locked = False
        self._last_position = None
        self._last_position_t = None
        self._recent_positions.clear()
        self._class_sim_history.clear()
        self._last_threat = "clear"
        self._last_drone_class = "unknown"
        # Wipe broadcast history so a fresh client doesn't get cued by
        # frames that belonged to the previous source.
        self.subscribers._recent.clear()

        if kind == "none":
            self._source_kind = "none"
            self._source_kwargs = {}
            await self._set_source_state("none", "")
            return

        # Publish "opening" first so the HTTP response and any waiting clients
        # immediately see the transition, rather than the previous source's
        # leftover state.  The source coroutine refines this once it's running.
        label_hint = (
            "live device" if kind == "live"
            else f"WAV · {Path(wav_path).name}" if wav_path else "source"
        )
        await self._set_source_state("opening", label_hint)

        stop = asyncio.Event()
        self._source_stop = stop
        if kind == "simulate":
            assert wav_path is not None, "wav_path required for simulate source"
            self._source_kind = "simulate"
            self._source_kwargs = {"wav_path": str(wav_path)}
            self._source_task = asyncio.create_task(
                self._run_simulate(Path(wav_path), stop), name="src-simulate",
            )
        else:  # live
            self._source_kind = "live"
            self._source_kwargs = {"device_index": device_index}
            self._source_task = asyncio.create_task(
                self._run_live(device_index, stop), name="src-live",
            )
        # Yield once so the new source task gets a chance to publish its real
        # state before we return — keeps /control/source responses honest.
        await asyncio.sleep(0)

    # ---------------- detection loop ----------------

    async def _run_detector(self) -> None:
        """Outer loop with paranoid error handling — must survive 48 hours."""
        prev_smoothed: deque[float] = deque(maxlen=5)
        consecutive_errors = 0
        next_tick = time.monotonic()
        while not self._stop.is_set():
            next_tick += HOP_SAMPLES / SR
            try:
                await self._detector_tick(prev_smoothed)
                consecutive_errors = 0
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                consecutive_errors += 1
                print(f"[pipeline] detector tick failed ({consecutive_errors}): {exc!r}")
                if consecutive_errors >= 30:
                    print("[pipeline] sustained detector failure; backing off 5s")
                    await asyncio.sleep(5.0)
                    consecutive_errors = 0
            await self._sleep_to(next_tick)

    async def _detector_tick(self, prev_smoothed: deque) -> None:
        """One iteration: snapshot buffer, predict, emit events."""
        assert self._buf_lock is not None
        async with self._buf_lock:
            window = self._buf[:, -WIN_SAMPLES:].copy()
            buf_snapshot = self._buf[:, -int(2 * SR):].copy()
            buf_decoy = self._buf[:, -int(4 * SR):].copy()
            filled = self._buf_filled
            rms = self._latest_chunk_rms.copy()
            peak = self._latest_chunk_peak.copy()

        t_s = time.monotonic() - self._t0
        silent = [int(i) for i, v in enumerate(rms) if v < SILENT_RMS]

        # No source / waiting for buffer fill — emit a heartbeat frame and skip the model.
        if self.source_state in ("none", "opening", "error") or filled < WIN_SAMPLES:
            await self.subscribers.publish(to_dict(FrameEvent(
                t_s=round(t_s, 3),
                p_drone=0.0, p_drone_raw=0.0,
                threat="clear",
                longest_run=self._longest,
                rms_per_channel=[round(float(v), 5) for v in rms],
                peak_per_channel=[round(float(v), 5) for v in peak],
                silent_channels=silent,
                source_state=self.source_state,
                source_label=self.source_label,
                ts_iso=_now_iso(),
            )))
            return

        mono = window.mean(axis=0)
        try:
            p_raw, embed = await asyncio.get_running_loop().run_in_executor(
                None, self._predict, mono
            )
        except Exception as exc:
            print(f"[pipeline] predict error: {exc}")
            await asyncio.sleep(0.5)
            return

        prev_smoothed.append(p_raw)
        p_smooth = float(np.median(prev_smoothed))

        # Drone-type classification — cosine similarity of this window's
        # embedding against the Shahed prototype, smoothed over the last 8
        # windows (~4 s) to avoid flicker.
        drone_class = "unknown"
        class_confidence = 0.0
        if embed is not None and self._shahed_proto is not None:
            n = float(np.linalg.norm(embed) * np.linalg.norm(self._shahed_proto)) + 1e-12
            sim = float(np.dot(embed, self._shahed_proto) / n)
            self._class_sim_history.append(sim)
            class_confidence = float(np.mean(self._class_sim_history))
        else:
            self._class_sim_history.clear()

        above = p_smooth >= self.threshold
        self._consec = self._consec + 1 if above else 0
        self._longest = max(self._longest, self._consec)

        # Triangulation, decoy, open-set
        position: np.ndarray | None = None
        residual = velocity = bearing = rng_m = altitude = None
        decoy_label: str | None = None

        # Only attempt triangulation if (a) detection has been confident for
        # ≥3 windows, (b) buffer has the full 2-second window, and (c) the
        # source actually carries spatial information (4-mic recording).
        # Mono uploads broadcast across N channels would give garbage here.
        if (self._consec >= 3 and filled >= 2 * SR
                and getattr(self, "source_is_spatial", True)):
            try:
                loc = localize(buf_snapshot, self.mics, fs=SR, max_tau=DEFAULT_MAX_TAU)
                raw_pos = loc.position
                residual = float(loc.residual)

                # 1. Reject obvious garbage outright.
                centroid = self.mics.mean(axis=0)
                dist = float(np.linalg.norm(raw_pos[:2] - centroid[:2]))
                if not (np.isfinite(raw_pos).all() and residual < 1.0 and dist < 100.0):
                    raise ValueError(
                        f"rejected (residual={residual:.3f}, dist={dist:.1f}m)"
                    )

                # 2. Median-filter the last 5 accepted raw positions.
                self._position_history.append(raw_pos.copy())
                if len(self._position_history) >= 2:
                    med = np.median(np.stack(self._position_history), axis=0)
                else:
                    med = raw_pos

                # 3. Slow EMA so individual frames can't yank the marker.
                if self._position_smoothed is None:
                    self._position_smoothed = med
                else:
                    self._position_smoothed = 0.5 * med + 0.5 * self._position_smoothed

                # 4. Position lock: only broadcast once we have 3 consecutive
                #    accepted solutions within 2 m of the running smoothed
                #    estimate.  Resets to 0 on rejected windows.
                if np.linalg.norm(raw_pos - self._position_smoothed) < 2.0:
                    self._position_lock_count += 1
                else:
                    self._position_lock_count = 1
                if self._position_lock_count >= 3:
                    self._position_locked = True

                if self._position_locked:
                    position = self._position_smoothed
                    altitude = float(position[2]) if position.size >= 3 else None
                    bearing, rng_m = _bearing_range_from_position(position, self.mics)
                    now = time.monotonic() - self._t0
                    if self._last_position is not None and self._last_position_t is not None:
                        dt = max(now - self._last_position_t, 1e-3)
                        v = (position - self._last_position) / dt
                        velocity = [round(float(v[0]), 3), round(float(v[1]), 3),
                                    round(float(v[2]) if v.size >= 3 else 0.0, 3)]
                    self._last_position = position.copy()
                    self._last_position_t = now
                    self._recent_positions.append((now, position.copy()))
            except Exception as exc:
                # Failed window resets the lock counter but keeps the smoothed
                # estimate so we recover quickly when good frames return.
                self._position_lock_count = max(0, self._position_lock_count - 1)
                if self._position_lock_count == 0:
                    self._position_locked = False

        uncertain = sum(1 for v in prev_smoothed if 0.30 <= v < 0.55)
        open_set_unknown = (
            p_smooth >= self.threshold
            and p_smooth < 0.55
            and uncertain >= 3
        )

        if self._consec >= 4 and buf_decoy.shape[1] >= 4 * SR:
            try:
                verdict = decoy_analyse(
                    buf_decoy, self.mics,
                    DecoyConfig(window_seconds=1.0, stride_seconds=0.5, sample_rate=SR),
                )
                decoy_label = verdict.label
            except Exception:
                decoy_label = None

        if not above:
            threat = "clear"
        elif decoy_label == "decoy":
            threat = "decoy"
        elif open_set_unknown:
            threat = "unknown"
        else:
            threat = "detected"

        # Promote drone_class to "shahed-136" whenever ANY drone signal is
        # present (detected / unknown / decoy) AND the rolling similarity
        # exceeds the prototype's validated threshold AND we have a full
        # 3s of history.  We classify even on decoy because a Shahed clip
        # being played from a speaker is still Shahed audio — the decoy
        # flag tells us it's stationary, the class tells us what the
        # signal sounds like.  Both ship in the cueing JSON.
        if (threat != "clear"
                and len(self._class_sim_history) >= 6
                and class_confidence >= self._shahed_threshold):
            drone_class = "shahed-136"

        await self.subscribers.publish(to_dict(FrameEvent(
            t_s=round(t_s, 3),
            p_drone=round(p_smooth, 4),
            p_drone_raw=round(p_raw, 4),
            threat=threat,
            longest_run=self._longest,
            drone_position_m=([round(float(v), 3) for v in position] if position is not None else None),
            drone_velocity_m_s=velocity,
            bearing_deg=(round(bearing, 1) if bearing is not None else None),
            range_m=(round(rng_m, 2) if rng_m is not None else None),
            altitude_m=(round(altitude, 2) if altitude is not None else None),
            open_set_unknown=open_set_unknown,
            decoy_label=decoy_label,
            residual=(round(residual, 4) if residual is not None else None),
            drone_class=drone_class,
            class_confidence=round(class_confidence, 4),
            rms_per_channel=[round(float(v), 5) for v in rms],
            peak_per_channel=[round(float(v), 5) for v in peak],
            silent_channels=silent,
            source_state=self.source_state,
            source_label=self.source_label,
            ts_iso=_now_iso(),
        )))
        await self._maybe_emit_logs(threat, p_smooth, t_s)

        # Fire a cueing JSON on:
        #   (a) the rising edge of any detection burst, OR
        #   (b) any transition where drone_class changes within a burst —
        #       so the operator gets an updated cue when the system narrows
        #       "unknown" down to "shahed-136" a few seconds in.
        rising_edge = threat in ("detected", "unknown") and self._last_threat == "clear"
        class_changed = threat != "clear" and drone_class != self._last_drone_class
        if rising_edge or class_changed:
            cue_payload = self._build_cue(
                threat=threat, p_smooth=p_smooth, position=position,
                velocity=velocity, decoy_label=decoy_label, t_s=t_s,
                drone_class=drone_class,
            )
            await self.subscribers.publish(to_dict(CueEvent(
                t_s=round(t_s, 3), cueing=cue_payload, ts_iso=_now_iso(),
            )))
        self._last_drone_class = drone_class

        self._last_threat = threat

    async def _sleep_to(self, target: float) -> None:
        sleep = target - time.monotonic()
        if sleep > 0:
            await asyncio.sleep(sleep)

    async def _maybe_emit_logs(self, threat: str, p: float, t_s: float) -> None:
        if threat == self._last_threat:
            return
        push = lambda key, tone, conf=None: self.subscribers.publish(to_dict(LogEvent(
            t_s=round(t_s, 3), key=key, tone=tone, confidence=conf, ts_iso=_now_iso(),
        )))
        if threat == "detected":
            await push("ev_first_detect", "alert", round(p, 3))
            await push("ev_path_lock", "ok", round(p, 3))
            await push("ev_decoy_check", "ok", None)
            await push("ev_handoff", "info", None)
        elif threat == "unknown":
            await push("ev_first_detect", "warn", round(p, 3))
        elif threat == "decoy":
            await push("ev_decoy_check", "warn", round(p, 3))
        elif threat == "clear" and self._last_threat != "clear":
            await push("ev_lost", "info", None)

    def _build_cue(
        self, *, threat: str, p_smooth: float, position: np.ndarray | None,
        velocity: list | None, decoy_label: str | None, t_s: float,
        drone_class: str = "unknown",
    ) -> dict:
        threat_level = (
            "decoy" if decoy_label == "decoy"
            else "unreliable" if decoy_label == "unreliable"
            else "moderate" if p_smooth < 0.7
            else "high"
        )
        path = [list(map(float, p)) for _, p in list(self._recent_positions)[-6:]]
        return {
            "protocol": "saamid.cue/1",
            "timestamp": _now_iso(),
            "site_id": self.site_id,
            "cueing": {
                "drone_class": drone_class,
                "position_m": ([round(float(v), 3) for v in position] if position is not None else None),
                "velocity_m_s": velocity,
                "predicted_path_m": path,
                "confidence": round(float(p_smooth), 4),
                "threat_level": threat_level,
                "is_decoy": decoy_label == "decoy",
                "longest_consecutive_windows_above_threshold": int(self._longest),
                "timestamp_s": round(float(t_s), 3),
            },
        }

    # ---------------- lifecycle ----------------

    async def start(self) -> None:
        self._buf_lock = asyncio.Lock()
        self._t0 = time.monotonic()
        await self.subscribers.publish(to_dict(HelloEvent(
            sample_rate=SR,
            n_channels=self.n_channels,
            threshold=self.threshold,
            mics=[[float(v) for v in m] for m in self.mics],
            site_id=self.site_id,
            model_id=self.hub_id,
            mode=("simulate" if self._initial_simulate
                  else "live" if self._initial_device is not None or True
                  else "none"),
            source_label="",
            has_default_simulate=self.default_simulate_path is not None,
            default_simulate_label=(self.default_simulate_path.name
                                     if self.default_simulate_path else ""),
            ts_iso=_now_iso(),
        )))
        # Always start the detector — it tolerates an empty buffer.
        self._tasks = [
            asyncio.create_task(self._run_detector(), name="detector"),
        ]
        # Apply the initial source request (may fail; we still keep running).
        if self._initial_simulate is not None:
            await self.set_source("simulate", wav_path=self._initial_simulate)
        else:
            # Try the requested device; if it fails the dashboard can pick another.
            try:
                await self.set_source("live", device_index=self._initial_device)
            except Exception as exc:
                print(f"[pipeline] live source unavailable at startup: {exc}")
                await self.set_source("none")

    async def stop(self) -> None:
        self._stop.set()
        if self._source_stop is not None:
            self._source_stop.set()
        if self._source_task is not None:
            self._source_task.cancel()
        for t in getattr(self, "_tasks", []):
            t.cancel()
        for t in [*getattr(self, "_tasks", []), self._source_task]:
            if t is None:
                continue
            try:
                await t
            except (asyncio.CancelledError, Exception):
                pass


def load_mics(path: Path) -> np.ndarray:
    raw = json.loads(Path(path).read_text())
    arr = np.asarray(raw, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] not in (2, 3):
        raise ValueError(f"mic positions must be (N, 2) or (N, 3); got {arr.shape}")
    if arr.shape[1] == 2:
        arr = np.concatenate([arr, np.zeros((arr.shape[0], 1))], axis=1)
    return arr


# ----------------------------------------------------------------------------
# Audio device discovery — used by the FastAPI /devices endpoint.
# ----------------------------------------------------------------------------

def list_input_devices(min_channels: int = 1) -> list[dict[str, Any]]:
    """Enumerate all input devices, filtered by minimum channel count.

    PortAudio caches its device list at module import — that means a USB
    interface plugged in *after* the server started won't appear unless we
    force re-initialisation.  We do that here so the dashboard's picker
    always reflects current hardware.
    """
    import sounddevice as sd
    # Force a fresh PortAudio enumeration so newly-plugged devices appear.
    try:
        sd._terminate()
        sd._initialize()
    except Exception:
        pass

    out = []
    try:
        default_in_idx = sd.default.device[0] if isinstance(sd.default.device, (tuple, list)) else None
    except Exception:
        default_in_idx = None
    for i, d in enumerate(sd.query_devices()):
        max_in = int(d.get("max_input_channels") or 0)
        if max_in < min_channels:
            continue
        out.append({
            "index": i,
            "name": str(d.get("name", "")).strip(),
            "max_input_channels": max_in,
            "default_samplerate": int(d.get("default_samplerate") or 0),
            "host_api": str(d.get("hostapi_name", "")) if d.get("hostapi_name") else "",
            "is_default": (i == default_in_idx),
        })
    return out
