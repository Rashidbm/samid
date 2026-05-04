# Saamid Early Warning System — operations dashboard

Real-time acoustic UAV detection rig for live drone-marathon scoring.
Bilingual Arabic-first dashboard, in-UI device picker, live VU meters per
mic, hot-swappable audio source, live-tunable detection threshold, and a
WebSocket cueing handoff that mirrors `scripts/triangulate.py` exactly.

---

## Drone-marathon day — exact rehearsal script

This is a **live hardware demo**: real drone flying, our mic array on the
ground, our laptop on the table, dashboard projected.  Run through this
end-to-end at least twice before the actual run.

### T-30 minutes — boot the rig

```bash
cd ~/samid
uv sync                                                    # confirm deps
uv run python -m scripts.serve                             # boot the server
```

Open `http://localhost:8000` full-screen on the projector.  Default load
is **Arabic / dark mode** — matches the splash and the deck.  The header
shows a **green pulsing LIVE** badge.

If no 4-channel device is plugged in yet, the source bar is yellow and
the device picker auto-opens.  That's expected.

### T-15 minutes — plug in the mic interface

1. Plug the USB audio interface (4 mics → 1 USB into the laptop).
2. In the source bar, click `Refresh` then click the interface name in
   the picker (e.g. `Behringer UMC404HD · 4 ch · 48 kHz`).
3. Bar turns **green: LIVE · <interface name> · 4 ch @ 48 kHz**.  Four
   VU bars appear.

### T-10 minutes — mic preflight

Tap each mic with a finger one at a time:

- M1 tap → only the M1 VU bar in the source bar AND mic editor lights up
- Same for M2, M3, M4
- If any bar stays grey → **bad cable / wrong channel / phantom power
  off**.  Fix before the drone flies.

### T-5 minutes — measure the array geometry

Measure the actual mic positions in metres relative to mic 1 (front-left).
Type them into the **Microphone Editor** panel on the right side of the
dashboard.  Edits push to the backend automatically (debounced 350 ms) —
triangulation will use the new positions on the next frame.

The default config in `configs/mics_4_square.json` is a 2 m square; if
that's what you set up, no edits needed.

### T-2 minutes — drone flyby

Have someone fly the drone in a slow pass over the array, ~5–15 m above
the centroid.  Watch:

- **VU bars** climb across all four mics
- **`p(drone)` gauge** climbs above the threshold (red arc fills)
- **Threat panel** flips from "خالٍ" / "Clear" to "تم الرصد" / "Detected"
- **Red drone marker** appears on the map, with a fading trail
- **Bearing & Range** panel populates with degrees + metres
- **Event log** fills: `ev_first_detect → ev_path_lock → ev_decoy_check
  → ev_handoff`
- **Cueing JSON** updates with position, velocity, threat_level

### If detection is weak

Use the **threshold slider (θ)** in the source bar.  Drag from 0.25 down
toward 0.10 — pushes a `POST /control/threshold` to the backend on every
change (debounced 150 ms).  The gauge's threshold tick updates in real
time so you can see what level you're at.

### If the source disconnects mid-demo

Source bar turns red.  Click `Pick input device` → re-select the
interface.  Audio resumes, detection resumes — no server restart needed.

---

## Source modes

| Command | What it does |
|---------|--------------|
| `uv run python -m scripts.serve` | Boots the server, tries the OS default input.  If it can't find a 4-ch device, stays alive in `none` state — pick a device from the dashboard. |
| `uv run python -m scripts.serve --device 5` | Pre-selects sounddevice index 5.  Use `--list-devices` first. |
| `uv run python -m scripts.serve --simulate data/abdulrahman/DroneAbdulrahman.wav` | Replays a real-drone phone recording through the pipeline.  Use this for screen-recordings and rehearsal without hardware. |
| `uv run python -m scripts.serve --simulate data/test_4mic.wav` | 4-channel synthetic test.  Triangulation produces realistic positions. |
| open `http://localhost:8000/?demo=1` | Forces the dashboard's offline keyframe demo (no backend needed at all). |

---

## What the dashboard surfaces from the backend

| Element | Source |
|---|---|
| Header `LIVE` / `RECONNECTING` / `DEMO` pill | WebSocket connection state |
| Source bar (state, label, VU meters) | `source` and `frame.rms_per_channel` events |
| Threshold slider (θ) | Two-way: `hello.threshold` initial, `POST /control/threshold` on change |
| `p(drone)` gauge | `frame.p_drone` (smoothed across last 5 windows) |
| Threat panel + alert banner | `frame.threat` (`clear` / `unknown` / `detected` / `decoy`) |
| Map drone marker | `frame.drone_position_m` (median-of-3 + EMA-smoothed in pipeline) |
| Bearing & Range | `frame.bearing_deg`, `frame.range_m` from triangulation |
| Event log | `log` events on threat-state transitions |
| Cueing JSON panel | `cue` events, payload mirrors `scripts/triangulate.py` |
| Mic editor inputs | Two-way: `hello.mics` initial, `POST /control/mics` on edit |
| Per-mic VU bars | `frame.rms_per_channel` (logarithmic dBFS scale, -80→0% / -20→100%) |
| Silent-mic warning | `frame.silent_channels` indices (mic tag turns red) |

---

## API surface

| Path | Method | Purpose |
|---|---|---|
| `/` | GET | Dashboard HTML |
| `/assets/*` | GET | CSS, fonts, logo |
| `/health` | GET | `{ status, source_state, source_label, n_channels, threshold, model_loaded }` |
| `/config` | GET | Mic positions, threshold, model id, current source state |
| `/devices` | GET | All input devices, plus subset with ≥ N channels (forces PortAudio re-enumeration so USB hot-plug works) |
| `/control/source` | POST | `{kind:"live", device_index:N}` / `{kind:"simulate", wav_path:"..."}` / `{kind:"none"}` |
| `/control/mics` | POST | `{ mics: [[x,y,z], ...] }` — count must match `n_channels` |
| `/control/threshold` | POST | `{ threshold: 0.0..1.0 }` |
| `/ws` | WS | Stream of `hello` / `source` / `frame` / `log` / `cue` events |

---

## Hardware compatibility

Anything that shows up in `--list-devices` (or in the dashboard picker)
with at least as many input channels as you have mics will work.  Tested
classes:

- **4-channel USB interfaces** — Behringer UMC404HD, TASCAM US-4×4,
  MOTU M4, Zoom F4/F8 in USB Audio Interface mode
- **Aggregate devices** — macOS Audio MIDI Setup → Aggregate Device
  combining N USB lavaliers; Linux PipeWire `module-combine`; Windows
  VoiceMeeter

The pipeline captures at the device's **native sample rate** (auto-tries
48 / 44.1 / 32 / 16 kHz) and resamples to 16 kHz internally.  No manual
config needed.

---

## Pre-flight checklist (do this before the marathon)

1. **`uv sync`** on the actual demo laptop.
2. **`uv run python -m scripts.serve --simulate data/abdulrahman/DroneAbdulrahman.wav`**
   once — confirms the model weights download and cache.
3. **Pre-grant macOS microphone permission** for the terminal you'll use:
   System Settings → Privacy & Security → Microphone → enable for
   Terminal/iTerm.  First-time popup is blocking and would freeze the demo.
4. **Plug in the actual mic interface.**  Run `uv run python -m
   scripts.serve --list-devices` — the interface name should appear with
   ≥ 4 input channels.
5. **End-to-end mic-tap test** as described in the rehearsal script
   above.  All four bars must light up independently.
6. **End-to-end drone test** if you have access to the venue / a similar
   space — fly the drone, confirm detection peaks above the threshold.

If detection is weak in the actual venue, use the threshold slider.
Don't restart the server.

---

## Cueing JSON shape

The `cue` events on the WebSocket carry the same payload as
`scripts/triangulate.py`.  Subscribe to the same WS and filter
`type == "cue"` to push into a kinetic-defense bus:

```json
{
  "protocol": "saamid.cue/1",
  "timestamp": "2026-05-03T19:42:11.231Z",
  "site_id": "RUH-14",
  "cueing": {
    "drone_class": "unknown",
    "position_m": [11.5, 10.5, 0],
    "velocity_m_s": [3.13, 1.04, 0],
    "predicted_path_m": [...],
    "confidence": 0.98,
    "threat_level": "high",
    "is_decoy": false,
    "longest_consecutive_windows_above_threshold": 12,
    "timestamp_s": 14.2
  }
}
```

---

## Known model performance (real-world recordings)

Verified against every drone recording in the repo using the deployed
checkpoint `Rashidbm/samid-drone-detector` at default threshold 0.25:

| Recording | Description | Smoothed peak p(drone) | Verdict |
|---|---|---|---|
| `data/abdulrahman/DroneAbdulrahman.wav` | Phone mic, real flyby | **0.96** | DRONE DETECTED (69 consecutive windows) |
| `data/test_dji/dji_compilation.wav` | DJI commercial | **0.94** | DRONE DETECTED (17/17 windows) |
| `data/test_new/dji3.wav` | DJI Air | **0.93** | DRONE DETECTED (29 consecutive) |
| `data/test_new/whatsapp_drone.wav` | WhatsApp video clip | **0.96** | DRONE DETECTED (47 consecutive) |
| `data/test_real/test_drone.wav` | Bebop close-mic | **0.82** | DRONE DETECTED |

The model is the post-fix version (after symmetric augmentation +
domain adaptation + frozen-backbone fine-tune).  Real-world peaks are
0.82–0.96 — well above the 0.25 threshold.

---

## Layout

```
samid/
├── dashboard/                 # static site — runs in any browser
│   ├── index.html             # entry point
│   ├── live.js                # WebSocket client + REST helpers + React hook
│   ├── i18n.js                # AR/EN strings + offline demo flight path
│   ├── *.jsx                  # React components (transpiled in-browser)
│   ├── assets/                # CSS, fonts, logo, favicon
│   └── audit.html             # logo overlay test (palette proof)
└── server/
    ├── app.py                 # FastAPI: dashboard + /ws + /devices + /control/*
    ├── pipeline.py            # audio → AST detect → triangulate → decoy → emit
    └── events.py              # event payload dataclasses
```

## Development

The dashboard is React-via-Babel-CDN — no build step.  Edit a JSX file and
refresh.  Backend hot-reloads with:

```bash
uv run python -m scripts.serve --reload --simulate data/test_4mic.wav
```
