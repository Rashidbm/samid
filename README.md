---
title: Saamid Early Warning System
emoji: 🛸
colorFrom: green
colorTo: yellow
sdk: docker
app_port: 7860
pinned: false
short_description: Acoustic UAV detection, triangulation & live cueing handoff
---

# Saamid Early Warning System
**منظومة صامد للانذار المبكر** — acoustic drone detection, triangulation, and live cueing handoff for the Saudi sovereign defense stack.

AST classifier fine-tuned on geronimobasso + NUS DroneAudioSet with domain
adaptation on real-world recordings.  Trained model:
[Rashidbm/samid-drone-detector](https://huggingface.co/Rashidbm/samid-drone-detector).

Real-world detection peaks (deployed checkpoint, threshold 0.25):

| Recording | Description | Peak p(drone) |
|---|---|---|
| `data/abdulrahman/DroneAbdulrahman.wav` | Phone mic, real flyby | **0.96** |
| `data/test_dji/dji_compilation.wav` | DJI commercial drones | **0.94** |
| `data/test_new/dji3.wav` | DJI Air | **0.93** |
| `data/test_new/whatsapp_drone.wav` | WhatsApp clip | **0.96** |
| `data/test_real/test_drone.wav` | Bebop close-mic | **0.82** |

---

## Deploy & run

There are four ways to bring this up depending on what hardware you have
and where you want it to run.  Pick one.

### A. One-command local install (the marathon path)

For running on a laptop with a USB audio interface plugged in.

```bash
git clone <this-repo> samid && cd samid
./scripts/bootstrap.sh                  # installs uv + system audio libs +
                                        # caches the model + runs a self-test
uv run python -m scripts.serve          # boot the server
```

Open [http://localhost:8000](http://localhost:8000), pick your USB audio
interface from the source bar, play a drone clip or fly the actual drone.

The bootstrap script is idempotent — re-run it anytime.  After it succeeds,
the model weights are cached locally so the demo works **without internet**.

**macOS only:** pre-grant microphone permission to your terminal in
System Settings → Privacy & Security → Microphone.  First-time popup is
blocking and would freeze the demo.

### B. Docker (portable, any platform)

```bash
docker compose --profile demo up        # simulate mode, no hardware needed
# → http://localhost:8000

docker compose --profile live up        # Linux only — passes /dev/snd through
                                        # to the container for real mic capture
```

Or, build and run the image directly:

```bash
docker build -t saamid .
docker run --rm -p 8000:8000 saamid     # demo mode (default env vars)
```

The Dockerfile bakes in the model weights and audio libraries, so the
container boots offline.  ~1.6 GB image, ~600 MB compressed.

### C. Public cloud demo (Fly.io)

Drops a live, internet-reachable URL serving the dashboard in simulate
mode — useful for sharing the system with people who can't run it
themselves.  Audio capture isn't possible in the cloud, so this loops a
real drone phone recording through the full pipeline.

```bash
brew install flyctl
flyctl auth login
flyctl launch --no-deploy --copy-config --name <your-app-name>
flyctl deploy
# → https://<your-app-name>.fly.dev
```

Config lives in `fly.toml`.  `auto_stop_machines = "stop"` keeps cost near
zero when nobody's watching.

### D. Local hardware + public URL (tunnel)

If you want **real mic capture** AND **a public URL** (e.g. for remote
judging while the rig is in front of you), boot locally then expose the
port with a tunnel:

```bash
# terminal 1 — local backend with real mics
uv run python -m scripts.serve

# terminal 2 — tunnel
brew install cloudflared
cloudflared tunnel --url http://localhost:8000
# → free public https://random-words.trycloudflare.com URL
```

Or with ngrok:

```bash
ngrok http 8000
```

WebSocket survives both tunnels.  Latency adds ~50–100 ms over the LAN
path but is invisible to the operator.

---

## What's in the box

```
samid/
├── dashboard/              # bilingual AR/EN dashboard (React-via-CDN)
│   ├── README.md           # marathon-day rehearsal script lives here
│   ├── index.html          # entry point
│   ├── live.js             # WebSocket client + REST helpers
│   ├── *.jsx               # threat panel, gauge, map, mic editor, cueing JSON
│   └── assets/             # CSS, fonts, logo, favicon
├── server/                 # FastAPI backend
│   ├── app.py              # routes: /, /ws, /devices, /control/*
│   ├── pipeline.py         # audio → AST → triangulate → decoy → emit
│   └── events.py           # event payload dataclasses
├── src/                    # research + training code
│   ├── triangulation.py    # GCC-PHAT + least-squares localisation
│   ├── decoy.py            # stationary-source heuristic
│   ├── openset.py          # FDBD / energy / max-logit OOD scores
│   ├── train.py            # training loop
│   └── data.py, model.py, losses.py, metrics.py, augment.py
├── scripts/
│   ├── bootstrap.sh        # one-command local setup
│   ├── serve.py            # CLI for the FastAPI backend
│   ├── standalone_inference.py    # no-server inference for one file
│   ├── triangulate.py      # multi-mic localisation CLI
│   ├── finetune_abdulrahman.py    # domain adaptation on real recordings
│   ├── push_to_hub.py      # publish model to HF
│   └── download_geronimobasso.py
├── configs/
│   └── mics_4_square.json  # default 2 m square 4-mic geometry
├── data/                   # demo & test recordings (real drones included)
├── Dockerfile              # portable runtime image
├── docker-compose.yml      # demo / live profiles
└── fly.toml                # optional Fly.io public deployment
```

---

## API surface (for integration with defense systems)

The backend speaks the same cueing JSON that `scripts/triangulate.py`
emits, broadcast over a WebSocket.

| Path | Method | Purpose |
|---|---|---|
| `/` | GET | Dashboard HTML |
| `/health` | GET | `{ status, source_state, source_label, n_channels, threshold, model_loaded }` |
| `/config` | GET | Mic positions, threshold, model id, current source state |
| `/devices` | GET | All input devices, plus subset with ≥ N channels |
| `/control/source` | POST | Hot-swap audio source (live device / WAV / none) |
| `/control/mics` | POST | Update mic positions live (count must match) |
| `/control/threshold` | POST | Adjust detection threshold live |
| `/ws` | WS | Stream of `hello` / `source` / `frame` / `log` / `cue` events |

Cueing JSON shape:

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

Subscribe to the WS, filter `type == "cue"`, push into your kinetic-defense
bus.  Full schema in `server/events.py`.

---

## CLI commands

```bash
# Inference on one audio file (no server)
uv run python scripts/standalone_inference.py --wav your_clip.mp3 --threshold 0.25

# Live mic inference (no server, no dashboard)
uv run python scripts/standalone_inference.py

# Multi-mic source localisation on a multi-channel WAV
uv run python -m scripts.triangulate --wav multichannel.wav --mics configs/mics_4_square.json

# Backend with a specific audio device
uv run python -m scripts.serve --list-devices
uv run python -m scripts.serve --device 5

# Backend in simulate mode (no hardware)
uv run python -m scripts.serve --simulate data/abdulrahman/DroneAbdulrahman.wav
```

---

## Training

```bash
uv run python scripts/download_geronimobasso.py
uv run python -m src.train
uv run python scripts/finetune_abdulrahman.py --abd-wav your_recording.wav
uv run python scripts/push_to_hub.py --repo username/samid-drone-detector
```

---

## Documentation

- **Dashboard + marathon-day rehearsal script** → [`dashboard/README.md`](dashboard/README.md)
- **Pitch deck outline** → [`configs/slides_outline.md`](configs/slides_outline.md)

---

## License

MIT
