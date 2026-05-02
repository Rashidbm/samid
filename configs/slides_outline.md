# Pitch deck — 5 slides

## Slide 1 — The threat

**Title:** Iranian Shahed-136 attacks on Saudi infrastructure

**Visual:** photo of Prince Sultan Air Base damage / news clipping

**Bullets:**
- Iran's Shahed drones have hit Saudi soil; an American serviceman was killed at Prince Sultan
- Existing defenses cost $150,000+ per unit; only protect a few sites
- Acoustic detection is passive, jam-proof, and battle-proven (Ukraine's Sky Map)

**One-liner:** "We built the next generation of acoustic drone defense — Saudi-sovereign, $50 per node."

---

## Slide 2 — How it works (the pipeline)

**Visual:** `configs/visuals/pipeline_diagram.png` or the design HTML diagram

**Bullets (one line each):**
- Microphone audio in (single mic or 4-mic synchronized array)
- Sliding 1-second window with 0.5-second hop
- Audio Spectrogram Transformer detects drone presence per window
- Median filter + 3-consecutive-window rule eliminates false positives
- Multi-mic triangulation locates drone in 3D space
- Decoy detection distinguishes flying drones from speakers faking drone audio
- Open-set detection flags unknown UAV signatures as threats
- JSON cueing handoff to existing defense systems

---

## Slide 3 — Honest evaluation

**Visual:** `configs/visuals/before_after.png` (real-world peak before vs after)

**Bullets:**
- Trained on geronimobasso (180k clips). Got 99.7% F1 on held-out test
- Tested honestly on 5 real-world recordings — model failed (peaks 0.10–0.52)
- Diagnosed shortcut learning (dataset bias)
- Fixed via symmetric augmentation + domain adaptation + frozen-backbone fine-tune
- After fix: real-world peaks 0.82–0.99
- 12-minute fine-tune cut to 2.2 minutes (5.5× speedup)

**Quote line:** "We caught the bias, fixed it, and proved the fix on data the model had never seen."

---

## Slide 4 — Generation 4 vs the field

**Visual:** comparison table

| | Sky Map (Ukraine) | Squarehead (Norway) | Our system |
|---|---|---|---|
| Per-node cost | $400–1000 | $150,000 | **$50** |
| Connectivity | cellular → cloud | network | **edge-only** |
| Drone type ID | mostly binary | binary | **multi-class + open-set** |
| Decoy spoofing | vulnerable | vulnerable | **detected** |
| Cueing latency | 30 s (cloud) | n/a | **<1 s** |
| Battle-proven | yes | no | POC |

**One-liner:** "Sky Map at Prince Sultan is generation 3. We're generation 4."

---

## Slide 5 — Live demo + roadmap

**Demo (90 sec):**
- Play Abdulrahman's recording on speaker → system shows drone-flyby probability rising and falling, peak 99%
- Play DJI compilation → system identifies drone with 17/17 windows above threshold
- Run multi-mic recording through triangulate → shows 3D position + trajectory + decoy check

**Roadmap (text):**
- Now: detection, triangulation, decoy, open-set, cueing — code shipping today
- Next: LoRA per-drone-type classification (DJI / Shahed / custom)
- Next: $5 ESP32 deployment with TinyML distillation
- Next: Saudi field-recording dataset for in-environment tuning

**Closing line:** "Built here, owned here, ready to deploy."

---

## Visuals to have ready (already in repo)

- `configs/visuals/confusion_matrix.png`
- `configs/visuals/metrics_summary.png`
- `configs/visuals/threshold_sweep.png`
- `configs/visuals/loss_curve.png`
- `configs/visuals/abdulrahman_timeline.png`
- `configs/visuals/before_after.png`
- `configs/visuals/overfit_methodology.png`
- `configs/pipeline_diagram.png`
- The Claude design pipeline HTML/PDF/PNG

## Demo audio files to have on hand

- `data/abdulrahman/DroneAbdulrahman.wav` — real phone recording, full flyby
- `data/test_dji/dji_compilation.wav` — DJI commercial drones compilation
- `data/test_new/whatsapp_drone.wav` — WhatsApp video drone clip
- `data/test_new/dji3.wav` — DJI Air recording
- `data/test_real/test_drone.wav` — Bebop close-mic
- `data/test_4mic.wav` — 4-channel synthetic for triangulation demo

## One-paragraph elevator pitch (memorize)

> Iran's Shahed-136 has attacked Saudi infrastructure. Sky Map at Prince Sultan
> is great but it's foreign, $400 per node, requires the cloud, and can't
> identify drone types or detect decoys. Our system runs on $50 of hardware,
> works offline, identifies drones, triangulates their position with a 4-mic
> array, distinguishes real drones from stationary speaker decoys, and flags
> unknown UAV signatures as threats by default. We trained on 180,000 audio
> samples, tested honestly on real-world recordings, caught a dataset bias,
> fixed it with domain adaptation, and verified end-to-end. Built in 10 days,
> owned by Saudi Arabia, ready to deploy.
