"""FastAPI app: serves the dashboard static files and pushes live events
over a WebSocket.  Configured at startup via environment variables (see
scripts/serve.py) so this module stays import-cheap.
"""
from __future__ import annotations
import asyncio
import json
import os
from pathlib import Path

import shutil
import subprocess
import time

from fastapi import Body, FastAPI, File, HTTPException, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from .pipeline import Pipeline, load_mics, list_input_devices


REPO_ROOT = Path(__file__).resolve().parent.parent
DASHBOARD_DIR = REPO_ROOT / "dashboard"
UPLOAD_DIR = Path("/tmp/saamid-uploads")
UPLOAD_DIR.mkdir(exist_ok=True)
MAX_UPLOAD_BYTES = 50 * 1024 * 1024     # 50 MB ceiling — protects HF Spaces' 2 GB ephemeral disk
RAW_AUDIO_EXTS = {".wav", ".flac", ".ogg"}
CONVERTIBLE_AUDIO_EXTS = {".mp3", ".m4a", ".aac", ".opus", ".webm", ".mp4", ".mov", ".caf"}


def _env_path(key: str, default: str) -> Path:
    return Path(os.environ.get(key, default)).expanduser().resolve()


def build_app() -> FastAPI:
    mics_path = _env_path("SAAMID_MICS", str(REPO_ROOT / "configs" / "mics_4_square.json"))
    threshold = float(os.environ.get("SAAMID_THRESHOLD", "0.25"))
    site_id = os.environ.get("SAAMID_SITE_ID", "RUH-14")
    hub_id = os.environ.get("SAAMID_HUB_ID", "Rashidbm/samid-drone-detector")
    simulate = os.environ.get("SAAMID_SIMULATE", "")
    device_idx_raw = os.environ.get("SAAMID_DEVICE", "")
    device_idx = int(device_idx_raw) if device_idx_raw.strip() else None

    mics = load_mics(mics_path)
    pipeline = Pipeline(
        mics=mics,
        hub_id=hub_id,
        threshold=threshold,
        site_id=site_id,
        simulate_wav=Path(simulate).expanduser().resolve() if simulate else None,
        device_index=device_idx,
    )

    app = FastAPI(title="Saamid Early Warning System")
    app.state.pipeline = pipeline

    @app.on_event("startup")
    async def _startup() -> None:
        # Load model in a worker thread so the HTTP port comes up fast and
        # the splash screen renders while weights are still downloading.
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, pipeline.load_model)
        await pipeline.start()

    @app.on_event("shutdown")
    async def _shutdown() -> None:
        await pipeline.stop()

    @app.get("/health")
    async def health() -> dict:
        return {
            "status": "ok",
            "source_state": pipeline.source_state,
            "source_label": pipeline.source_label,
            "n_channels": pipeline.n_channels,
            "threshold": pipeline.threshold,
            "model_loaded": pipeline._model is not None,
        }

    @app.get("/config")
    async def config() -> dict:
        return {
            "sample_rate": 16_000,
            "n_channels": pipeline.n_channels,
            "threshold": pipeline.threshold,
            "site_id": pipeline.site_id,
            "model_id": pipeline.hub_id,
            "mics": [[float(v) for v in m] for m in pipeline.mics],
            "source_state": pipeline.source_state,
            "source_label": pipeline.source_label,
        }

    @app.get("/devices")
    async def devices() -> dict:
        """List all input devices that can satisfy the current mic count."""
        try:
            return {
                "n_channels_required": pipeline.n_channels,
                "devices": list_input_devices(min_channels=pipeline.n_channels),
                "all_devices": list_input_devices(min_channels=1),
            }
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"device enumeration failed: {exc}")

    @app.post("/control/mics")
    async def control_mics(body: dict = Body(...)) -> dict:
        """Update mic positions live.  Channel count must stay constant.

        Body: { "mics": [[x, y, z], [x, y, z], ...] }   (z optional)
        """
        raw = body.get("mics")
        if not isinstance(raw, list) or not raw:
            raise HTTPException(status_code=400, detail="mics must be a non-empty list")
        try:
            arr = []
            for m in raw:
                if not isinstance(m, (list, tuple)) or len(m) not in (2, 3):
                    raise ValueError(f"each mic must be [x, y] or [x, y, z]; got {m!r}")
                xs = list(map(float, m))
                if len(xs) == 2:
                    xs.append(0.0)
                arr.append(xs)
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc))
        if len(arr) != pipeline.n_channels:
            raise HTTPException(
                status_code=400,
                detail=(f"mic count {len(arr)} doesn't match channel count "
                        f"{pipeline.n_channels} — restart server with --mics for a different array"),
            )
        import numpy as np
        pipeline.mics = np.asarray(arr, dtype=np.float64)
        return {"ok": True, "mics": arr, "n_channels": pipeline.n_channels}

    @app.post("/control/threshold")
    async def control_threshold(body: dict = Body(...)) -> dict:
        """Adjust the detection threshold live."""
        try:
            val = float(body.get("threshold"))
        except (TypeError, ValueError):
            raise HTTPException(status_code=400, detail="threshold must be a number")
        if not (0.0 < val < 1.0):
            raise HTTPException(status_code=400, detail="threshold must be in (0, 1)")
        pipeline.threshold = val
        return {"ok": True, "threshold": val}

    @app.post("/control/upload")
    async def control_upload(file: UploadFile = File(...)) -> dict:
        """Accept an audio file from the dashboard, transcode if needed,
        and switch the simulate source to it.  Replaces any prior upload.

        Used by the public demo: judges hit the URL, click "Upload audio",
        pick their own drone recording (WAV / MP3 / M4A / etc.), and watch
        the live pipeline run on it.
        """
        if not file.filename:
            raise HTTPException(status_code=400, detail="missing filename")
        suffix = Path(file.filename).suffix.lower()
        if suffix not in RAW_AUDIO_EXTS and suffix not in CONVERTIBLE_AUDIO_EXTS:
            raise HTTPException(
                status_code=415,
                detail=f"unsupported format {suffix!r}; accepted: "
                       f"{sorted(RAW_AUDIO_EXTS | CONVERTIBLE_AUDIO_EXTS)}",
            )

        # Stream to disk with a hard size cap.
        stamp = int(time.time() * 1000)
        raw_path = UPLOAD_DIR / f"upload-{stamp}{suffix}"
        bytes_written = 0
        try:
            with raw_path.open("wb") as out:
                while True:
                    chunk = await file.read(64 * 1024)
                    if not chunk:
                        break
                    bytes_written += len(chunk)
                    if bytes_written > MAX_UPLOAD_BYTES:
                        out.close()
                        raw_path.unlink(missing_ok=True)
                        raise HTTPException(
                            status_code=413,
                            detail=f"upload too large; max {MAX_UPLOAD_BYTES // 1024 // 1024} MB",
                        )
                    out.write(chunk)
        except HTTPException:
            raise
        except Exception as exc:
            raw_path.unlink(missing_ok=True)
            raise HTTPException(status_code=500, detail=f"failed to save upload: {exc}")

        # Transcode anything ffmpeg understands → WAV that soundfile can read.
        if suffix in CONVERTIBLE_AUDIO_EXTS:
            if shutil.which("ffmpeg") is None:
                raw_path.unlink(missing_ok=True)
                raise HTTPException(status_code=500, detail="ffmpeg not installed; cannot transcode")
            wav_path = raw_path.with_suffix(".wav")
            try:
                subprocess.run(
                    ["ffmpeg", "-y", "-loglevel", "error",
                     "-i", str(raw_path), str(wav_path)],
                    check=True, timeout=60,
                )
            except subprocess.CalledProcessError as exc:
                raw_path.unlink(missing_ok=True)
                raise HTTPException(status_code=400,
                                    detail=f"ffmpeg failed to decode upload: {exc}")
            except subprocess.TimeoutExpired:
                raw_path.unlink(missing_ok=True)
                raise HTTPException(status_code=400,
                                    detail="ffmpeg transcode timed out (file > 60s of work?)")
            raw_path.unlink(missing_ok=True)
            play_path = wav_path
        else:
            play_path = raw_path

        # Tidy up older uploads — keep only the file we're about to play.
        for old in UPLOAD_DIR.glob("upload-*"):
            if old.resolve() != play_path.resolve():
                try: old.unlink()
                except Exception: pass

        try:
            await pipeline.set_source("simulate", wav_path=play_path)
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"source switch failed: {exc}")

        return {
            "ok": True,
            "filename": file.filename,
            "bytes": bytes_written,
            "playing": play_path.name,
            "source_state": pipeline.source_state,
            "source_label": pipeline.source_label,
        }

    @app.post("/control/upload_array")
    async def control_upload_array(files: list[UploadFile] = File(...)) -> dict:
        """Accept N audio files (one per microphone) and stack them into a
        single multi-channel WAV so the simulate pipeline gets honest spatial
        audio instead of broadcast-mono.

        File order matters: index i in the upload corresponds to mic i in
        configs/mics_*.json (mic 1 first, etc.).  The dashboard's per-mic
        upload modal enforces this ordering.

        Files may be at different sample rates / lengths — we resample each
        to 16 kHz and crop to the shortest duration.
        """
        import numpy as np
        import soundfile as sf

        n_required = pipeline.n_channels
        if len(files) != n_required:
            raise HTTPException(
                status_code=400,
                detail=f"need exactly {n_required} files (one per mic), got {len(files)}",
            )

        saved: list[Path] = []
        try:
            # 1. Save each upload, transcoding non-WAV inputs via ffmpeg.
            for i, f in enumerate(files):
                if not f.filename:
                    raise HTTPException(status_code=400, detail=f"file {i+1}: missing filename")
                suffix = Path(f.filename).suffix.lower()
                if suffix not in RAW_AUDIO_EXTS and suffix not in CONVERTIBLE_AUDIO_EXTS:
                    raise HTTPException(
                        status_code=415,
                        detail=f"file {i+1} ({f.filename}): unsupported format {suffix!r}",
                    )

                stamp = int(time.time() * 1000)
                raw = UPLOAD_DIR / f"upload-arr-{stamp}-mic{i+1}{suffix}"
                bytes_written = 0
                with raw.open("wb") as out:
                    while True:
                        chunk = await f.read(64 * 1024)
                        if not chunk:
                            break
                        bytes_written += len(chunk)
                        if bytes_written > MAX_UPLOAD_BYTES:
                            out.close()
                            raw.unlink(missing_ok=True)
                            raise HTTPException(
                                status_code=413,
                                detail=f"file {i+1}: max {MAX_UPLOAD_BYTES // 1024 // 1024} MB per file",
                            )
                        out.write(chunk)

                if suffix in CONVERTIBLE_AUDIO_EXTS:
                    if shutil.which("ffmpeg") is None:
                        raw.unlink(missing_ok=True)
                        raise HTTPException(
                            status_code=500,
                            detail="ffmpeg not installed; cannot transcode",
                        )
                    wav = raw.with_suffix(".wav")
                    try:
                        subprocess.run(
                            ["ffmpeg", "-y", "-loglevel", "error",
                             "-i", str(raw), str(wav)],
                            check=True, timeout=60,
                        )
                    except subprocess.CalledProcessError as exc:
                        raw.unlink(missing_ok=True)
                        raise HTTPException(
                            status_code=400,
                            detail=f"file {i+1} ({f.filename}): ffmpeg decode failed",
                        )
                    raw.unlink(missing_ok=True)
                    saved.append(wav)
                else:
                    saved.append(raw)

            # 2. Read each, take channel 0 (judges might submit stereo per-mic
            #    by accident — that's fine, we just want the first channel).
            tracks: list[tuple[np.ndarray, int]] = []
            for i, p in enumerate(saved):
                arr, sr = sf.read(str(p), dtype="float32", always_2d=True)
                if arr.shape[1] > 1:
                    arr = arr[:, :1]   # collapse to mono per mic
                tracks.append((arr.flatten(), int(sr)))

            # 3. Resample everything to 16 kHz.
            target_sr = 16_000
            from math import gcd
            from scipy.signal import resample_poly
            resampled: list[np.ndarray] = []
            for arr, sr in tracks:
                if sr != target_sr:
                    g = gcd(sr, target_sr)
                    arr = resample_poly(arr, target_sr // g, sr // g).astype(np.float32, copy=False)
                resampled.append(arr.astype(np.float32, copy=False))

            # 4. Crop to shortest, stack into (n_channels, samples).
            min_len = min(len(a) for a in resampled)
            if min_len < target_sr:        # need ≥1 second to do anything useful
                raise HTTPException(
                    status_code=400,
                    detail=f"shortest file is {min_len/target_sr:.2f}s — need ≥ 1s",
                )
            stacked = np.stack([a[:min_len] for a in resampled])

            # 5. Write the assembled multi-channel WAV.
            stamp = int(time.time() * 1000)
            out_path = UPLOAD_DIR / f"upload-arr-{stamp}-stacked.wav"
            sf.write(str(out_path), stacked.T, target_sr, subtype="PCM_16")

            # 6. Cleanup component files + older uploads.
            for p in saved:
                try: p.unlink()
                except Exception: pass
            for old in UPLOAD_DIR.glob("upload-*"):
                if old.resolve() != out_path.resolve():
                    try: old.unlink()
                    except Exception: pass

            # 7. Switch source.
            await pipeline.set_source("simulate", wav_path=out_path)

            return {
                "ok": True,
                "n_files": len(files),
                "filenames": [f.filename for f in files],
                "playing": out_path.name,
                "sample_rate": target_sr,
                "samples_per_channel": int(min_len),
                "duration_s": round(min_len / target_sr, 2),
                "source_state": pipeline.source_state,
                "source_label": pipeline.source_label,
            }
        except HTTPException:
            for p in saved:
                try: p.unlink()
                except Exception: pass
            raise
        except Exception as exc:
            for p in saved:
                try: p.unlink()
                except Exception: pass
            raise HTTPException(status_code=500, detail=f"array upload failed: {exc}")

    @app.post("/control/reset_default")
    async def control_reset_default() -> dict:
        """Switch the simulate source back to whatever WAV the server was
        started with (the SAAMID_SIMULATE env var)."""
        if pipeline.default_simulate_path is None:
            raise HTTPException(status_code=400, detail="no default simulate WAV configured")
        if not pipeline.default_simulate_path.exists():
            raise HTTPException(status_code=404,
                                detail=f"default WAV missing: {pipeline.default_simulate_path}")
        try:
            await pipeline.set_source("simulate", wav_path=pipeline.default_simulate_path)
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"reset failed: {exc}")
        return {"ok": True, "playing": pipeline.default_simulate_path.name}

    @app.post("/control/source")
    async def control_source(body: dict = Body(...)) -> dict:
        """Hot-swap the audio source.

        Body shapes:
          { "kind": "live", "device_index": 2 }
          { "kind": "simulate", "wav_path": "data/test_4mic.wav" }
          { "kind": "none" }
        """
        kind = str(body.get("kind", "")).lower()
        if kind not in ("none", "live", "simulate"):
            raise HTTPException(status_code=400, detail=f"unknown kind: {kind!r}")
        try:
            if kind == "simulate":
                wav = body.get("wav_path") or body.get("wav")
                if not wav:
                    raise HTTPException(status_code=400, detail="wav_path required")
                p = Path(wav).expanduser().resolve()
                if not p.exists():
                    raise HTTPException(status_code=404, detail=f"wav not found: {p}")
                await pipeline.set_source("simulate", wav_path=p)
            elif kind == "live":
                idx = body.get("device_index")
                if idx is not None:
                    idx = int(idx)
                await pipeline.set_source("live", device_index=idx)
            else:
                await pipeline.set_source("none")
            return {
                "ok": True,
                "source_state": pipeline.source_state,
                "source_label": pipeline.source_label,
            }
        except HTTPException:
            raise
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"source switch failed: {exc}")

    @app.websocket("/ws")
    async def ws(socket: WebSocket) -> None:
        await socket.accept()
        q = await pipeline.subscribers.add()
        try:
            while True:
                event = await q.get()
                await socket.send_text(json.dumps(event, default=float))
        except WebSocketDisconnect:
            pass
        except Exception as exc:
            print(f"[ws] {exc}")
        finally:
            await pipeline.subscribers.remove(q)

    # Static dashboard.  We don't mount at "/" directly so /health, /config,
    # /ws keep priority; index.html is served on GET /.
    if DASHBOARD_DIR.exists():
        app.mount("/assets", StaticFiles(directory=DASHBOARD_DIR / "assets"), name="assets")

        @app.get("/")
        async def root() -> FileResponse:
            return FileResponse(DASHBOARD_DIR / "index.html")

        @app.get("/{path:path}")
        async def static_passthrough(path: str) -> FileResponse:
            f = DASHBOARD_DIR / path
            if f.is_file():
                return FileResponse(f)
            return JSONResponse({"detail": "not found"}, status_code=404)
    else:
        @app.get("/")
        async def root() -> dict:
            return {"detail": f"dashboard dir missing: {DASHBOARD_DIR}"}

    return app


app = build_app()
