"""
Top-level supervisor for the whole pipeline.

Loops:
  1. (re)launches geronimobasso download until the dataset is fully present.
  2. (re)launches MMAUD download in parallel until all 5 zips finish.
  3. Once geronimobasso is ready, runs smoke_test once.
  4. Then runs training with --auto-resume on a restart loop:
       - if it exits non-zero (crash, OOM, kernel restart), it relaunches
       - resumes from the latest checkpoint each time.
  5. Network outages → all child processes retry; orchestrator never dies.

This script is designed to be run with nohup so it survives shell exit:
    nohup uv run python -m scripts.orchestrate > orchestrate.log 2>&1 &
    echo $! > orchestrate.pid

Stop it cleanly with:
    kill $(cat orchestrate.pid)
"""

from __future__ import annotations
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LOG_DIR = ROOT / "logs"
LOG_DIR.mkdir(exist_ok=True)
STOP = False


def log(msg: str) -> None:
    print(f"[orch {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def _stop(*_: object) -> None:
    global STOP
    log("stop signal received")
    STOP = True


signal.signal(signal.SIGTERM, _stop)
signal.signal(signal.SIGINT, _stop)


# --------------------------- readiness probes ------------------------------- #


def geronimobasso_ready() -> bool:
    """Cheap probe — checks for marker written by download_geronimobasso.py."""
    return (ROOT / "data" / ".geronimobasso.done").exists()


def mmaud_ready() -> bool:
    expected = {"Mavic2.zip", "Mavic3.zip", "Phantom4.zip", "Avata.zip", "M300.zip"}
    present = {p.name for p in (ROOT / "data" / "mmaud").glob("*.zip")}
    return expected.issubset(present)


# --------------------------- subprocess helpers ----------------------------- #


def run_step(name: str, cmd: list[str], log_path: Path,
             max_retries: int | None = None,
             pause_on_fail: int = 30) -> int:
    """
    Run a command; if it fails, retry with exponential backoff.
    Returns last return code (0 if eventually succeeded).
    """
    attempt = 0
    while not STOP:
        attempt += 1
        log(f"[{name}] attempt {attempt} -> {' '.join(cmd)}")
        with log_path.open("ab") as fp:
            fp.write(f"\n----- {name} attempt {attempt} @ {time.ctime()} -----\n".encode())
            proc = subprocess.Popen(cmd, cwd=str(ROOT), stdout=fp, stderr=subprocess.STDOUT)
            try:
                rc = proc.wait()
            except KeyboardInterrupt:
                proc.terminate()
                proc.wait()
                return 130
        log(f"[{name}] exit {rc}")
        if rc == 0:
            return 0
        if max_retries is not None and attempt >= max_retries:
            log(f"[{name}] giving up after {attempt} attempts")
            return rc
        wait = min(pause_on_fail * (2 ** (attempt - 1)), 600)
        log(f"[{name}] retrying in {wait}s")
        for _ in range(wait):
            if STOP:
                return rc
            time.sleep(1)
    return 0


def run_in_background(name: str, cmd: list[str], log_path: Path) -> subprocess.Popen:
    log(f"[{name}] launching background -> {' '.join(cmd)}")
    fp = log_path.open("ab")
    fp.write(f"\n----- {name} launched @ {time.ctime()} -----\n".encode())
    return subprocess.Popen(cmd, cwd=str(ROOT), stdout=fp, stderr=subprocess.STDOUT)


# ---------------------------- main loop ------------------------------------- #


def main() -> int:
    # Step 1+2: launch BOTH downloads in parallel; restart them if they die.
    procs: dict[str, subprocess.Popen] = {}

    def ensure_running(name: str, cmd: list[str], log_path: Path,
                       ready_fn) -> None:
        # If the dataset is ready, don't launch.
        if ready_fn():
            log(f"[{name}] already ready, no download needed")
            procs.pop(name, None)
            return
        # If we have a live process, leave it.
        proc = procs.get(name)
        if proc is not None and proc.poll() is None:
            return
        # Otherwise (re)spawn.
        procs[name] = run_in_background(name, cmd, log_path)

    # Smoke + train state machine
    smoke_done = False
    train_started = False
    train_proc: subprocess.Popen | None = None

    last_status = 0.0
    while not STOP:
        ensure_running(
            "dl_geronimobasso",
            [sys.executable, "scripts/download_geronimobasso.py"],
            LOG_DIR / "dl_geronimobasso.log",
            geronimobasso_ready,
        )
        ensure_running(
            "dl_mmaud",
            [sys.executable, "scripts/download_mmaud.py"],
            LOG_DIR / "dl_mmaud.log",
            mmaud_ready,
        )

        # Once geronimobasso is ready, do smoke test once, then start training.
        if geronimobasso_ready():
            if not smoke_done:
                rc = run_step(
                    "smoke",
                    [sys.executable, "-m", "scripts.smoke_test"],
                    LOG_DIR / "smoke.log",
                    max_retries=3,
                )
                if rc == 0:
                    smoke_done = True
                    log("[smoke] PASSED")
                else:
                    log("[smoke] FAILED 3x — pausing 5 min before retry")
                    time.sleep(300)
                    continue

            if smoke_done and not train_started:
                # Use --auto-resume so subsequent restarts pick up the latest ckpt.
                train_proc = run_in_background(
                    "train",
                    [sys.executable, "-m", "src.train", "--auto-resume"],
                    LOG_DIR / "train.log",
                )
                train_started = True

            # If training process died with non-zero, relaunch via auto-resume.
            if train_proc is not None and train_proc.poll() is not None:
                rc = train_proc.returncode
                if rc == 0:
                    log("[train] finished cleanly. Done.")
                    break
                log(f"[train] crashed with rc={rc} — relaunching with auto-resume in 30s")
                time.sleep(30)
                train_proc = run_in_background(
                    "train",
                    [sys.executable, "-m", "src.train", "--auto-resume"],
                    LOG_DIR / "train.log",
                )

        # Periodic status line
        if time.time() - last_status > 60:
            last_status = time.time()
            geron_ok = geronimobasso_ready()
            mmaud_ok = mmaud_ready()
            log(f"status: geron={geron_ok} mmaud={mmaud_ok} "
                f"smoke={smoke_done} train={'running' if train_proc and train_proc.poll() is None else 'idle'}")
        time.sleep(15)

    # On stop signal, terminate children gracefully.
    for name, p in list(procs.items()) + ([("train", train_proc)] if train_proc else []):
        if p is not None and p.poll() is None:
            log(f"terminating {name} pid={p.pid}")
            p.terminate()
    for name, p in list(procs.items()) + ([("train", train_proc)] if train_proc else []):
        if p is not None:
            try:
                p.wait(timeout=20)
            except subprocess.TimeoutExpired:
                p.kill()
    return 0


if __name__ == "__main__":
    sys.exit(main())
