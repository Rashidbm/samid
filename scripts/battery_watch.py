"""
Battery-aware supervisor.

  - When battery >= START_PCT and on AC power -> ensure orchestrator running
  - When battery <= STOP_PCT  OR  on battery power -> ensure orchestrator stopped
  - Hysteresis prevents flapping near the threshold.

Design:
  We do NOT run the orchestrator ourselves. We only spawn it when conditions
  are right and kill it (cleanly) when they're not. The orchestrator handles
  everything else (downloads, smoke, training resume).

Run detached:
    nohup uv run python scripts/battery_watch.py >> logs/battery.log 2>&1 < /dev/null &
    echo $! > logs/battery_watch.pid
    disown
"""

from __future__ import annotations
import os
import re
import signal
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LOG_DIR = ROOT / "logs"
LOG_DIR.mkdir(exist_ok=True)

ORCH_PID_FILE = LOG_DIR / "orchestrate.pid"

START_PCT = 80      # only start orchestrator when battery >= 80%
STOP_PCT = 50       # kill orchestrator if battery drops below 50% (safety)
POLL_SECS = 60      # check once a minute

STOP = False


def log(msg: str) -> None:
    print(f"[batt {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def _stop(*_: object) -> None:
    global STOP
    log("stop signal received")
    STOP = True


signal.signal(signal.SIGTERM, _stop)
signal.signal(signal.SIGINT, _stop)


# ----------------------------- battery state -------------------------------- #


def battery_state() -> dict:
    """Return {'pct': int, 'on_ac': bool, 'charging': bool}."""
    try:
        out = subprocess.check_output(["pmset", "-g", "batt"], text=True, timeout=5)
    except Exception as exc:
        log(f"pmset failed: {exc}")
        return {"pct": 0, "on_ac": False, "charging": False}
    on_ac = "AC Power" in out
    m = re.search(r"(\d+)%", out)
    pct = int(m.group(1)) if m else 0
    charging = "charging" in out.lower() or "charged" in out.lower()
    return {"pct": pct, "on_ac": on_ac, "charging": charging}


# ----------------------------- orchestrator ctl ----------------------------- #


def orch_pid() -> int | None:
    if not ORCH_PID_FILE.exists():
        return None
    try:
        pid = int(ORCH_PID_FILE.read_text().strip())
        # validate
        os.kill(pid, 0)
        return pid
    except (OSError, ValueError):
        return None


def start_orchestrator() -> None:
    if orch_pid() is not None:
        return
    log("starting orchestrator (battery healthy)")
    cmd = [
        "/bin/bash", "-c",
        f"cd {ROOT} && nohup uv run python -m scripts.orchestrate "
        f">> logs/orchestrate.log 2>&1 < /dev/null & "
        f"echo $! > logs/orchestrate.pid && disown",
    ]
    subprocess.run(cmd, check=False)
    time.sleep(2)
    pid = orch_pid()
    log(f"orchestrator pid={pid}")


def stop_orchestrator() -> None:
    pid = orch_pid()
    if pid is None:
        return
    log(f"stopping orchestrator pid={pid} (battery low or on battery)")
    try:
        os.kill(pid, signal.SIGTERM)
    except OSError:
        pass
    time.sleep(20)
    if orch_pid() is not None:
        log("forcing kill")
        try:
            os.kill(pid, signal.SIGKILL)
        except OSError:
            pass
        # also nuke children
        subprocess.run(["pkill", "-9", "-f", "scripts.orchestrate"], check=False)
        subprocess.run(["pkill", "-9", "-f", "src.train"], check=False)
        subprocess.run(["pkill", "-9", "-f", "scripts/download_"], check=False)


# ----------------------------- main loop ------------------------------------ #


def main() -> int:
    log(f"watchdog started; START_PCT={START_PCT}, STOP_PCT={STOP_PCT}, poll={POLL_SECS}s")
    last_state = None
    while not STOP:
        st = battery_state()
        running = orch_pid() is not None

        # State summary line, only when something changes.
        snapshot = (st["pct"], st["on_ac"], running)
        if snapshot != last_state:
            log(
                f"battery={st['pct']}% on_ac={st['on_ac']} charging={st['charging']} "
                f"running={running}"
            )
            last_state = snapshot

        # Decision logic with hysteresis.
        if running:
            # Stop if battery is critically low OR we're on battery power.
            if (not st["on_ac"]) or (st["pct"] <= STOP_PCT):
                stop_orchestrator()
        else:
            # Start only if battery is healthy AND on AC power.
            if st["on_ac"] and st["pct"] >= START_PCT:
                start_orchestrator()

        # Sleep with stop responsiveness.
        for _ in range(POLL_SECS):
            if STOP:
                break
            time.sleep(1)

    log("watchdog exiting")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
