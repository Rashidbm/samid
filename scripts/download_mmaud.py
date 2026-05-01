"""
Download MMAUD V1 'Folder Data' zips (pre-extracted, no ROS needed).

V1 = Rooftop Simple flights, 5 drone types.
Each .zip contains the audio + sensor data already extracted.
We skip the rosbags (.bag) — they need ROS to read.

Source page: https://ntu-aris.github.io/MMAUD/
License: CC-BY-NC-SA 4.0 (academic / non-commercial — fine for hackathon POC)

OneDrive direct download trick: append `&download=1` to the share URL.
"""

from __future__ import annotations
from pathlib import Path
import sys
import time

import requests
from rich.console import Console
from rich.progress import (
    Progress,
    BarColumn,
    DownloadColumn,
    TransferSpeedColumn,
    TimeRemainingColumn,
    TextColumn,
)

console = Console()

DEST = Path(__file__).resolve().parent.parent / "data" / "mmaud"
DEST.mkdir(parents=True, exist_ok=True)

# V1 — Rooftop Simple. Pre-extracted .zip "Folder Data". Sizes are approximate.
V1_FOLDER_ZIPS: list[tuple[str, str, str]] = [
    (
        "Mavic2.zip",
        "https://entuedu-my.sharepoint.com/:u:/g/personal/shyuan_staff_main_ntu_edu_sg/"
        "ESfINWq7uCpKkjH3YhEGZ24BmyKhcgFfF09dTZYte-C-0g?e=mww8va",
        "DJI Mavic2 — V1 Rooftop Simple — Folder Data",
    ),
    (
        "Mavic3.zip",
        "https://entuedu-my.sharepoint.com/:u:/g/personal/shyuan_staff_main_ntu_edu_sg/"
        "ETe4YTM-IKdMnA11Q1Dv_PgBWGfQ38iwYoFpkFWthhfJsQ?e=CJU8JN",
        "DJI Mavic3 — V1 Rooftop Simple — Folder Data",
    ),
    (
        "Phantom4.zip",
        "https://entuedu-my.sharepoint.com/:u:/g/personal/shyuan_staff_main_ntu_edu_sg/"
        "ERY9v7zK3hxLtxqHKVa8aDgBDSji1W_1LFeYiXN9FKJMnA?e=sTaDCI",
        "DJI Phantom4 — V1 Rooftop Simple — Folder Data",
    ),
    (
        "Avata.zip",
        "https://entuedu-my.sharepoint.com/:u:/g/personal/shyuan_staff_main_ntu_edu_sg/"
        "EU-kJE4eYctGkb6VOif4JEwBwSRjUpcWjFaDc4D-v-mUrA?e=Uxpq5y",
        "DJI Avata — V1 Rooftop Simple — Folder Data",
    ),
    (
        "M300.zip",
        "https://entuedu-my.sharepoint.com/:u:/g/personal/shyuan_staff_main_ntu_edu_sg/"
        "EacBt-X3QERBvH0Gw1d3kbMB7VECC1ATgSZHWFYzuUjGuA?e=nmBbUg",
        "DJI M300 — V1 Rooftop Simple — Folder Data",
    ),
]


def append_download(url: str) -> str:
    sep = "&" if "?" in url else "?"
    return f"{url}{sep}download=1"


def _request_with_retries(url: str, headers: dict | None = None, attempts: int = 8):
    import time as _t
    last_exc: Exception | None = None
    for i in range(attempts):
        try:
            r = requests.get(url, stream=True, allow_redirects=True, timeout=60, headers=headers)
            r.raise_for_status()
            return r
        except Exception as exc:
            last_exc = exc
            wait = min(60, 2 ** i)
            console.print(f"[yellow]net error ({exc}); retry {i+1}/{attempts} in {wait}s[/yellow]")
            _t.sleep(wait)
    raise RuntimeError(f"giving up after {attempts} attempts: {last_exc}")


def download(url: str, dest_path: Path, label: str) -> None:
    """
    Resumable HTTP download. If `<dest>.part` exists, resume via Range.
    If `<dest>` exists, skip.
    """
    if dest_path.exists() and dest_path.stat().st_size > 0:
        console.print(f"[dim]✓ already present: {dest_path.name}[/dim]")
        return

    direct = append_download(url)
    tmp = dest_path.with_suffix(dest_path.suffix + ".part")
    resume_from = tmp.stat().st_size if tmp.exists() else 0
    headers = {"Range": f"bytes={resume_from}-"} if resume_from > 0 else None
    if resume_from > 0:
        console.print(f"[cyan]resuming {tmp.name} from {resume_from/1e6:.1f} MB[/cyan]")

    r = _request_with_retries(direct, headers=headers)
    # If server ignores Range and returns 200, we have to overwrite:
    if resume_from > 0 and r.status_code == 200:
        console.print("[yellow]server ignored Range — restarting from 0[/yellow]")
        resume_from = 0
        tmp.unlink(missing_ok=True)

    total = int(r.headers.get("content-length", 0)) + resume_from

    with Progress(
        TextColumn("[bold blue]{task.description}", justify="right"),
        BarColumn(bar_width=None),
        "[progress.percentage]{task.percentage:>3.1f}%",
        "•",
        DownloadColumn(),
        "•",
        TransferSpeedColumn(),
        "•",
        TimeRemainingColumn(),
        console=console,
    ) as progress:
        task = progress.add_task(label, total=total or None, completed=resume_from)
        mode = "ab" if resume_from > 0 else "wb"
        with tmp.open(mode) as f:
            for chunk in r.iter_content(chunk_size=1 << 20):  # 1 MB
                if chunk:
                    f.write(chunk)
                    progress.update(task, advance=len(chunk))
    tmp.replace(dest_path)


def download_with_per_file_retries(url: str, dest_path: Path, label: str,
                                    max_attempts: int = 30) -> bool:
    """Retry a single file until it completes, with exp backoff.
    Each attempt resumes from the .part file via Range header."""
    import time as _t
    for i in range(1, max_attempts + 1):
        try:
            download(url, dest_path, label)
            return True
        except Exception as exc:
            wait = min(60, 2 ** min(i, 6))
            console.print(
                f"[yellow]{dest_path.name} attempt {i}/{max_attempts} failed "
                f"({type(exc).__name__}: {exc}). Retrying in {wait}s…[/yellow]"
            )
            _t.sleep(wait)
    console.print(f"[red]Giving up on {dest_path.name} after {max_attempts} attempts[/red]")
    return False


def main() -> int:
    console.rule("[bold cyan]Downloading MMAUD V1 Folder Data zips")
    console.print(f"Destination: {DEST}\n")
    console.print(
        "[yellow]Note:[/yellow] total ~70 GB. Each drone is 11–20 GB. "
        "Run repeatedly to resume — already-downloaded files are skipped.\n"
    )

    failed: list[str] = []
    for fname, url, label in V1_FOLDER_ZIPS:
        ok = download_with_per_file_retries(url, DEST / fname, label)
        if not ok:
            failed.append(fname)

    if failed:
        console.print(f"\n[red]These files did not finish: {failed}[/red]")
        # Non-zero exit so the orchestrator respawns us.
        return 2

    # Marker so the orchestrator knows MMAUD is fully ready.
    (DEST.parent / ".mmaud.done").write_text(time.strftime("%Y-%m-%dT%H:%M:%S\n"))
    console.print("\n[green]MMAUD V1 download complete.[/green]")
    return 0


if __name__ == "__main__":
    sys.exit(main())
