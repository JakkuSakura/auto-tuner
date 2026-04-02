from __future__ import annotations

import json
import platform
import shutil
import subprocess
import threading
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path


def _utc_now_iso() -> str:
    return datetime.now(UTC).isoformat()


def _nvidia_smi_samples() -> list[dict[str, object]]:
    if shutil.which("nvidia-smi") is None:
        return []

    query_fields = [
        "index",
        "uuid",
        "name",
        "utilization.gpu",
        "utilization.memory",
        "memory.used",
        "memory.total",
        "temperature.gpu",
        "power.draw",
    ]
    cmd = [
        "nvidia-smi",
        f"--query-gpu={','.join(query_fields)}",
        "--format=csv,noheader,nounits",
    ]
    result = subprocess.run(cmd, check=False, capture_output=True, text=True)
    if result.returncode != 0:
        return []

    samples: list[dict[str, object]] = []
    for raw_line in result.stdout.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        parts = [part.strip() for part in line.split(",")]
        if len(parts) != len(query_fields):
            continue
        payload: dict[str, object] = {}
        for key, value in zip(query_fields, parts, strict=True):
            payload[key] = value
        samples.append(payload)
    return samples


@dataclass(frozen=True)
class GpuMonitorConfig:
    interval_s: float = 1.0


class GpuMonitor:
    def __init__(self, path: Path, config: GpuMonitorConfig | None = None) -> None:
        self._path = path
        self._config = config or GpuMonitorConfig()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with self._path.open("a", encoding="utf-8") as f:
            f.write(
                json.dumps(
                    {
                        "type": "header",
                        "timestamp": _utc_now_iso(),
                        "interval_s": self._config.interval_s,
                    }
                )
                + "\n"
            )
            f.flush()

        self._thread = threading.Thread(target=self._run, name="auto-tuner-gpu-monitor", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        thread = self._thread
        if thread is not None:
            thread.join(timeout=5)

    def _run(self) -> None:
        # Best-effort only: never break training because monitoring failed.
        while not self._stop.is_set():
            timestamp = _utc_now_iso()
            try:
                rows = self._collect(timestamp)
                if rows:
                    with self._path.open("a", encoding="utf-8") as f:
                        for row in rows:
                            f.write(json.dumps(row) + "\n")
                        f.flush()
            except Exception:
                # Swallow all exceptions; monitoring is optional.
                pass
            time.sleep(self._config.interval_s)

    def _collect(self, timestamp: str) -> list[dict[str, object]]:
        system = platform.system()
        if system == "Darwin":
            return [
                {
                    "type": "sample",
                    "timestamp": timestamp,
                    "backend": "apple",
                    "status": "unsupported",
                    "reason": "Real-time GPU utilization is not implemented for macOS in this project yet.",
                }
            ]

        samples = _nvidia_smi_samples()
        if not samples:
            return [
                {
                    "type": "sample",
                    "timestamp": timestamp,
                    "backend": "unknown",
                    "status": "unavailable",
                    "reason": "No NVIDIA GPU telemetry source available (missing nvidia-smi).",
                }
            ]

        rows: list[dict[str, object]] = []
        for sample in samples:
            rows.append(
                {
                    "type": "sample",
                    "timestamp": timestamp,
                    "backend": "nvidia-smi",
                    **sample,
                }
            )
        return rows

