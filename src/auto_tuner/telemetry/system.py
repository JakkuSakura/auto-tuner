from __future__ import annotations

import json
import os
import platform
import shutil
import subprocess
import sys
from datetime import UTC, datetime


def _utc_now_iso() -> str:
    return datetime.now(UTC).isoformat()


def _physical_memory_bytes() -> int | None:
    if sys.platform == "darwin":
        result = subprocess.run(["sysctl", "-n", "hw.memsize"], capture_output=True, text=True)
        if result.returncode != 0:
            return None
        try:
            return int(result.stdout.strip())
        except ValueError:
            return None

    try:
        page_size = os.sysconf("SC_PAGE_SIZE")
        pages = os.sysconf("SC_PHYS_PAGES")
        return int(page_size) * int(pages)
    except (AttributeError, ValueError, OSError):
        return None
    return None


def _nvidia_smi_gpus() -> list[dict[str, object]]:
    if shutil.which("nvidia-smi") is None:
        return []
    fields = [
        "index",
        "uuid",
        "name",
        "driver_version",
        "memory.total",
        "compute_cap",
    ]
    cmd = [
        "nvidia-smi",
        f"--query-gpu={','.join(fields)}",
        "--format=csv,noheader,nounits",
    ]
    result = subprocess.run(cmd, check=False, capture_output=True, text=True)
    if result.returncode != 0:
        return []

    gpus: list[dict[str, object]] = []
    for raw_line in result.stdout.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        parts = [part.strip() for part in line.split(",")]
        if len(parts) != len(fields):
            continue
        payload: dict[str, object] = {}
        for key, value in zip(fields, parts, strict=True):
            payload[key] = value
        gpus.append(payload)
    return gpus


def collect_system_info() -> dict[str, object]:
    gpus = _nvidia_smi_gpus()
    gpu_backend = "nvidia-smi" if gpus else "unknown"
    return {
        "timestamp": _utc_now_iso(),
        "python": {"version": sys.version, "executable": sys.executable},
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
            "processor": platform.processor(),
        },
        "memory": {"physical_bytes": _physical_memory_bytes()},
        "gpu": {"backend": gpu_backend, "devices": gpus},
        "env": {
            "HF_HOME": os.getenv("HF_HOME", ""),
            "TRANSFORMERS_CACHE": os.getenv("TRANSFORMERS_CACHE", ""),
            "CUDA_VISIBLE_DEVICES": os.getenv("CUDA_VISIBLE_DEVICES", ""),
        },
    }


def dumps_system_info() -> str:
    return json.dumps(collect_system_info(), indent=2)
