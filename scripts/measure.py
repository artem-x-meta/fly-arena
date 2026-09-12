"""Run a command, streaming its log and sampling process RSS (Windows/Linux).

Usage: python scripts/measure.py runs/audit/baseline -- python -m fly_arena ...
The sampled RSS is distinct from GPU memory and includes no other processes.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import platform
import subprocess
import sys
import time

import psutil


def memory_bytes(process):
    # Windows venv python.exe is a small redirector; the runtime is its child.
    # Include the launched process tree (and video encoder), never unrelated apps.
    try:
        parent = psutil.Process(process.pid)
        members = [parent] + parent.children(recursive=True)
        infos = []
        for member in members:
            try:
                infos.append(member.memory_info())
            except psutil.NoSuchProcess:
                pass
        return sum(info.rss for info in infos), sum(
            getattr(info, "peak_wset", info.rss) for info in infos)
    except psutil.NoSuchProcess:
        return None, None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--attach", type=int, help="Observe an already running process tree")
    parser.add_argument("--gpu", action="store_true", help="Sample NVIDIA device-wide memory, distinct from process VRAM")
    parser.add_argument("prefix", type=Path)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if command and Path(command[0]).is_file():
        command[0] = str(Path(command[0]).resolve())
    args.prefix.parent.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    samples, peak, gpu_samples = [], 0, []
    with args.prefix.with_suffix(".log").open("w", encoding="utf-8") as log:
        process = psutil.Process(args.attach) if args.attach else subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
        try:
            while (process.is_running() if args.attach else process.poll() is None):
                rss, high = memory_bytes(process)
                wall = time.perf_counter() - start
                if rss is not None:
                    peak = max(peak, rss, high or 0)
                    samples.append([round(wall, 3), rss])
                if len(samples) % 20 == 0:
                    print(f"elapsed {wall:.1f}s | peak RSS {peak / 2**20:.1f} MiB", flush=True)
                    if args.gpu:
                        try:
                            gpu = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total,memory.used", "--format=csv,noheader,nounits"],
                                                 capture_output=True, text=True, timeout=3)
                            gpu_samples.append({"wall_s": wall, "output": gpu.stdout.strip(), "error": gpu.stderr.strip()})
                        except (OSError, subprocess.TimeoutExpired) as error:
                            gpu_samples.append({"wall_s": wall, "error": str(error)})
                time.sleep(.25)
        except KeyboardInterrupt:
            if not args.attach:
                process.terminate()
                process.wait()
    returncode = None if args.attach else process.returncode
    report = {"command": command, "platform": platform.platform(),
              "wall_seconds": time.perf_counter() - start, "exit_code": returncode,
              "peak_rss_bytes": peak, "rss_samples_wall_s_bytes": samples,
              "vram": "not measured by this utility", "sample_period_s": .25,
              "rss_scope": "launched process tree including Windows venv redirector and video encoder",
              "attached_pid": args.attach}
    report["gpu_device_memory_samples"] = gpu_samples
    report["vram"] = "device-wide NVIDIA counters only; process GPU allocation unavailable" if args.gpu else "not measured by this utility"
    args.prefix.with_suffix(".json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "rss_samples_wall_s_bytes"}))
    raise SystemExit(returncode or 0)


if __name__ == "__main__":
    main()
