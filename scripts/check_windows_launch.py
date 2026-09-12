"""Exercise a private Windows console Ctrl+C and an actual MuJoCo window.

Each signal test owns its hidden console; no signals go to the user's console.
Only windows belonging to the launched process tree receive keyboard/close input.
"""
from __future__ import annotations

import argparse
import ctypes
import json
from pathlib import Path
import signal
import subprocess
import sys
import time

import psutil
import shutil

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def descendants(process):
    try:
        parent = psutil.Process(process.pid)
        return [parent] + parent.children(recursive=True)
    except psutil.NoSuchProcess:
        return []


def find_window(process):
    ids = {p.pid for p in descendants(process)}
    found = []
    callback_type = ctypes.WINFUNCTYPE(ctypes.c_bool, ctypes.c_void_p, ctypes.c_void_p)

    @callback_type
    def callback(hwnd, _):
        pid = ctypes.c_ulong()
        ctypes.windll.user32.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
        if pid.value in ids:
            text = ctypes.create_unicode_buffer(512)
            ctypes.windll.user32.GetWindowTextW(hwnd, text, 512)
            if "mujoco" in text.value.lower():
                found.append((hwnd, text.value))
        return True

    ctypes.windll.user32.EnumWindows(callback, 0)
    return found[0] if found else None


def wait_until(predicate, seconds=50):
    start = time.monotonic()
    while time.monotonic() - start < seconds:
        value = predicate()
        if value:
            return value
        time.sleep(.1)
    raise TimeoutError("Windows integration test timed out")


def worker(kind, output):
    # This process has a private hidden console created by the parent harness.
    signal.signal(signal.SIGINT, lambda *_: None)
    signal.signal(signal.SIGBREAK, lambda *_: None)
    output.mkdir(parents=True, exist_ok=True)
    # A rerun must not read the pause event of an earlier run in this directory.
    shutil.rmtree(output / "run", ignore_errors=True)
    log_path = output / "process.log"
    command = [sys.executable, "-m", "fly_arena", "run", "--scenario", "feeding-contact",
               "--seconds", "0", "--output", str(output / "run")]
    if kind == "ctrl-c":
        command.append("--headless")
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
        try:
            window = None
            if kind == "window":
                window = wait_until(lambda: find_window(process))
                hwnd, title = window
                ctypes.windll.user32.ShowWindow(ctypes.c_void_p(hwnd), 0)
                wait_until(lambda: "physics " in log_path.read_text(encoding="utf-8", errors="replace"))
                # Exercise the same Space key path handled by GLFW/the viewer.
                ctypes.windll.user32.PostMessageW(ctypes.c_void_p(hwnd), 0x100, 0x20, 1)
                ctypes.windll.user32.PostMessageW(ctypes.c_void_p(hwnd), 0x101, 0x20, (1 << 30) | (1 << 31))
                def paused_event():
                    events = []
                    for path in (output / "run").glob("*-events-*.jsonl"):
                        events.extend(json.loads(line) for line in path.read_text().splitlines())
                    return next((e for e in events if e["kind"] == "pause"), None)
                pause_record = wait_until(paused_event, 10)
                time.sleep(.5)
                ctypes.windll.user32.PostMessageW(ctypes.c_void_p(hwnd), 0x10, 0, 0)
            else:
                wait_until(lambda: "physics " in log_path.read_text(encoding="utf-8", errors="replace"))
                if not ctypes.windll.kernel32.GenerateConsoleCtrlEvent(0, 0):
                    raise ctypes.WinError()
            process.wait(timeout=30)
            if process.returncode:
                raise RuntimeError(log_path.read_text(encoding="utf-8", errors="replace"))
            from fly_arena.checkpoint import load_checkpoint
            state, compatibility = load_checkpoint(output / "run" / "checkpoints" / "latest.npz")
            t = state["organism"]["clocks"]["physics_time_s"]
            assert t > 0
            assert abs(state["brain"]["clock"] / 1000 - t) < 1e-9
            assert abs(state["arena"]["physics_steps"] * .0001 - t) < 1e-9
            if kind == "window":
                assert abs(t - pause_record["physics_time_s"]) < 1e-9
            result = {"kind": kind, "exit_code": process.returncode,
                      "checkpoint_physics_s": t, "clocks_aligned": True,
                      "window_title": window[1] if window else None,
                      "space_key_events_sent": kind == "window",
                      "pause_physics_frozen_until_window_close": kind == "window",
                      "pause_state_equivalence": "tested separately by full-state simulation tests"}
            (output / "result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
        finally:
            for member in reversed(descendants(process)):
                try:
                    member.terminate()
                except psutil.NoSuchProcess:
                    pass


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", choices=("ctrl-c", "window"))
    parser.add_argument("--output", type=Path, default=ROOT / "runs" / "windows-check")
    args = parser.parse_args()
    if sys.platform != "win32":
        raise SystemExit("This check requires Windows")
    if args.worker:
        worker(args.worker, args.output.resolve())
        return
    args.output.mkdir(parents=True, exist_ok=True)
    results = []
    for kind in ("ctrl-c", "window"):
        directory = args.output.resolve() / kind
        directory.mkdir(parents=True, exist_ok=True)
        startup = subprocess.STARTUPINFO()
        startup.dwFlags |= subprocess.STARTF_USESHOWWINDOW
        startup.wShowWindow = subprocess.SW_HIDE
        with (directory / "harness.log").open("w") as log:
            process = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), "--worker", kind,
                                        "--output", str(directory)], cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                                       creationflags=subprocess.CREATE_NEW_CONSOLE, startupinfo=startup)
            process.wait(timeout=120)
        if process.returncode:
            raise RuntimeError((directory / "harness.log").read_text(errors="replace"))
        results.append(json.loads((directory / "result.json").read_text()))
        print(f"{kind}: PASS", flush=True)
    (args.output / "results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
