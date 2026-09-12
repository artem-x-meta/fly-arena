"""Bounded actual-window close test, targeting only this child process tree."""
import ctypes
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.check_windows_launch import find_window, wait_until, descendants
from fly_arena.checkpoint import load_checkpoint

output = ROOT / "runs/circuit-lab-audit/window"
output.mkdir(parents=True, exist_ok=True)
with (output / "process.log").open("w") as log:
    process = subprocess.Popen([sys.executable, "-m", "fly_circuit_lab", "run", "--control", "paper",
                                "--seconds", "0", "--output", str(output)], cwd=ROOT,
                               stdout=log, stderr=subprocess.STDOUT)
    try:
        hwnd, title = wait_until(lambda: find_window(process), 45)
        ctypes.windll.user32.ShowWindow(ctypes.c_void_p(hwnd), 0)
        time.sleep(1.)
        ctypes.windll.user32.PostMessageW(ctypes.c_void_p(hwnd), 0x10, 0, 0)
        process.wait(timeout=40)
        assert process.returncode == 0
        state, compatibility = load_checkpoint(output / "checkpoints/latest.npz")
        base = state["legacy_state"]
        t = base["organism"]["clocks"]["physics_time_s"]
        paper_clock = base["looming_detector"]["probe"]["stream"]["clock"] * .0005
        assert t > 0 and abs(t - paper_clock) < 1e-9
        assert abs(base["brain"]["clock"] / 1000 - t) < 1e-9
        (output / "result.json").write_text(json.dumps({"status": "pass", "window_title": title,
            "physics_s": t, "paper_s": paper_clock, "exit_code": 0,
            "scope": "Actual MuJoCo window launch and WM_CLOSE, complete lab snapshot saved"}, indent=2))
        print("Circuit laboratory actual-window close: PASS")
    finally:
        if process.poll() is None:
            for child in reversed(descendants(process)):
                child.terminate()
