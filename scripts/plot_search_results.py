"""Export the held-out comparison and trajectories from saved measurements."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
output = ROOT / "runs/search-evaluation"
fig, axes = plt.subplots(1, 3, figsize=(14, 4.4), layout="constrained")
colors = ["#9b5264", "#247c88"]
for index, scene in enumerate(("near", "distant")):
    old_path = ROOT / "runs/search-baseline" / ("baseline.json" if scene == "near" else "baseline-distant.json")
    old = json.loads(old_path.read_text())["results"]
    new = [json.loads(p.read_text()) for p in sorted((output / scene).glob("seed*.json"))]
    ax = axes[index]
    for group, results in enumerate((old, new)):
        for i, row in enumerate(results):
            value = row["first_intake_s"]
            ax.scatter(group + (i - 3.5) * .025, value if value is not None else 20,
                       marker="o" if value is not None else "x", color=colors[group], s=38)
        fed = [r["first_intake_s"] for r in results if r["first_intake_s"] is not None]
        if fed:
            median = float(np.median(fed))
            ax.plot([group - .16, group + .16], [median, median], color=colors[group], lw=3)
        ax.text(group, 21.5, f"{len(fed)}/8 feed", ha="center", fontsize=11)
    ax.set(xticks=[0, 1], xticklabels=["v0.2 baseline", "search profile"], ylim=(-.5, 23),
           title="Nearby food" if scene == "near" else "Food 12 mm away", ylabel="First intake (physical seconds)")
    ax.axhline(20, ls=":", color="#777777", lw=1)
    ax.grid(axis="y", alpha=.2)
    ax.spines[["top", "right"]].set_visible(False)
import csv
ax = axes[2]
for path in sorted((output / "distant").glob("seed*.csv")):
    rows = list(csv.DictReader(path.open()))
    ax.plot([float(r["x_mm"]) for r in rows], [float(r["y_mm"]) for r in rows], lw=1, alpha=.8)
for angle in (40, 200):
    x, y = 12 * np.cos(np.deg2rad(angle)), 12 * np.sin(np.deg2rad(angle))
    ax.add_patch(plt.Circle((x, y), 1, color="#d58b39", alpha=.8))
ax.scatter([.55], [0], color="black", s=25, zorder=5)
ax.set(title="Distant scene: all eight paths", xlabel="x (mm)", ylabel="y (mm)", aspect="equal", xlim=(-21,21), ylim=(-21,21))
ax.grid(alpha=.2)
fig.suptitle("Held-out seeds 1-8, 20 seconds each; body/organism diagnostic without connectome", fontsize=12)
fig.savefig(output / "search-comparison.png", dpi=180)
fig.savefig(output / "search-comparison.svg")
print(output / "search-comparison.png")
