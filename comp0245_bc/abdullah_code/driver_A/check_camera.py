"""How accurate is the camera? The robot stands still; the target is placed at known spots. Error in cm against distance.

    python check_camera.py --out results/camera.json
"""
import argparse
import json
import os
import time

import numpy as np

from camera import TargetCamera, add_target_marker
from quiet import run_main

HERE = os.path.dirname(os.path.abspath(__file__))
DISTANCES = [0.5, 0.65, 0.8, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--width", type=int, default=320)
    ap.add_argument("--pitch", type=float, default=30.0)
    ap.add_argument("--out", default=None)
    ap.add_argument("--image", default=None, help="save one picture from each distance to this folder")
    a = ap.parse_args()
    from rollout import World
    world = World((0.0, 0.0, 0.0), np.array([1.0, 0.0, 0.5]))
    cam = TargetCamera(world, width=a.width, height_px=a.width * 3 // 4, pitch_deg=a.pitch)
    marker = add_target_marker(world, [1.0, 0.0, 0.5])
    rng, rows, times = np.random.default_rng(0), [], []
    for d in DISTANCES:
        errs, along, missed = [], [], 0
        for i in range(a.n):
            bearing, z = rng.uniform(-0.3, 0.3), rng.uniform(0.3, 1.0)
            tgt = np.array([d * np.cos(bearing), d * np.sin(bearing), z])
            world.client.resetBasePositionAndOrientation(marker, tgt.tolist(), [0, 0, 0, 1])
            t0 = time.perf_counter()
            m, n = cam.measure(np.zeros(2), 0.0)
            times.append(time.perf_counter() - t0)
            if m is None:
                missed += 1
                continue
            errs.append(float(np.linalg.norm(m - tgt)))
            if a.image and i == 0:
                import matplotlib.pyplot as plt
                os.makedirs(a.image, exist_ok=True)
                plt.imsave(os.path.join(a.image, f"d{d}.png"), cam.image(np.zeros(2), 0.0)[0])
        rows.append({"distance_m": d, "seen": a.n - missed, "n": a.n, "median_cm": 100 * float(np.median(errs)) if errs else None,
                     "p95_cm": 100 * float(np.percentile(errs, 95)) if errs else None})
        print(rows[-1], flush=True)
    print(f"{1000 * np.median(times):.0f} ms per image at {a.width} px")
    if a.out:
        with open(a.out, "w") as fh:
            json.dump({"width": a.width, "pitch_deg": a.pitch, "ms_per_image": 1000 * float(np.median(times)), "rows": rows}, fh, indent=1)


if __name__ == "__main__":
    run_main(main, HERE)
