"""Input-noise data augmentation (Lecture 6): train on readings that wobble the way real sensors do.

Why: model R holds 92% in a clean world but only 8% with sensor noise (Stage B). It never saw a wobbly
reading in training. Here, every training batch gets fresh random noise added to its inputs (different every
epoch), while the label stays the expert's clean command: "even if your readings wobble, do what the expert
would do". Mathematically close to L2 regularisation (Bishop 1995): it makes the network's output smoother.

The noise must look like the real thing. Sensor noise enters the raw readings (base position, heading,
joint angles, joint speeds), and every derived input moves with it: the hand position (forward kinematics),
the IK direction (Jacobian), the history differences, the averages. feature_noise_scale() measures exactly
that: it adds the test-time noise recipe (rollout.py, tier 4) to the raw readings of a sample of recorded
drives, rebuilds all inputs with the normal feature code, and takes the spread of the change per input.
Training then adds Gaussian noise with that per-input spread, times a random level per row
(0, 0.5, 1 or 2 = clean, light, medium, strong; the test uses 0.005 / 0.01 / 0.02 = 0.5 / 1 / 2 x 0.01).
"""
from _paths import ROOT, find_result  # noqa: F401  (project folder; lets the code folders import each other)
import glob
import os

import numpy as np

HERE = ROOT
REF_SIGMA = 0.01                    # the noise level the scale is measured at (tier-4 "medium")
LEVELS = np.array([0.0, 0.5, 1.0, 2.0])


def _noisy_copy(arrays, sigma, rng):
    """The recording as a noisy sensor would have shown it (tier-4 recipe, see rollout.py)."""
    import pinocchio as pin
    from ik_features import _model
    dm, ee_id, _, _ = _model()
    a = {k: v.copy() for k, v in arrays.items()}
    n = len(a["t"])
    a["base_pos"][:, :2] += rng.normal(0, sigma, (n, 2))
    a["base_bearing"] = a["base_bearing"] + rng.normal(0, sigma, n)
    yaw = a["base_bearing"]
    a["base_quat"] = np.column_stack([np.zeros(n), np.zeros(n), np.sin(yaw / 2), np.cos(yaw / 2)]).astype(np.float32)
    a["q_mes"] = a["q_mes"] + rng.normal(0, sigma, a["q_mes"].shape)
    a["qd_mes"] = a["qd_mes"] + rng.normal(0, 10 * sigma, a["qd_mes"].shape)
    ee = np.zeros((n, 3), np.float32)
    for i in range(n):
        q_pin = dm.ReoderJoints2PinVec(np.concatenate((a["base_pos"][i], a["base_quat"][i], a["q_mes"][i])), "pos")
        pin.forwardKinematics(dm.pin_model, dm.pin_data, q_pin)
        pin.updateFramePlacements(dm.pin_model, dm.pin_data)
        ee[i] = dm.pin_data.oMf[ee_id].translation
    a["ee_pos"] = ee
    return a


def feature_noise_scale(a, run_data, n_runs=15, seed=0):
    """Per-input spread of the change caused by REF_SIGMA sensor noise (raw input units). Cached."""
    from check_runs import load_run
    tag = f"h{a.history}_g{int(a.goal)}_i{int(a.integral)}_k{int(getattr(a, 'ik', False))}_r{int(getattr(a, 'relative', False))}"
    tag += "_d" + "-".join(getattr(a, "drop", []) or [])
    cache = os.path.join(HERE, "splits", f"noise_scale_{tag}.npy")
    if os.path.exists(cache):
        return np.load(cache)
    rng = np.random.default_rng(seed)
    files = sorted(glob.glob(os.path.join(HERE, "data", "runs", "run_*.npz")))[:n_runs]
    diffs = []
    for f in files:
        _, arrays = load_run(f)
        Xc, _, _ = run_data(arrays, a)
        Xn, _, _ = run_data(_noisy_copy(arrays, REF_SIGMA, rng), a)
        diffs.append(Xn - Xc)
    scale = np.concatenate(diffs).std(0).astype(np.float32)
    np.save(cache, scale)
    return scale


def add_noise(xb, scale_std, gen):
    """xb: standardised input batch (torch). scale_std: per-input noise spread in standardised units."""
    import torch
    lv = torch.as_tensor(LEVELS, dtype=xb.dtype)[torch.randint(len(LEVELS), (len(xb),), generator=gen)]
    return xb + torch.randn(xb.shape, generator=gen) * lv[:, None] * scale_std[None, :]
