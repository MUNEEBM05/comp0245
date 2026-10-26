"""Where targets and starting poses come from.

One place defines the ranges for both training data and the benchmark, so they can never
drift apart. Each scene is generated from a single integer seed, so any run can be rebuilt.

Seed blocks (never overlap, so test scenes can never leak into training):
    0       - 99_999     training / validation runs
    1_000_000+           benchmark (test) scenes, see make_benchmark.py later
"""
import numpy as np

# ------------------------------------------------------------------------------------------
# Training ranges (in-distribution)
# ------------------------------------------------------------------------------------------
# Target height: above the floor with margin (arm tip must not scrape the ground), and up to
# 1.1 m, the top of the template's "mid height" standoff band. 1.1-1.25 m is held out (tier 2).
TRAIN_TARGET_Z = (0.15, 1.10)
# Target horizontal distance from the world origin. The expert parks `standoff` (0.35-0.8 m)
# short of the target along the origin->target line, so r >= 1.0 keeps the parking spot on the
# target's side of the origin. 2.0-2.5 m is held out (tier 2).
TRAIN_TARGET_R = (1.0, 2.0)
# Robot start: anywhere within 1 m of the origin, facing any direction.
TRAIN_START_R = (0.0, 1.0)

# ------------------------------------------------------------------------------------------
# Held-out ranges (out-of-distribution, benchmark only)
# ------------------------------------------------------------------------------------------
OOD_TARGET_Z = (1.10, 1.25)
OOD_TARGET_R = (2.0, 2.5)
OOD_START_R = (1.0, 2.0)

BENCHMARK_SEED_OFFSET = 1_000_000


def _uniform_disc(rng, r_range):
    """Uniform over the AREA of an annulus (not uniform in radius, which bunches near the centre)."""
    r_lo, r_hi = r_range
    r = np.sqrt(rng.uniform(r_lo ** 2, r_hi ** 2))
    theta = rng.uniform(-np.pi, np.pi)
    return r * np.cos(theta), r * np.sin(theta)


def sample_target(rng, r_range=TRAIN_TARGET_R, z_range=TRAIN_TARGET_Z):
    x, y = _uniform_disc(rng, r_range)
    z = rng.uniform(*z_range)
    return np.array([x, y, z])


def sample_start(rng, r_range=TRAIN_START_R):
    x, y = _uniform_disc(rng, r_range)
    yaw = rng.uniform(-np.pi, np.pi)
    return np.array([x, y, yaw])


def training_scene(seed):
    """Target and start for training run `seed`. Same seed -> same scene, on any machine."""
    if seed >= BENCHMARK_SEED_OFFSET:
        raise ValueError("Seeds >= 1,000,000 are reserved for the benchmark.")
    rng = np.random.default_rng(seed)
    target = sample_target(rng)
    start = sample_start(rng)
    return target, start
