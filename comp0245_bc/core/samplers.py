"""Where targets and starting poses come from, for training runs and for the frozen benchmark.

One place defines the ranges for both training data and the benchmark, so they can never
drift apart. Each scene is generated from a single integer seed, so any run can be rebuilt.

Seed blocks (never overlap, so test scenes can never leak into training):
    0       - 99_999     training / validation runs
    1_000_000+           benchmark (test) scenes, see make_benchmark.py later
"""
from _paths import ROOT, find_result  # noqa: F401  (project folder; lets the code folders import each other)
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

# "Objects anywhere" (tier 6 and collect.py --wide): the object can be much closer or further, lower or
# higher, including places the arm cannot reach (too high) or should not try (on the floor).
WIDE_TARGET_R = (0.5, 3.0)
WIDE_TARGET_Z = (0.05, 1.6)
WIDE_START_R = (0.0, 2.0)


def wide_training_scene(seed):
    """Training scene with the object anywhere in the wide range (seeds in the training block)."""
    if seed >= BENCHMARK_SEED_OFFSET:
        raise ValueError("Seeds >= 1,000,000 are reserved for the benchmark.")
    rng = np.random.default_rng([seed, 6])
    return sample_target(rng, WIDE_TARGET_R, WIDE_TARGET_Z), sample_start(rng, WIDE_START_R)


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


# ------------------------------------------------------------------------------------------
# Benchmark scenes (generated once by make_benchmark.py, then frozen)
# ------------------------------------------------------------------------------------------
def _parking_goal(target):
    """Where the expert parks for this target (same formula as the template)."""
    standoff = float(np.interp(target[2], [0.1, 0.5, 1.1, 1.25], [0.80, 0.65, 0.65, 0.35]))
    b = np.arctan2(target[1], target[0])
    return target[:2] - standoff * np.array([np.cos(b), np.sin(b)])


def benchmark_scene(tier, index):
    """One frozen test scene. Each (tier, index) has its own seed in the benchmark block.

    Tier 1  in-distribution: same ranges as training, unseen seeds.
    Tier 2  out-of-distribution targets: higher (1.10-1.25 m) or farther (2.0-2.5 m) than training.
    Tier 3  unusual starts: half start far (1-2 m) facing away from the parking spot,
            half start close (0.3-0.7 m) to it, where the expert sometimes orbits.
    Tier 7  everything at once: normal scene + random mix of sensor noise, delay 0-2 ms, motor noise,
            payload in the hand, floor friction and base mass (Stage B).
    Tier 6  objects anywhere: targets 0.5-3 m away and 0.05-1.6 m high (some unreachable), starts up to 2 m.
    Tier 5  changed physics: floor friction 0.5-1.3 (template 1.0) and base mass x0.7-1.3 (added in Stage B).
    Tier 4  disturbances: tier-1-like scenes plus sensor noise and a command delay. (A sideways shove
            was tried and dropped: the template's expert never re-aligns once pushed off its heading
            and orbits instead, so the EXPERT fails every shoved scene; see README.)
    """
    seed = BENCHMARK_SEED_OFFSET + 10_000 * tier + index
    rng = np.random.default_rng(seed)
    disturbance = {}
    if tier == 1:
        target, start = sample_target(rng), sample_start(rng)
    elif tier == 2:
        if index % 2 == 0:
            target = sample_target(rng, z_range=OOD_TARGET_Z)          # too high
        else:
            target = sample_target(rng, r_range=OOD_TARGET_R)          # too far
        start = sample_start(rng)
    elif tier == 3:
        target = sample_target(rng)
        goal = _parking_goal(target)
        if index % 2 == 0:                                            # far, facing away
            x, y = _uniform_disc(rng, OOD_START_R)
            away = np.arctan2(y - goal[1], x - goal[0])
            yaw = away + rng.uniform(-0.5, 0.5)
        else:                                                         # close to the parking spot
            dx, dy = _uniform_disc(rng, (0.3, 0.7))
            x, y = goal[0] + dx, goal[1] + dy
            yaw = rng.uniform(-np.pi, np.pi)
        start = np.array([x, y, (yaw + np.pi) % (2 * np.pi) - np.pi])
    elif tier == 4:
        target, start = sample_target(rng), sample_start(rng)
        disturbance = {
            "obs_noise": float(rng.choice([0.005, 0.01, 0.02])),   # std of sensor noise (m / rad / rad/s scaled)
            "delay_steps": int(rng.choice([1, 2, 3, 4])),          # command arrives this many ms late (expert
                                                                   # survives up to 4 ms; 6 ms crashes it)
        }
    elif tier == 7:
        # Stage B "everything at once": a normal scene plus a random mix of all disturbances
        target, start = sample_target(rng), sample_start(rng)
        disturbance = {"obs_noise": float(rng.choice([0.0, 0.005, 0.01, 0.02])),
                       "delay_steps": int(rng.integers(0, 3)),
                       "torque_noise": float(rng.choice([0.0, 1.0, 2.0, 3.0])),
                       "payload_kg": float(rng.choice([0.0, 0.25, 0.5])),
                       "floor_friction": float(rng.uniform(0.6, 1.2)),
                       "base_mass_scale": float(rng.uniform(0.8, 1.2))}
    elif tier == 6:
        # Objects anywhere: wide target range incl. unreachable (too high) and floor-level targets
        target, start = sample_target(rng, WIDE_TARGET_R, WIDE_TARGET_Z), sample_start(rng, WIDE_START_R)
    elif tier == 5:
        # Changed physics (Stage B): slippier or grippier floor, lighter or heavier base. Partly beyond
        # the training randomisation (friction 0.6-1.2, mass x0.8-1.2) to test generalisation.
        target, start = sample_target(rng), sample_start(rng)
        disturbance = {"floor_friction": float(rng.uniform(0.5, 1.3)),
                       "base_mass_scale": float(rng.uniform(0.7, 1.3))}
    else:
        raise ValueError(f"unknown tier {tier}")
    return {"tier": tier, "index": index, "seed": seed, "target": [float(v) for v in target],
            "start_pose": [float(v) for v in start], "disturbance": disturbance}
