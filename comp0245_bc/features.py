"""Turn robot measurements into model inputs, and name the outputs.

The same functions build the inputs for training (whole recorded runs at once) and for driving
(one step at a time), so the model always sees exactly the same kind of numbers.

All inputs are computed from the measurements the brief names (base_pos, base_bearing, q_mes,
qd_mes) plus the target. The hand position is derived from those through the robot's geometry
(forward kinematics); the parking spot from the target by the template's formula. Nothing comes
from the expert's internal state. The input groups, in the order they are stacked:

    basic (26)     hand-to-target error, target (robot frame and world), base pose, 7 arm angles, 7 speeds
    history (13)   how the base moved / turned over the last 0.1 s and 0.5 s; arm joint change over 0.1 s
    goal (5)       parking spot seen from the robot, and how far off facing the target
    integral (6)   hand-to-target error averaged over the last 0.5 s and 2 s
    wheel_vel (4)  ablation only: measured wheel speeds (the copycat trap)

Outputs (11): 7 arm torques (or torque minus gravity, see train.py) and 4 wheel speeds.
Deliberately NOT inputs: raw time (meaningless with random starts), past commands and wheel speeds
(the model would learn to repeat itself), the expert's phase and arm reference (not measurable).
"""
import numpy as np

ARM = slice(4, 11)

FEATURE_NAMES = (
    ["err_x", "err_y", "err_z"]                      # target minus hand, world frame
    + ["tgt_fwd", "tgt_left", "tgt_z"]               # target seen from the robot (base frame)
    + ["tgt_x", "tgt_y"]                             # target, world frame (the expert parks relative to the origin)
    + ["base_x", "base_y", "sin_yaw", "cos_yaw"]     # where the base is and which way it faces
    + [f"q{i}" for i in range(1, 8)]                 # arm joint angles
    + [f"qd{i}" for i in range(1, 8)]                # arm joint speeds
)
OUTPUT_NAMES = [f"torque_{i}" for i in range(1, 8)] + ["wheel_fr", "wheel_fl", "wheel_bl", "wheel_br"]


def make_features(base_pos, base_bearing, q_mes, qd_mes, ee_pos, target):
    """All arguments may be single steps (1-D) or whole runs (2-D, one row per step).

    Returns an array of shape (n_steps, len(FEATURE_NAMES)).
    """
    base_pos = np.atleast_2d(base_pos)
    yaw = np.atleast_1d(base_bearing).astype(float)
    q = np.atleast_2d(q_mes)[:, ARM]
    qd = np.atleast_2d(qd_mes)[:, ARM]
    ee = np.atleast_2d(ee_pos)
    tgt = np.atleast_2d(target)
    if tgt.shape[0] == 1 and base_pos.shape[0] > 1:
        tgt = np.repeat(tgt, base_pos.shape[0], axis=0)

    c, s = np.cos(yaw), np.sin(yaw)
    dx, dy = tgt[:, 0] - base_pos[:, 0], tgt[:, 1] - base_pos[:, 1]
    fwd = c * dx + s * dy            # rotate into the robot's frame
    left = -s * dx + c * dy

    return np.column_stack([
        tgt - ee,
        fwd, left, tgt[:, 2],
        tgt[:, 0], tgt[:, 1],
        base_pos[:, 0], base_pos[:, 1], s, c,
        q, qd,
    ]).astype(np.float32)


def make_targets(torque_arm, wheel_cmds):
    """The 11 outputs the model predicts: 7 arm torques then 4 wheel speeds."""
    return np.column_stack([np.atleast_2d(torque_arm), np.atleast_2d(wheel_cmds)]).astype(np.float32)


def run_to_xy(arrays):
    """Inputs and outputs for one recorded run (dict of arrays from check_runs.load_run)."""
    X = make_features(arrays["base_pos"], arrays["base_bearing"], arrays["q_mes"], arrays["qd_mes"],
                      arrays["ee_pos"], arrays["target"])
    Y = make_targets(arrays["torque_arm"], arrays["wheel_cmds"])
    return X, Y


# ------------------------------------------------------------------------------------------
# History features (v3): how the robot moved recently. Reveals the expert's hidden phase
# (turning on the spot vs driving straight) without using the expert's internal state.
# Past COMMANDS are deliberately not used (the model would learn to repeat itself).
# ------------------------------------------------------------------------------------------
HISTORY_LAGS_S = (0.1, 0.5)
HISTORY_NAMES = ([f"{n}_{int(l * 1000)}ms" for l in HISTORY_LAGS_S for n in ("d_fwd", "d_left", "d_yaw")]
                 + [f"dq{i}_100ms" for i in range(1, 8)])


def _wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def history_features(base_pos_now, yaw_now, q_now, base_pos_past, yaw_past, q_past):
    """Change between now and the past samples (one past sample per lag; arm uses the first lag).

    base_pos_past / yaw_past: lists, one entry per lag in HISTORY_LAGS_S; q_past: arm angles at lag 1.
    All arguments may be 1-D (one step) or 2-D (whole run).
    """
    bp = np.atleast_2d(base_pos_now)
    yaw = np.atleast_1d(yaw_now).astype(float)
    c, s = np.cos(yaw), np.sin(yaw)
    cols = []
    for bpp, ywp in zip(base_pos_past, yaw_past):
        bpp = np.atleast_2d(bpp)
        dx, dy = bp[:, 0] - bpp[:, 0], bp[:, 1] - bpp[:, 1]
        cols += [c * dx + s * dy, -s * dx + c * dy, _wrap(yaw - np.atleast_1d(ywp))]
    dq = np.atleast_2d(q_now)[:, ARM] - np.atleast_2d(q_past)[:, ARM]
    return np.column_stack(cols + [dq]).astype(np.float32)


def run_history(arrays, log_dt=0.01):
    """History features for a whole recorded run (rows are log_dt apart; early rows reuse row 0)."""
    n = len(arrays["t"])
    idx = np.arange(n)
    past = [np.maximum(idx - int(round(l / log_dt)), 0) for l in HISTORY_LAGS_S]
    return history_features(arrays["base_pos"], arrays["base_bearing"], arrays["q_mes"],
                            [arrays["base_pos"][p] for p in past], [arrays["base_bearing"][p] for p in past],
                            arrays["q_mes"][past[0]])


def run_to_xy_history(arrays):
    X, Y = run_to_xy(arrays)
    return np.column_stack([X, run_history(arrays)]), Y


# ------------------------------------------------------------------------------------------
# Parking-spot features (variant A): where the expert's parking spot is, seen from the robot.
# The spot follows from the target by the template's own formula, so the model can be told it.
# Makes "I have arrived and face the right way: stop" an explicit, easy-to-learn condition.
# ------------------------------------------------------------------------------------------
GOAL_NAMES = ["goal_fwd", "goal_left", "goal_dist", "face_err_sin", "face_err_cos"]
_STANDOFF_Z = [0.1, 0.5, 1.1, 1.25]
_STANDOFF = [0.80, 0.65, 0.65, 0.35]


def goal_features(base_pos, base_bearing, target):
    bp = np.atleast_2d(base_pos)
    yaw = np.atleast_1d(base_bearing).astype(float)
    tgt = np.atleast_2d(target)
    if tgt.shape[0] == 1 and bp.shape[0] > 1:
        tgt = np.repeat(tgt, bp.shape[0], axis=0)
    standoff = np.interp(tgt[:, 2], _STANDOFF_Z, _STANDOFF)
    tb = np.arctan2(tgt[:, 1], tgt[:, 0])
    gx, gy = tgt[:, 0] - standoff * np.cos(tb), tgt[:, 1] - standoff * np.sin(tb)
    c, s = np.cos(yaw), np.sin(yaw)
    dx, dy = gx - bp[:, 0], gy - bp[:, 1]
    face = _wrap(tb - yaw)                       # how far the robot is from facing the target
    return np.column_stack([c * dx + s * dy, -s * dx + c * dy, np.hypot(dx, dy),
                            np.sin(face), np.cos(face)]).astype(np.float32)


def run_xy(arrays, history=False, goal=False):
    """Inputs and outputs for one recorded run, with the chosen optional input groups."""
    X, Y = run_to_xy(arrays)
    parts = [X]
    if history:
        parts.append(run_history(arrays))
    if goal:
        parts.append(goal_features(arrays["base_pos"], arrays["base_bearing"], arrays["target"]))
    return np.column_stack(parts), Y


def feature_names(history=False, goal=False):
    return FEATURE_NAMES + (HISTORY_NAMES if history else []) + (GOAL_NAMES if goal else [])


# ------------------------------------------------------------------------------------------
# Integral features (--integral): the expert closes the last centimetre with an internal
# reference that keeps creeping while an error remains (it integrates over time). These give
# the model its own memory of that, built from measurements only: the hand-to-target error
# averaged over the last ~0.5 s and ~2 s (exponential moving averages). "Averaged error stays
# small but not zero" = "I am stuck just short: push a bit harder".
# ------------------------------------------------------------------------------------------
INTEGRAL_TAUS_S = (0.5, 2.0)
INTEGRAL_NAMES = [f"err_avg_{a}_{int(t * 1000)}ms" for t in INTEGRAL_TAUS_S for a in "xyz"]


class ErrorAverager:
    """Streaming version (for driving): call update(error) every 10 ms."""

    def __init__(self, dt=0.01):
        self.alpha = [np.exp(-dt / t) for t in INTEGRAL_TAUS_S]
        self.avg = None

    def update(self, err):
        err = np.asarray(err, float)
        if self.avg is None:
            self.avg = [err.copy() for _ in INTEGRAL_TAUS_S]
        else:
            self.avg = [a * m + (1 - a) * err for a, m in zip(self.alpha, self.avg)]
        return np.concatenate(self.avg).astype(np.float32)


def run_integral(arrays, log_dt=0.01):
    """Integral features for a whole recorded run (rows are log_dt apart)."""
    err = (arrays["target"] - arrays["ee_pos"]).astype(float)
    avg = ErrorAverager(log_dt)
    return np.array([avg.update(e) for e in err], dtype=np.float32)


WHEEL_VEL_NAMES = ["wheel_vel_fr", "wheel_vel_fl", "wheel_vel_bl", "wheel_vel_br"]


def run_xy_full(arrays, history=False, goal=False, integral=False, action="torque", wheel_vel=False, smooth=None,
                smooth_qd=True):
    """Inputs and outputs with all optional groups.

    action="torque":    outputs = 7 arm torques + 4 wheel speeds (the brief's outputs)
    action="reference": outputs = 7 arm reference offsets (q_ref - q) + 7 reference speeds + 4 wheel speeds;
                        a fixed PD law (the template's Kp=100, Kd=30, plus gravity) turns them into torques.
    smooth=tau:         arm speeds and history built from readings smoothed over ~tau seconds (model M)
    smooth_qd=False:    with smooth, keep the arm speeds raw and smooth only the history (model M2)
    """
    if smooth:   # model M: arm speeds and history from smoothed readings, positions stay raw
        sm = smooth_run(arrays, smooth)
        # M2 keeps the arm speeds raw: they are the arm's brake, and a lagged brake makes it swing
        X, Y = run_to_xy(dict(arrays, qd_mes=sm["qd_mes"]) if smooth_qd else arrays)
        parts = [X]
        if history:
            parts.append(run_history(sm))
        if goal:
            parts.append(goal_features(arrays["base_pos"], arrays["base_bearing"], arrays["target"]))
        X = np.column_stack(parts)
    else:
        X, Y = run_xy(arrays, history=history, goal=goal)
    if integral:
        X = np.column_stack([X, run_integral(arrays)])
    if wheel_vel:   # measured wheel speeds (qd_mes 0-3): risk of the "copycat" shortcut, tested as an ablation
        X = np.column_stack([X, arrays["qd_mes"][:, 0:4]]).astype(np.float32)
    if action == "reference":
        q = arrays["q_mes"][:, ARM]
        Y = np.column_stack([arrays["q_des_arm"] - q, arrays["qd_des_arm"], arrays["wheel_cmds"]]).astype(np.float32)
    return X, Y


def output_names(action="torque"):
    if action == "reference":
        return ([f"q_ref_offset_{i}" for i in range(1, 8)] + [f"qd_ref_{i}" for i in range(1, 8)]
                + OUTPUT_NAMES[7:])
    return OUTPUT_NAMES


def feature_names_full(history=False, goal=False, integral=False, wheel_vel=False):
    return (feature_names(history, goal) + (INTEGRAL_NAMES if integral else [])
            + (WHEEL_VEL_NAMES if wheel_vel else []))


# ------------------------------------------------------------------------------------------
# Smoothed inputs (--smooth, model M): under sensor noise the arm speeds and the 0.1 s history
# changes are mostly noise (noise_check.py). M feeds the network smoothed versions instead:
# base position, heading, arm angles and arm speeds are exponentially averaged over the last
# ~tau seconds, and the speeds and history are built from those. Positions stay raw (noise is
# small next to them, and smoothing would only add lag).
# M crashed even on clean scenes: the network brakes the arm with the speed inputs, and a 50 ms
# old speed brakes at the wrong moment. M2 (smooth_qd=False) smooths only the history.
# ------------------------------------------------------------------------------------------
SMOOTH_TAU_S = 0.05


class Smoother:
    """Streaming version (for driving): call update(...) every 10 ms with the raw readings."""

    def __init__(self, tau=SMOOTH_TAU_S, dt=0.01):
        self.k = 1.0 - np.exp(-dt / tau)
        self.s = None

    def update(self, base_pos, yaw, q, qd):
        new = [np.array(base_pos, float), float(yaw), np.array(q, float), np.array(qd, float)]
        if self.s is None:
            self.s = new
        else:
            bp, yw, qq, qv = self.s
            self.s = [bp + self.k * (new[0] - bp),
                      _wrap(yw + self.k * _wrap(new[1] - yw)),   # the short way round past +-180 deg
                      qq + self.k * (new[2] - qq),
                      qv + self.k * (new[3] - qv)]
        return self.s


def smooth_run(arrays, tau=SMOOTH_TAU_S, log_dt=0.01):
    """Copy of a recorded run with base position, heading, arm angles and speeds smoothed."""
    sm = Smoother(tau, log_dt)
    rows = [sm.update(*r) for r in zip(arrays["base_pos"], arrays["base_bearing"],
                                       arrays["q_mes"], arrays["qd_mes"])]
    out = dict(arrays)
    for i, key in enumerate(["base_pos", "base_bearing", "q_mes", "qd_mes"]):
        out[key] = np.array([r[i] for r in rows])
    return out